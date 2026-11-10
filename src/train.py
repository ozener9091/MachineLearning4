import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAME = "breast-cancer-classifier"
EXPERIMENT = "lab4-breast-cancer"

p = argparse.ArgumentParser()
p.add_argument("--algorithm", choices=["lr", "rf"], default="lr")
p.add_argument("--c", type=float, default=1.0)
p.add_argument("--trees", type=int, default=100)
p.add_argument("--depth", type=int, default=5)
p.add_argument("--register", action="store_true")
a = p.parse_args()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment(EXPERIMENT)

d = load_breast_cancer(as_frame=True)
X, y = d.data, d.target

# 60/20/20; final test set is NOT used in this script.
Xdev, Xtest, ydev, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
Xtrain, Xval, ytrain, yval = train_test_split(Xdev, ydev, test_size=0.25, random_state=42, stratify=ydev)

if a.algorithm == "lr":
    model = make_pipeline(StandardScaler(), LogisticRegression(C=a.c, max_iter=5000, random_state=42))
else:
    model = RandomForestClassifier(n_estimators=a.trees, max_depth=a.depth, random_state=42, n_jobs=1)

digest = hashlib.sha256(d.frame.to_csv(index=False).encode()).hexdigest()

with mlflow.start_run(run_name=f"{a.algorithm}-experiment") as run:
    mlflow.log_params({
        "algorithm": a.algorithm,
        "seed": 42,
        "dataset": "sklearn.load_breast_cancer",
        "n_train": len(Xtrain),
        "n_validation": len(Xval),
        "n_test": len(Xtest),
    })
    if a.algorithm == "lr":
        mlflow.log_params({"C": a.c})
    else:
        mlflow.log_params({"trees": a.trees, "depth": a.depth})

    mlflow.set_tags({
        "dataset_sha256": digest,
        "purpose": "lab4",
        "selection_metric": "val_f1_macro",
    })

    model.fit(Xtrain, ytrain)
    pred = model.predict(Xval)

    mlflow.log_metrics({
        "val_accuracy": accuracy_score(yval, pred),
        "val_f1_macro": f1_score(yval, pred, average="macro"),
    })

    mlflow.log_input(
        mlflow.data.from_pandas(
            d.frame,
            source="https://scikit-learn.org/stable/datasets/toy_dataset.html",
            targets="target",
        ),
        context="full_dataset_before_split",
    )

    mlflow.log_dict(
        {
            "train_indices": Xtrain.index.tolist(),
            "validation_indices": Xval.index.tolist(),
            "test_indices": Xtest.index.tolist(),
            "columns": X.columns.tolist(),
            "target_names": d.target_names.tolist(),
        },
        "split.json",
    )

    mlflow.log_artifact(__file__, artifact_path="source")
    mlflow.log_artifact("requirements.txt", artifact_path="source")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(yval, pred, ax=ax)
    mlflow.log_figure(fig, "validation_confusion_matrix.png")
    plt.close(fig)

    info = mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=Xtrain.head(2),
        signature=infer_signature(Xtrain, model.predict(Xtrain)),
        pip_requirements=Path("requirements.txt").read_text().splitlines(),
    )
    print(json.dumps({"run_id": run.info.run_id, "model_uri": info.model_uri}, indent=2))

    if a.register:
        mv = mlflow.register_model(info.model_uri, MODEL_NAME)
        print("REGISTERED_VERSION=", mv.version)
