import json
from pathlib import Path

import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

m = mlflow.sklearn.load_model("exported_model")
d = load_breast_cancer(as_frame=True)
Xdev, Xtest, ydev, ytest = train_test_split(
    d.data, d.target, test_size=0.2, random_state=42, stratify=d.target
)
pred = m.predict(Xtest)
r = {
    "test_accuracy": accuracy_score(ytest, pred),
    "test_f1_macro": f1_score(ytest, pred, average="macro"),
}
Path("test_metrics.json").write_text(json.dumps(r, indent=2))
print(r)
