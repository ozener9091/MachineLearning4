import json
import os
import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

MODEL_NAME = "breast-cancer-classifier"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
c = MlflowClient()
v = c.get_model_version_by_alias(MODEL_NAME, "champion")
print("champion version =", v.version)

model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{v.version}")

out = Path("exported_model")
if out.exists():
    shutil.rmtree(out)
mlflow.sklearn.save_model(
    model, str(out),
    pip_requirements=Path("requirements.txt").read_text().splitlines(),
)

d = load_breast_cancer(as_frame=True)
Xdev, Xtest, ydev, ytest = train_test_split(
    d.data, d.target, test_size=0.2, random_state=42, stratify=d.target
)

metadata = {
    "name": MODEL_NAME,
    "version": v.version,
    "run_id": v.run_id,
    "features": d.data.columns.tolist(),
    "classes": d.target_names.tolist(),
}
(out / "metadata.json").write_text(json.dumps(metadata, indent=2))

request = {"instances": Xtest.iloc[:3].to_dict(orient="records")}
Path("sample_request.json").write_text(json.dumps(request, indent=2))
Path("expected_predictions.json").write_text(json.dumps(model.predict(Xtest.iloc[:3]).tolist()))
print(json.dumps(metadata, indent=2))
