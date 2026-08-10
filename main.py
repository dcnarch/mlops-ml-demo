from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlruns_dir = Path.home() / "mlops-mlflow"
mlruns_dir.mkdir(parents=True, exist_ok=True)

# Set up MLflow to log locally in a guaranteed writable user-profile folder.
mlflow.set_tracking_uri(mlruns_dir.as_uri())
mlflow.set_experiment("RandomForest_Iris_Experiment")

# Load dataset
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define model parameters
n_estimators = 100
max_depth = 5

# Start MLflow run
with mlflow.start_run() as run:
    print("Run ID:", run.info.run_id)  # Useful to track run in UI

    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Make predictions and log metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    # Log and register the model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_model",
        registered_model_name="RandomForestClassifierModel"
    )

    print(f"Model logged and registered with accuracy: {accuracy:.4f}")
