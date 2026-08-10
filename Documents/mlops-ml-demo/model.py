from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Save model to a writable location in the user profile
output_dir = Path.home() / "Documents" / "mlops-model-output"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "iris_model.pkl"
joblib.dump(clf, output_path)
print(f"Model trained and saved as {output_path}")
