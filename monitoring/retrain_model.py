
import pandas as pd
import joblib
import json

from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

BASE_DIR = Path(__file__).resolve().parent.parent
MONITOR_DIR = BASE_DIR / "monitoring"

# Load existing training data and incoming batch
train_data = pd.read_csv(MONITOR_DIR / "train_data.csv")
new_batch = pd.read_csv(MONITOR_DIR / "new_batch.csv")

# Keep some incoming records separate for fair validation
new_train, validation_data = train_test_split(
    new_batch,
    test_size=0.20,
    random_state=42
)

# Combine existing training data with incoming training records
combined_data = pd.concat(
    [train_data, new_train],
    ignore_index=True
)

X_train = combined_data.drop(columns=["math score"])
y_train = combined_data["math score"]

X_val = validation_data.drop(columns=["math score"])
y_val = validation_data["math score"]

print("RETRAINING DATA")
print("----------------")
print("Existing training records:", len(train_data))
print("Incoming training records:", len(new_train))
print("Combined training records:", len(combined_data))
print("Validation records:", len(validation_data))

# Load existing model and clone its preprocessing pipeline
champion = joblib.load(BASE_DIR / "model.joblib")
candidate = clone(champion)

# Train candidate model
candidate.fit(X_train, y_train)

# Save candidate separately
candidate_path = MONITOR_DIR / "candidate_model.joblib"
joblib.dump(candidate, candidate_path)

# Evaluate both models on the same unseen validation records
champion_predictions = champion.predict(X_val)
candidate_predictions = candidate.predict(X_val)

def calculate_metrics(actual, predicted):
    return {
        "MAE": round(mean_absolute_error(actual, predicted), 4),
        "RMSE": round(sqrt(mean_squared_error(actual, predicted)), 4),
        "R2": round(r2_score(actual, predicted), 4)
    }

champion_metrics = calculate_metrics(y_val, champion_predictions)
candidate_metrics = calculate_metrics(y_val, candidate_predictions)

print("\nCHAMPION MODEL")
print(champion_metrics)

print("\nCANDIDATE MODEL")
print(candidate_metrics)

# Save comparison results for Task D
comparison = {
    "champion": champion_metrics,
    "candidate": candidate_metrics
}

with open(MONITOR_DIR / "model_comparison.json", "w") as file:
    json.dump(comparison, file, indent=4)

print("\nCandidate model saved:", candidate_path)
print("Comparison report saved successfully!")
print("\nTask C completed successfully!")
