
import pandas as pd
import joblib
import json

from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

BASE_DIR = Path(__file__).resolve().parent.parent

# Load existing model
model = joblib.load(BASE_DIR / "model.joblib")

# Load incoming labelled data
data = pd.read_csv(BASE_DIR / "monitoring" / "new_batch.csv")

# Separate features and actual target
X = data.drop(columns=["math score"])
y_actual = data["math score"]

# Generate predictions
y_pred = model.predict(X)

# Calculate regression metrics
mae = mean_absolute_error(y_actual, y_pred)
mse = mean_squared_error(y_actual, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_actual, y_pred)

# Display results
print("EXISTING MODEL EVALUATION")
print("-------------------------")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

# Save evaluation report
report = {
    "model": "existing_champion",
    "MAE": round(mae, 4),
    "RMSE": round(rmse, 4),
    "R2": round(r2, 4)
}

report_path = BASE_DIR / "monitoring" / "evaluation_report.json"

with open(report_path, "w") as file:
    json.dump(report, file, indent=4)

print("\nEvaluation report saved successfully!")
