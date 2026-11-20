
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "StudentsPerformance.csv"
OUTPUT_DIR = BASE_DIR / "monitoring"

# Load original dataset
df = pd.read_csv(DATA_PATH)

print("Original Dataset Shape:", df.shape)

# Required columns
required_columns = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
    "math score",
    "reading score",
    "writing score"
]

# Validate columns
missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

print("\nColumn validation: PASSED")

# Check missing values
print("\nMissing Values:")
print(df[required_columns].isnull().sum())

if df[required_columns].isnull().any().any():
    raise ValueError("Dataset contains missing values!")

print("\nMissing value validation: PASSED")

# Check duplicate records
duplicates = df.duplicated().sum()
print("\nDuplicate Records:", duplicates)

# Remove duplicates if present
df = df.drop_duplicates().reset_index(drop=True)

print("Dataset after duplicate removal:", df.shape)

# Split into existing training data and incoming batch
train_data, new_batch = train_test_split(
    df,
    test_size=0.20,
    random_state=42
)

# Save incoming labelled batch
new_batch.to_csv(OUTPUT_DIR / "new_batch.csv", index=False)
train_data.to_csv(OUTPUT_DIR / "train_data.csv", index=False)
print("\nExisting Training Data:", train_data.shape)
print("Incoming New Batch:", new_batch.shape)

# Regression target summary
print("\nIncoming Batch Math Score Summary:")
print(new_batch["math score"].describe())

print("\nTask A completed successfully!")
