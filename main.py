
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Student Performance Prediction API")

MODEL_PATH = Path(__file__).parent / "model.joblib"
model = joblib.load(MODEL_PATH)


class StudentData(BaseModel):
    gender: str
    race_ethnicity: str = Field(alias="race/ethnicity")
    parental_education: str = Field(
        alias="parental level of education"
    )
    lunch: str
    test_preparation: str = Field(
        alias="test preparation course"
    )
    reading_score: int = Field(ge=0, le=100)
    writing_score: int = Field(ge=0, le=100)


@app.get("/")
def home():
    return {"message": "Student Performance API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(student: StudentData):
    features = pd.DataFrame([{
        "gender": student.gender,
        "race/ethnicity": student.race_ethnicity,
        "parental level of education": student.parental_education,
        "lunch": student.lunch,
        "test preparation course": student.test_preparation,
        "reading score": student.reading_score,
        "writing score": student.writing_score
    }])

    prediction = model.predict(features)[0]

    return {
        "predicted_math_score": round(float(prediction), 2)
    }
