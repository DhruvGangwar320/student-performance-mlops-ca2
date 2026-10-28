
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict():
    response = client.post("/predict", json={
        "gender": "female",
        "race/ethnicity": "group B",
        "parental level of education": "bachelor's degree",
        "lunch": "standard",
        "test preparation course": "none",
        "reading_score": 72,
        "writing_score": 74
    })
    assert response.status_code == 200
    assert "predicted_math_score" in response.json()
