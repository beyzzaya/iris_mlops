import os
import pytest
from fastapi.testclient import TestClient

# Model dosyası yoksa testleri atla (örneğin CI'da model eğitimi yapılmadıysa)
model_exists = (
    os.path.exists("models/optimized/best_random_forest.pkl")
    or os.path.exists("models/random_forest.pkl")
)

@pytest.mark.skipif(not model_exists, reason="Model file not found")
def test_predict_endpoint():
    from app import app  # FastAPI instance importu
    client = TestClient(app)

    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    # Beklenen çıktı yapısı
    assert "class_name" in data
    assert isinstance(data["probabilities"], list)
    assert len(data["probabilities"]) >= 3

