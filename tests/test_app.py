import os
import pytest
from fastapi.testclient import TestClient
import math
import sys

# Model dosyası yoksa testleri atla (örneğin CI'da model eğitimi yapılmadıysa)
model_exists = (
    os.path.exists("models/optimized/best_random_forest.pkl")
    or os.path.exists("models/random_forest.pkl")
)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

@pytest.mark.skipif(not model_exists, reason="Model file not found")
def test_predict_endpoint():
    from src.app import app 
    client = TestClient(app)

    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    # Beklenen yapı
    assert "class_name" in data
    assert "probabilities" in data
    assert "classes" in data

    assert isinstance(data["probabilities"], list)
    assert isinstance(data["classes"], list)
    assert len(data["probabilities"]) == len(data["classes"]) >= 3

    # Olasılıklar 0-1 aralığında ve ~1'e toplam
    assert all(0.0 <= p <= 1.0 for p in data["probabilities"])
    assert math.isclose(sum(data["probabilities"]), 1.0, rel_tol=1e-6, abs_tol=1e-6)

