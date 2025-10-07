import os
import pytest
import numpy as np

@pytest.mark.skipif(
    not (
        os.path.exists("models/optimized/best_random_forest.pkl")
        or os.path.exists("models/random_forest.pkl")
    ),
    reason="Model file not found",
)
def test_model_load_and_predict():
    from model import IrisModel

    # Model nesnesi oluştur
    model = IrisModel()
    assert model.is_loaded, "Model could not be loaded."

    # Test verisi
    X = np.array([[5.1, 3.5, 1.4, 0.2]])

    # Tahmin yap
    pred, proba = model.predict(X)

    # Beklenen çıktı boyutları
    assert pred.shape == (1,), "Prediction output shape mismatch"
    assert proba.shape[0] == 1, "Probability output shape mismatch"
