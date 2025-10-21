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
    from src.model import IrisModel  # dosya yoluna göre ayarla

    model = IrisModel()
    assert model.is_loaded, "Model could not be loaded."

    X = np.array([[5.1, 3.5, 1.4, 0.2]])

    pred, proba = model.predict(X)

    assert pred.shape == (1,), "Prediction output shape mismatch"
    assert proba.shape[0] == 1, "Probability output shape mismatch"
    assert proba.shape[1] >= 3, "Probability output class count mismatch"
    assert np.isclose(proba.sum(), 1.0, atol=1e-6), "Probabilities do not sum to 1"
