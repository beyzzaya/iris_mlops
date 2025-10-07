import os
import joblib
import numpy as np
from typing import Tuple


DEFAULT_OPT_MODEL_PATH = "models/optimized/best_random_forest.pkl"
DEFAULT_BASE_MODEL_PATH = "models/random_forest.pkl"


class IrisModel:
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self._load()


    def _load(self):
        candidates = []
        if self.model_path:
            candidates.append(self.model_path)
        candidates.extend([DEFAULT_OPT_MODEL_PATH, DEFAULT_BASE_MODEL_PATH])


        for path in candidates:
            if os.path.exists(path):
                try:
                    self.model = joblib.load(path)
                    self.is_loaded = True
                    self.model_path = path
                    print(f"Model loaded from: {path}")
                    return
                except Exception as e:
                    print(f"Model load failed for {path}: {e}")
        print("No model file found. Please run training/optimization first.")


    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        preds = self.model.predict(X)
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X)
        else:
            # Eğer predict_proba yoksa, tek-sıcak yaklaşık dağılım döndür
            n_classes = len(set(preds))
            probas = np.zeros((len(preds), n_classes))
            for i, p in enumerate(preds):
                probas[i, int(p)] = 1.0
        return preds, probas