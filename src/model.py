import os
import joblib
import numpy as np
from typing import Tuple, Optional, Union
import pandas as pd 


DEFAULT_OPT_MODEL_PATH = "models/optimized/best_random_forest.pkl"
DEFAULT_BASE_MODEL_PATH = "models/random_forest.pkl"


class IrisModel:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self._load()

    def __repr__(self) -> str:
        return f"IrisModel(is_loaded={self.is_loaded}, path={self.model_path!r})"

    def _load(self):
        candidates = []
        if self.model_path:
            candidates.append(self.model_path)
        candidates.extend([DEFAULT_OPT_MODEL_PATH, DEFAULT_BASE_MODEL_PATH])

        for path in candidates:
            if os.path.exists(path):
                try:
                    # mmap ile daha hızlı yükleme (büyük dosyada faydalı)
                    self.model = joblib.load(path, mmap_mode=None)
                    self.is_loaded = True
                    self.model_path = path
                    print(f"Model loaded from: {path}")
                    return
                except Exception as e:
                    print(f"Model load failed for {path}: {e}")
        print("No model file found. Please run training/optimization first.")

    def _ensure_ready(self):
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")

    def _to_ndarray(self, X: Union[np.ndarray, "pd.DataFrame", list]) -> np.ndarray:
        if pd is not None and isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if not np.issubdtype(X.dtype, np.number):
            X = X.astype(float)
        return X

    def predict(self, X: Union[np.ndarray, "pd.DataFrame", list]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            preds: shape (n_samples,)  -> model etiketleri (model.classes_ ile uyumlu)
            probas: shape (n_samples, n_classes) -> sütun sırası model.classes_ ile aynı
        """
        self._ensure_ready()
        X = self._to_ndarray(X)

        preds = self.model.predict(X)

        # Olasılıkları güvenli şekilde üret:
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X)
            # probas sütun sırası model.classes_ ile uyumlu
        else:
            # predict_proba yoksa one-hot tahmini üret, sınıf sırası model.classes_
            if hasattr(self.model, "classes_"):
                classes = np.array(self.model.classes_)
                class_to_idx = {c: i for i, c in enumerate(classes)}
                probas = np.zeros((len(preds), len(classes)), dtype=float)
                for i, p in enumerate(preds):
                    j = class_to_idx[p]
                    probas[i, j] = 1.0
            else:
                # En kötü senaryo: sınıf sayısını preds’ten tahmin et
                # (tavsiye edilmez ama çökermez)
                classes = np.unique(preds)
                class_to_idx = {c: i for i, c in enumerate(classes)}
                probas = np.zeros((len(preds), len(classes)), dtype=float)
                for i, p in enumerate(preds):
                    probas[i, class_to_idx[p]] = 1.0

        return preds, probas