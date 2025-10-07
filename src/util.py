from typing import List, Dict
import numpy as np
from sklearn.datasets import load_iris


_FEATURE_ORDER = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

def get_class_names() -> List[str]:
    data = load_iris()
    return list(data.target_names)



def ensure_feature_order(records: List[Dict]) -> np.ndarray:
    """Gelen dict listelerini doğru kolon sırasına koyup np.array döndürür."""
    matrix = []
    for r in records:
        row = [float(r[k]) for k in _FEATURE_ORDER]
        matrix.append(row)
    return np.array(matrix, dtype=float)
