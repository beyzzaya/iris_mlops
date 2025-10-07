from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np

from .model import IrisModel
from .util import get_class_names, ensure_feature_order  

app = FastAPI(title="Iris Classifier API", version="1.0.0")
model_wrapper = IrisModel()
class IrisInput(BaseModel):
    sepal_length: float = Field(..., ge=0)
    sepal_width: float = Field(..., ge=0)
    petal_length: float = Field(..., ge=0)
    petal_width: float = Field(..., ge=0)

class IrisBatchInput(BaseModel):
    items: List[IrisInput]

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_wrapper.is_loaded}


@app.post("/predict")
def predict(item: IrisInput):
    if not model_wrapper.is_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")
    X = ensure_feature_order([item.dict()])
    pred_idx, proba = model_wrapper.predict(X)
    classes = get_class_names()
    return {
        "class_index": int(pred_idx[0]),
        "class_name": classes[int(pred_idx[0])],
        "probabilities": proba[0].tolist(),
        "classes": classes,
    }
@app.post("/predict-batch")
def predict_batch(batch: IrisBatchInput):
    if not model_wrapper.is_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")
    X = ensure_feature_order([x.dict() for x in batch.items])
    pred_idx, proba = model_wrapper.predict(X)
    classes = get_class_names()
    results = []
    for i in range(len(pred_idx)):
        results.append({
        "class_index": int(pred_idx[i]),
        "class_name": classes[int(pred_idx[i])],
        "probabilities": proba[i].tolist(),
        })
    return {"classes": classes, "results": results}