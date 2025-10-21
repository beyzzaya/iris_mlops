from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np

# Bağımlılıkları içe aktar (model.py ve util.py'nin yan yana olması gerekir)
from .model import IrisModel
from .util import get_class_names, ensure_feature_order 

# FastAPI uygulamasını başlat
app = FastAPI(title="Iris Sınıflandırıcı API", version="1.0.0")

# Modeli yükle (model.py'deki mantık, dosya yoksa modeli otomatik olarak eğitir)
model_wrapper = IrisModel()

# --- Pydantic Girdi Şemaları ---

class IrisInput(BaseModel):
    """Tek bir iris gözlemi için girdi verisi şeması."""
    sepal_length: float = Field(..., ge=0, description="Çanak yaprak uzunluğu (cm)")
    sepal_width: float  = Field(..., ge=0, description="Çanak yaprak genişliği (cm)")
    petal_length: float = Field(..., ge=0, description="Taç yaprak uzunluğu (cm)")
    petal_width: float  = Field(..., ge=0, description="Taç yaprak genişliği (cm)")

class IrisBatchInput(BaseModel):
    """Toplu tahmin için birden fazla gözlem listesi."""
    items: List[IrisInput]

# --- Yardımcı Fonksiyonlar ---

def _resolve_class_names_for_model():
    """
    Modelin `classes_` özniteliğini kullanarak, olasılık dizisindeki
    sütun sırasına karşılık gelen okunabilir sınıf adlarını döndürür.
    """
    base_names = get_class_names()
    classes = getattr(model_wrapper.model, "classes_", None)
    
    if classes is None:
        return base_names

    classes_arr = np.array(classes)
    
    if np.issubdtype(classes_arr.dtype, np.integer):
        return [base_names[int(c)] for c in classes_arr]
    else:
        return list(map(str, classes_arr))

def _pred_label_to_name(label):
    """Tek bir tahmin etiketini (int veya str) okunabilir ada çevirir."""
    base_names = get_class_names()
    # Eğer etiket bir sayı ise, onu int'e dönüştür (NumPy türlerini önlemek için)
    if isinstance(label, (int, np.integer)):
        label = int(label)
        if 0 <= label < len(base_names):
            return base_names[label]
    return str(label)

# --- API Uç Noktaları ---

@app.get("/health", summary="API ve model sağlığını kontrol et")
def health():
    """Modelin yüklü olup olmadığını gösteren basit sağlık kontrolü."""
    return {"status": "ok", "model_loaded": model_wrapper.is_loaded}

@app.post("/predict", summary="Tek bir iris gözlemi için tahmin yap")
def predict(item: IrisInput):
    """Tek bir iris gözlemi için sınıf etiketi ve olasılıklarını döndürür."""
    if not model_wrapper.is_loaded:
        raise HTTPException(status_code=500, detail="Tahmin modeli yüklenmedi.")

    input_data = item.model_dump() if hasattr(item, 'model_dump') else item.dict()
    X = ensure_feature_order([input_data])
    
    pred_labels, proba = model_wrapper.predict(X)

    # NumPy int/float türlerini Python yerel türlerine dönüştür.
    pred_label = str(pred_labels[0]) if isinstance(pred_labels[0], (str, np.str_)) else int(pred_labels[0])
    
    classes_readable = _resolve_class_names_for_model()
    readable_class_name = _pred_label_to_name(pred_label)
    
    # KRİTİK DÜZELTME: Olasılıkları testin beklediği gibi sadece liste olarak döndür.
    probabilities = proba[0].tolist()
    
    return {
        "class_label": pred_label,                    
        "class_name": readable_class_name,            
        "probabilities": probabilities,               # <<< SADECE LİSTE
        "classes": classes_readable                   # <<< TESTE UYGUN ANAHTAR ADI
    }

@app.post("/predict-batch", summary="Birden fazla iris gözlemi için toplu tahmin yap")
def predict_batch(batch: IrisBatchInput):
    """Toplu iris gözlemleri için tahmin etiketleri ve olasılıkları döndürür."""
    if not model_wrapper.is_loaded:
        raise HTTPException(status_code=500, detail="Tahmin modeli yüklenmedi.")
    if not batch.items:
        raise HTTPException(status_code=400, detail="Girdi listesi boş olamaz.")

    item_dicts = [x.model_dump() for x in batch.items] if hasattr(batch.items[0], 'model_dump') else [x.dict() for x in batch.items]
    X = ensure_feature_order(item_dicts)
    
    pred_labels, proba = model_wrapper.predict(X)
    classes_readable = _resolve_class_names_for_model()

    results = []
    for i, label in enumerate(pred_labels):
        
        # Her etiketi Python yerel türüne dönüştür.
        converted_label = str(label) if isinstance(label, (str, np.str_)) else int(label)

        # Batch sonucunda da olasılıkları liste olarak döndür (tekli endpoint ile tutarlılık için)
        probabilities_list = proba[i].tolist()
        
        results.append({
            "class_label": converted_label,
            "class_name": _pred_label_to_name(converted_label), 
            "probabilities": probabilities_list,
        })
    
    return {"classes": classes_readable, "results": results}
