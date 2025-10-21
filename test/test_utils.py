from utils import ensure_feature_order, get_class_names

def test_ensure_feature_order():
    # Girdi: bir örnek dictionary listesi
    X = ensure_feature_order([
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    ])

    # Beklenen: 1 örnek, 4 özellik
    assert X.shape == (1, 4), "Feature order output shape should be (1, 4)"

def test_get_class_names():
    # Sınıf isimlerini al
    names = get_class_names()

    # Beklenen: Iris datasetinde 3 sınıf
    assert isinstance(names, list), "Class names should be returned as a list"
    assert len(names) == 3, "There should be 3 class names"
