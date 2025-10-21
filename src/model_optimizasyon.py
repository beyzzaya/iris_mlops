import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("------ Model Optimizasyonu ------")

    data_path = os.path.join("data", "raw", "iris_raw.csv")
    if not os.path.exists(data_path):
        print("İris dataseti bulunamadı.")
        return

    df = pd.read_csv(data_path)
    if "target" not in df.columns:
        print("target kolonu bulunamadı.")
        return

    X = df.drop(columns=["target"]).select_dtypes(include="number")
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 3, 5, 10],
        "min_samples_split": [2, 5, 10],
        # İstersen şunu da ekleyebilirsin:
        # "max_features": ["sqrt", "log2", None],
    }

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",        # alternatif: "f1_macro"
        refit="accuracy",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
        error_score="raise"
    )

    grid_search.fit(X_train, y_train)

    print("En iyi parametreler:", grid_search.best_params_)
    print(f"CV En iyi accuracy: {grid_search.best_score_:.4f}")

    # Test seti performansı
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Sonuçları kaydet
    os.makedirs(os.path.join("models", "optimized"), exist_ok=True)
    joblib.dump(best_model, os.path.join("models", "optimized", "best_random_forest.pkl"))
    print("Optimize edilmiş model kaydedildi!")

    # CV sonuçlarını CSV’ye yaz
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_path = os.path.join("models", "optimized", "rf_grid_cv_results.csv")
    cv_results.to_csv(cv_path, index=False)
    print(f"CV sonuçları kaydedildi: {cv_path}")

if __name__ == "__main__":
    main()
