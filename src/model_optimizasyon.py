import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def main():
    print("------ model Optimizasyonu ------")

    if not os.path.exists("data/processed/iris_processed.csv"):
        print("İris dataseti bulunamadı.")
        return 
    
    df = pd.read_csv("data/processed/iris_processed.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 3, 5, 10],
        "min_samples_split": [2, 5, 10]
    }

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("✅ En iyi parametreler:", grid_search.best_params_)
    print("✅ En iyi skor:", grid_search.best_score_)

    os.makedirs("models/optimized", exist_ok=True)
    joblib.dump(grid_search.best_estimator_, "models/optimized/best_random_forest.pkl")
    print("🎯 Optimize edilmiş model kaydedildi!")

if __name__ == "__main__":
    main()