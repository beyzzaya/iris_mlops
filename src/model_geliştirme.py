import pandas as pd
import os 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


def main():
    print("------ Model Geliştirme ------")

    if not os.path.exists("data/processed/iris_processed.csv"):
        print("iris dataseti bulunamadı.")
        return 
    
    df = pd.read_csv("data/processed/iris_processed.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

    os.makedirs("models", exist_ok=True)
    print("models adlı dosya oluşturuldu.")

    log_reg = LogisticRegression(max_iter=200)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
    joblib.dump(log_reg, "models/logistic_regression.pkl")
    print("logistic regression modeli models adlı dosyaya pkl olarak kaydedildi. ")


    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    joblib.dump(rf, "models/random_forest.pkl")
    print("Random Forest Classification modeli models dosyasına pkl olarak kaydedildi.")

    print("Tüm modeller kaydedildi!")

    
if __name__ =="__main__":
    main()

