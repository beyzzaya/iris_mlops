import os
import pandas as pd


def main():
    print("------ 2- fature engineering ------")
    if not os.path.exists("data/raw/iris_raw.csv"):
        print("iris dataseti bulunamadı.")
        return 
    
    df = pd.read_csv("data/raw/iris_raw.csv")
    iris = df.copy()

    iris["petal_ratio"] = iris["petal length (cm)"] - iris["petal width (cm)"]
    iris["sepal_petal_mul"] = iris["sepal length (cm)"] - iris["petal length (cm)"]


    print("Oran özelliği ve Polynomial feature özellikleri eklendi. ")

    os.makedirs("data/processed", exist_ok=True)
    print("data/processed dosyası oluşturuldu.")
    iris.to_csv("data/processed/iris_processed.csv", index=False)
    print("oluşturulan yeni featurelar data/processed/iris_processed.csv yoluna kaydedildi. ")

    print(f"iris datasının yeni shape: {iris.shape}")


if __name__ == "__main__":
    main()


