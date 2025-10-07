from sklearn.datasets import load_iris
import os 


def main():
    print("------ 1.Veri İndirme ------")

    os.makedirs("data/raw", exist_ok=True)

    iris = load_iris(as_frame=True)
    iris = iris.frame
    iris.to_csv("data/raw/iris_raw.csv", index=False)

    print(f"iris dataset's shape: {iris.shape}")
    print(f"iris datasetinde  {iris.isna().sum().sum()} boş değer var.")
    print(f"iris datasetinin  columnları: {iris.columns}")


if __name__ == "__main__":
    main()