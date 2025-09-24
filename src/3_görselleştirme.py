import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import os

def main():
    print("------ 3 Görselleştirme ------")
    if not os.path.exists("data/processed/iris_processed.csv"):
        print("dataseti bulunamadı.")
        return 
    
    df = pd.read_csv("data/processed/iris_processed.csv")
    print("iris dataseti yüklendi.")
    os.makedirs("görseller", exist_ok=True)
    print("görseller adında dosya oluşturuldu.")

    print("---hist plotlar oluşturuluyor ---")
    df.hist(figsize=(10,8))
    plt.tight_layout()
    plt.savefig("görseller/hist_plots.png")
    plt.close()
    print("--- Histogramlar kaydedildi. ---")

    print("--- pairplot oluşturuluyor ---")
    sns.pairplot(df, hue="target", diag_kind="hist")
    plt.savefig("görseller/pairplot.png")
    plt.close()
    print("Pairplot kaydedildi.")

    # 3. Korelasyon heatmap
    print("--- korelasyon heatmap oluşturuluyor ---")
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Korelasyon Matrisi")
    plt.savefig("görseller/correlation_heatmap.png")
    plt.close()
    print("Korelasyon heatmap kaydedildi.")

    print(" Tüm görseller başarıyla oluşturuldu ve kaydedildi.")

if __name__ == "__main__":
    main()

