iris_mlops/
│   README.md
│   requirements.txt
│   dvc.yaml
│   .gitignore
│   run_pipeline.py
│
├── .dvc/                 # DVC konfigürasyonları
│
├── görseller/            # EDA görselleri
│   ├── correlation_heatmap.png
│   ├── hist_plots.png
│   └── pairplot.png
│
│
├── models/               # Kaydedilen modeller
│
└── src/                  # Python scriptleri
    ├── 1_data_indirme.py
    ├── 2_feature_engineering.py
    ├── 3_görselleştirme.py
    ├── model_gelistirme.py
    └── model_optimizasyon.py













önce sanal env oluşturma: python -m venv irismlops
sonra env girme : irismlops\Scripts\activate
requirement.txt indirme: pip install -r requirement.txt

git başlat: git init
dvc başlat : dvc init

dosya oluşturma : mkdir src
dosyanın içine girme : cd src
dosyanın içinde .py file oluşturma: echo import pandas > 1_data_indirme.py

src dosysından çıkmak için : cd ..

src dosyasına girdikten sonra ve .py dosyasını oluşturduktan sonra git add yapıcaz: git add 1_data_indirme.py

sonra commit ekliyoruz: git commit -m "data indirme py oluşturuldu" 

dvc.yaml dosyamızı da oluşturmaya başlıyoruz:
echo > dvc.yaml

dvc yi de güncelledikçe ona da: git add dvc.yaml
daha sonra da : git commit -m "dvc güncellendi."

githuba bağlan : gh auth login
github repo oluştur : gh repo create iris_mlops --public --source=.
çalışmayı pushla : git push -u origin main




