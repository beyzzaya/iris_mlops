# tests/conftest.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # proje kökü (....\iris_mlops)
# ÖNEMLİ: kökü ekliyoruz, 'src' klasörünü değil.
sys.path.insert(0, str(ROOT))

