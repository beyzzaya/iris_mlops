import argparse
import os
import subprocess
import sys
import time


def run_command(command, description):
    """Komut çalıştır ve sonucu raporla."""
    print(f"\n{'=' * 60}")
    print(f"🚀 {description}")
    print(f"{'=' * 60}")

    start_time = time.time()

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        end_time = time.time()

        print(f"✅ {description} başarıyla tamamlandı!")
        print(f"⏱️  Süre: {end_time - start_time:.2f} saniye")

        if result.stdout:
            print("📋 Çıktı:")
            print(result.stdout)

        return True

    except subprocess.CalledProcessError as e:
        end_time = time.time()
        print(f"❌ {description} başarısız!")
        print(f"⏱️  Süre: {end_time - start_time:.2f} saniye")
        print(f"💥 Hata: {e}")

        if e.stdout:
            print("📋 Stdout:")
            print(e.stdout)
        if e.stderr:
            print("📋 Stderr:")
            print(e.stderr)

        return False


def check_dependencies():
    """Gerekli bağımlılıkların yüklü olup olmadığını kontrol et."""
    print("🔍 Bağımlılıklar kontrol ediliyor...")

    required_packages = [
        "pandas",
        "numpy",
        "sklearn",
        "matplotlib",
        "seaborn",
        "joblib",
    ]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ Eksik paketler: {', '.join(missing_packages)}")
        print("📦 Şu komutu çalıştırın: pip install -r requirements.txt")
        return False

    print("✅ Tüm bağımlılıklar mevcut!")
    return True


def setup_directories():
    """Gerekli dizinleri oluştur."""
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "models/optimized",
        "görseller",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("📁 Dizinler hazırlandı!")


def run_full_pipeline():
    """Tam pipeline'ı çalıştır."""
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║                      İris ML PIPELINE                         ║
║              Model Development & Optimization                 ║
╚══════════════════════════════════════════════════════════════╝
"""
    )

    pipeline_start_time = time.time()

    if not check_dependencies():
        sys.exit(1)

    setup_directories()

    steps = [
        ("python src/1_data_indirme.py", "Veri İndirme"),
        ("python src/2_feature_engineering.py", "Özellik Mühendisliği"),
        ("python src/3_görselleştirme.py", "Görselleştirme"),
        ("python src/model_gelistirme.py", "Model Geliştirme"),
        (
            "python src/model_optimizasyon.py",
            "En İyi Model Detaylı Optimizasyonu",
        ),
    ]

    successful_steps = 0
    failed_steps = []

    for i, (command, description) in enumerate(steps, 1):
        print(f"\n📍 Adım {i}/{len(steps)}: {description}")

        if run_command(command, description):
            successful_steps += 1
        else:
            failed_steps.append(description)
            print(f"⚠️  {description} başarısız, devam ediliyor...")

    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time

    print(f"\n{'=' * 60}")
    print("📊 PIPELINE ÖZETİ")
    print(f"{'=' * 60}")
    print(f"✅ Başarılı adımlar: {successful_steps}/{len(steps)}")

    if failed_steps:
        print(f"❌ Başarısız adımlar: {', '.join(failed_steps)}")

    print(f"⏱️  Toplam süre: {total_time:.2f} saniye ({total_time/60:.1f} dk)")

    return successful_steps == len(steps)


def run_dvc_pipeline():
    """DVC pipeline'ını çalıştır."""
    print("\n🔄 DVC Pipeline çalıştırılıyor...")

    if run_command("dvc repro", "DVC Pipeline"):
        print("✅ DVC Pipeline başarıyla tamamlandı!")
        run_command("dvc status", "DVC Status Kontrolü")
        return True

    print("❌ DVC Pipeline başarısız!")
    return False


def main():
    """Ana fonksiyon."""
    parser = argparse.ArgumentParser(description="İris ML Pipeline Runner")
    parser.add_argument(
        "--dvc",
        action="store_true",
        help="DVC pipeline kullan (varsayılan: doğrudan Python)",
    )
    parser.add_argument(
        "--step",
        type=str,
        help="Sadece belirli bir adımı çalıştır (data, features, gorsellestirme, models, optimize)",
    )

    args = parser.parse_args()

    if args.step:
        step_commands = {
            "data": ("python src/1_data_indirme.py", "Veri İndirme"),
            "features": (
                "python src/2_feature_engineering.py",
                "Özellik Mühendisliği",
            ),
            "gorsellestirme": (
                "python src/3_görselleştirme.py",
                "Görselleştirme",
            ),
            "models": ("python src/model_gelistirme.py", "Model Geliştirme"),
            "optimize": (
                "python src/model_optimizasyon.py",
                "En İyi Model Detaylı Optimizasyonu",
            ),
        }

        if args.step in step_commands:
            command, description = step_commands[args.step]
            run_command(command, description)
        else:
            print(f"❌ Bilinmeyen adım: {args.step}")
            print(f"✅ Mevcut adımlar: {', '.join(step_commands.keys())}")

    elif args.dvc:
        run_dvc_pipeline()
    else:
        success = run_full_pipeline()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
