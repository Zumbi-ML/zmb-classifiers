import os
import shutil
import subprocess
from pathlib import Path

RELEASE_DIR = Path("releases")
TEMP_DIRS = ["build", "dist", "zmb_classifiers.egg-info"]

def clean_temp_dirs():
    for folder in TEMP_DIRS:
        path = Path(folder)
        if path.exists():
            print(f"[INFO] Removendo diretório temporário: {folder}")
            shutil.rmtree(path)

def prepare_release_dir():
    if RELEASE_DIR.exists():
        print("[INFO] Limpando diretório releases/")
        for f in RELEASE_DIR.iterdir():
            if f.is_file():
                f.unlink()
    else:
        print("[INFO] Criando diretório releases/")
        RELEASE_DIR.mkdir()

def build_package():
    print("[INFO] Construindo pacote...")
    subprocess.run(["python", "setup.py", "sdist", "bdist_wheel", "--dist-dir", str(RELEASE_DIR)], check=True)

def main():
    print("=== BUILD DE RELEASE ZMB ===")
    clean_temp_dirs()
    prepare_release_dir()
    build_package()
    clean_temp_dirs() 
    print("✅ Build concluído. Artefatos disponíveis em releases/")

if __name__ == "__main__":
    main()