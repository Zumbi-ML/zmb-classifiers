import os
import shutil
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import subprocess

from zmb_classifiers.config import CONFIG

def clean_or_create_publish_dir(publish_dir):
    if os.path.exists(publish_dir):
        shutil.rmtree(publish_dir)
    os.makedirs(publish_dir, exist_ok=True)


def convert_model(source_dir, publish_dir, serialization_format):
    print(f"[1] Carregando modelo salvo como {serialization_format.upper()}")

    model = AutoModelForSequenceClassification.from_pretrained(
        source_dir,
        trust_remote_code=True
    )
    print(f"[2] Salvando modelo no formato {serialization_format.upper()}...")
    model.save_pretrained(
        publish_dir,
        safe_serialization=(serialization_format == "safetensors")
    )

    print("[3] Salvando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(source_dir)
    tokenizer.save_pretrained(publish_dir)


def upload_to_huggingface(publish_dir, hf_repo_id):
    print("[4] Enviando para o Hugging Face com huggingface-cli upload...")
    result = subprocess.run([
        "huggingface-cli", "upload",
        hf_repo_id,
        publish_dir,
        "--include", "*",
        "--repo-type", "model"
    ])

    if result.returncode == 0:
        print("✅ Upload concluído com sucesso.")
    else:
        print("❌ Falha no upload. Verifique se você está autenticado com `huggingface-cli login`.")

def main():
    source_model_dir = CONFIG["paths"]["best_model_dir"]
    publish_dir = CONFIG["paths"]["publish_dir"]
    serialization_format = CONFIG["model"].get("serialization_format", "bin").lower()
    hf_repo_id = CONFIG["model"]["hf_repo_id"]

    print("=== PUBLICADOR DE MODELO ZMB ===")
    clean_or_create_publish_dir(publish_dir)
    convert_model(source_model_dir, publish_dir, serialization_format)
    upload_to_huggingface(publish_dir, hf_repo_id)


if __name__ == "__main__":
    main()