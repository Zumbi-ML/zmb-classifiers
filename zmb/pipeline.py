import pandas as pd
import yaml
import os
from zmb.preprocess import preprocess_data
from zmb.train import train_model
from zmb.evaluate import evaluate_model

def run_pipeline(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    input_path = config["data"]["input"]
    print(f"[INFO] Lendo dados de {input_path}")
    df = pd.read_csv(input_path)

    print("[INFO] Pré-processando dados")
    train_dataset, test_dataset, tokenizer = preprocess_data(df, config["data"]["target"], config)

    print("[INFO] Treinando modelo BERTimbau")
    trainer = train_model(train_dataset, test_dataset, tokenizer, config)

    print("[INFO] Avaliando modelo")
    evaluate_model(trainer, test_dataset)

    print("[INFO] Salvando modelo")
    trainer.save_model(config["model"]["save_path"])
    tokenizer.save_pretrained(config["model"]["save_path"])