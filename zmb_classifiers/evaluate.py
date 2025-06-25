import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
from sklearn.metrics import classification_report
import yaml
import os
from datetime import datetime

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def preprocess_dataset(df, tokenizer, config):
    text_column = config["data"]["text_column"]
    target_column = config["data"]["target"]

    if df[target_column].dtype != int:
        df[target_column] = df[target_column].astype('category').cat.codes

    def tokenize(example):
        return tokenizer(
            example[text_column],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df, 
        test_size=config["training"]["test_size"], 
        random_state=config["training"]["random_state"]
    )
    test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)
    return test_df, test_dataset

def evaluate_model(config_path="config.yaml"):
    config = load_config(config_path)
    model_path = config["model"]["save_path"]

    print("[INFO] Carregando modelo e tokenizer...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_model"])

    print("[INFO] Lendo e preprocessando dados...")
    df = pd.read_csv(config["data"]["input"])
    test_texts_df, test_dataset = preprocess_dataset(df, tokenizer, config)

    print("[INFO] Avaliando modelo...")
    trainer = Trainer(model=model)
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids

    print("[INFO] Classification Report:")
    print(classification_report(labels, preds))

    print("[INFO] Salvando resultados detalhados...")
    result_df = pd.DataFrame({
        "texto": test_texts_df[config["data"]["text_column"]].values,
        "label_verdadeiro": labels,
        "label_predito": preds,
    })
    result_df["acertou"] = result_df["label_verdadeiro"] == result_df["label_predito"]

    os.makedirs("evaluation", exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_path = f"evaluation/resultados_avaliacao_{timestamp}.csv"
    
    result_df.to_csv(f"{output_path}", index=False)


    print(f"[INFO] Resultados salvos em {output_path}")


def main():
    evaluate_model()

if __name__ == "__main__":
    print("Iniciando avaliação")
    evaluate_model()