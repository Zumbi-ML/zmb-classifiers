import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
from sklearn.metrics import classification_report
from datetime import datetime
import os

from zmb_classifiers.config import CONFIG

def preprocess_dataset(df, tokenizer, config):
    text_column = config["data"]["text_column"]
    target_column = config["data"]["target"]

    if text_column not in df.columns:
        raise ValueError(f"Coluna '{text_column}' não encontrada no DataFrame")
    if target_column not in df.columns:
        raise ValueError(f"Coluna '{target_column}' não encontrada no DataFrame")

    df = df.dropna(subset=[text_column]).copy()
    df[text_column] = df[text_column].astype(str)

    if df[target_column].dtype not in ['int64', 'int32']:
        df[target_column] = df[target_column].astype('category').cat.codes

    def tokenize(examples):
        texts = [str(text) for text in examples[text_column]]
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=config["model"].get("max_length", 512)
        )

    test_dataset = Dataset.from_pandas(df).map(tokenize, batched=True)
    return df, test_dataset

def evaluate_model():
    model_path = CONFIG["paths"]["best_model_dir"]

    print("[INFO] Carregando modelo e tokenizer...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"[ERRO] Falha ao carregar modelo/tokenizer: {e}")
        raise

    print("[INFO] Lendo e preprocessando dados...")
    try:
        df = pd.read_csv(CONFIG["data"]["input"])
        test_texts_df, test_dataset = preprocess_dataset(df, tokenizer, CONFIG)
    except Exception as e:
        print(f"[ERRO] Falha no pré-processamento: {e}")
        raise

    print("[INFO] Avaliando modelo...")
    try:
        trainer = Trainer(model=model)
        predictions = trainer.predict(test_dataset)
        preds = predictions.predictions.argmax(-1)
        labels = predictions.label_ids

        print("\n[INFO] Classification Report:")
        print(classification_report(labels, preds, digits=4))
    except Exception as e:
        print(f"[ERRO] Falha na avaliação: {e}")
        raise

    print("[INFO] Salvando resultados detalhados...")
    try:
        result_df = pd.DataFrame({
            "texto": test_texts_df[CONFIG["data"]["text_column"]].values,
            "label_verdadeiro": labels,
            "label_predito": preds,
        })
        result_df["acertou"] = result_df["label_verdadeiro"] == result_df["label_predito"]

        from pathlib import Path
        eval_dir = Path(CONFIG["paths"]["evaluation_dir"])
        eval_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = eval_dir / f"resultados_avaliacao_{timestamp}.csv"
        result_df.to_csv(output_path, index=False)
        print(f"[INFO] Resultados salvos em {output_path}")
    except Exception as e:
        print(f"[ERRO] Falha ao salvar resultados: {e}")
        raise

if __name__ == "__main__":
    print("Iniciando avaliação")
    evaluate_model()