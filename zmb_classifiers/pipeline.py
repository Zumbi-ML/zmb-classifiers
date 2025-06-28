import pandas as pd
from zmb_classifiers.preprocess import preprocess_data
from zmb_classifiers.train import train_model
from zmb_classifiers.evaluate import evaluate_model

def run_pipeline(config):
    input_path = config["data"]["input"]
    print(f"[INFO] Lendo dados de {input_path}")
    df = pd.read_csv(input_path)

    print("[INFO] Pré-processando dados")
    train_dataset, test_dataset, tokenizer = preprocess_data(df, config["data"]["target"], config)

    print("[INFO] Treinando modelo BERTimbau")
    trainer = train_model(train_dataset, test_dataset, tokenizer, config)

    print("[INFO] Avaliando modelo")
    evaluate_model()

    print("[INFO] Salvando modelo")
    model_save_path = config["paths"]["best_model_dir"]
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)