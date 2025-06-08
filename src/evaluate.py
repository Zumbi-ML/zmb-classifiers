"""
Avaliação de um modelo Bertimbau fine-tuned usando um dataset de teste.

Exemplos de uso:

# Avaliar o modelo padrão no dataset padrão
python scripts/evaluate.py

# Avaliar um outro modelo em um novo dataset
python scripts/evaluate.py --model-dir models/bertimbau_finetuned --test-dataset data/processed/novo_dataset.csv
"""

import pandas as pd
import torch
import argparse
import os
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Configurações padrão
DEFAULT_MODEL_DIR = "models/bertimbau_finetuned"
DEFAULT_TEST_DATASET_PATH = "data/processed/dataset_v1.csv"
DEFAULT_BATCH_SIZE = 16

def main():
    # Parser de argumentos para customização
    parser = argparse.ArgumentParser(description="Avaliação de modelo Bertimbau fine-tuned")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR, help="Diretório do modelo salvo")
    parser.add_argument("--test-dataset", type=str, default=DEFAULT_TEST_DATASET_PATH, help="Caminho para o dataset de teste (CSV)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Tamanho do batch para avaliação")
    args = parser.parse_args()

    # Verifica se o diretório do modelo existe
    if not os.path.exists(args.model_dir):
        raise ValueError(f"❌ Diretório do modelo '{args.model_dir}' não encontrado. Verifique o caminho informado.")

    # 1. Carregar o modelo e o tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()  # Modo avaliação

    # 2. Carregar o dataset de teste
    df = pd.read_csv(args.test_dataset, quotechar='"')

    # 3. Preparar o Dataset da Hugging Face
    test_dataset = Dataset.from_dict({
        'text': df['text'].tolist(),
        'label': df['label'].tolist()
    })

    # 4. Tokenizar o dataset
    def tokenize_function(example):
        return tokenizer(example['text'], padding="max_length", truncation=True, max_length=512)

    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    # 5. Preparar o DataLoader manualmente
    test_loader = torch.utils.data.DataLoader(
        tokenized_test_dataset.with_format("torch"),
        batch_size=args.batch_size
    )

    # 6. Definir o dispositivo (GPU ou CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 7. Loop de avaliação
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_predictions.extend(preds.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    # 8. Calcular métricas
    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, average='binary')

    # 9. Mostrar os resultados
    print("\n📊 Resultados da Avaliação:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

if __name__ == "__main__":
    main()