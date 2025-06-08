"""
Script de fine-tuning do BERTimbau para classificação binária.

Atualizações:
- Corrige warnings de depreciação do transformers
- Adiciona early stopping
- Melhora tratamento de dados

Uso:
python src/train.py \
    --dataset-path data/03-ready_4_training/classifier-dataset.csv \
    --output-dir models/classifier_v1 \
    --num-epochs 5 \
    --learning-rate 2e-5 \
    --batch-size 8
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict

# Configurações padrão
DEFAULT_MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
DEFAULT_DATASET_PATH = "data/03-ready_4_training/classifier-dataset.csv"
DEFAULT_OUTPUT_DIR = "models/bertimbau_finetuned"
DEFAULT_NUM_LABELS = 2
DEFAULT_NUM_EPOCHS = 5
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_BATCH_SIZE = 8
DEFAULT_EARLY_STOPPING_PATIENCE = 2

def compute_metrics(eval_pred):
    """Calcula métricas de avaliação."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def load_and_prepare_data(dataset_path, test_size=0.1):
    """Carrega e prepara os dados para treinamento."""
    df = pd.read_csv(dataset_path, quotechar='"')
    
    # Divisão estratificada
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=test_size,
        random_state=42,
        stratify=df['label']
    )
    
    return DatasetDict({
        'train': Dataset.from_dict({'text': train_texts, 'label': train_labels}),
        'validation': Dataset.from_dict({'text': val_texts, 'label': val_labels})
    })

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning do BERTimbau para classificação binária")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--early-stopping-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    args = parser.parse_args()

    # 1. Carregar e preparar dados
    dataset = load_and_prepare_data(args.dataset_path)
    
    # 2. Inicializar tokenizer e model
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=DEFAULT_NUM_LABELS
    )

    # 3. Tokenização dos dados
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4. Configurar treinamento
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir="./logs",
        fp16=True,
        report_to="none"
    )

    # 5. Criar e executar trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )

    # 6. Treinar e salvar modelo
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"✅ Modelo treinado e salvo em {args.output_dir}")

if __name__ == "__main__":
    main()