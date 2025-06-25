from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import os

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

def train_model(train_dataset, test_dataset, tokenizer, config):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pretrained_model = config["model"]["pretrained_model"]
    num_labels = len(set(train_dataset["label"]))

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=num_labels
    )

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        weight_decay=0.01,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("[INFO] Treinando modelo BERTimbau")
    trainer.train()

    # ✅ SALVAMENTO FINAL DO MODELO PARA INFERÊNCIA
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Salvando o modelo final para inferência em: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ✅ (Opcional) Salvar o caminho do melhor checkpoint
    if trainer.state.best_model_checkpoint:
        with open(os.path.join(output_dir, "best_checkpoint.txt"), "w") as f:
            f.write(trainer.state.best_model_checkpoint)

    return trainer