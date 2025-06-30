import os
import torch
from transformers import EarlyStoppingCallback
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score
from zmb_classifiers.config import CONFIG

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

def train_model(train_dataset, test_dataset, tokenizer, config):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    pretrained_model = config["model"]["base_model"]
    num_labels = len(set(train_dataset["label"]))

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model,
            num_labels=num_labels,
            local_files_only=True
        )
    except:
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model,
            num_labels=num_labels
        )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=CONFIG["paths"]["checkpoints_dir"],
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        weight_decay=0.01,
        logging_dir=CONFIG["paths"]["logs_dir"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        dataloader_pin_memory=False,
        dataloader_num_workers=2,
        gradient_accumulation_steps=2,
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("[INFO] Treinando modelo BERTimbau")
    trainer.train()

    return trainer