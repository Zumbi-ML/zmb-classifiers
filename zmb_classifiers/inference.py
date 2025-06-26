import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from huggingface_hub import snapshot_download

class ZmbClassifier:
    def __init__(self, model_path=None):
        if model_path is None:
            # Nome do repositório no Hugging Face
            hf_repo = "j3ffsilva/zmb-classifier-model"
            cache_dir = os.path.expanduser("~/.cache/zmb_classifier_model")
            model_path = snapshot_download(repo_id=hf_repo, cache_dir=cache_dir)

        print(f"[INFO] Carregando modelo de: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model.eval()

        self.label_map = {
            0: "Sem referência racial",
            1: "Com referência racial"
        }

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        return {
            "text": text,
            "predicted_class": predicted_class_id,
            "predicted_label": self.label_map.get(predicted_class_id, f"Classe desconhecida: {predicted_class_id}")
        }

    def predict_batch(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_class_ids = torch.argmax(logits, dim=-1).tolist()
        return [
            {
                "text": text,
                "predicted_class": cid,
                "predicted_label": self.label_map.get(cid, f"Classe desconhecida: {cid}")
            }
            for text, cid in zip(texts, predicted_class_ids)
        ]
