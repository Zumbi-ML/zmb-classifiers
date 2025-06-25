from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

class ZMBClassifier:
    def __init__(self, model_path="./output"):
        if not os.path.exists(model_path):
            raise ValueError(f"Modelo não encontrado em: {model_path}")

        print(f"[INFO] Carregando modelo de: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

        # Mapeamento de classes para labels descritivos
        self.label_map = {
            0: "Sem referência racial",
            1: "Com referência racial"
        }

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        predicted_label = self.label_map.get(predicted_class_id, f"Classe desconhecida: {predicted_class_id}")

        return {
            "text": text,
            "predicted_class": predicted_class_id,
            "predicted_label": predicted_label
        }

    def predict_batch(self, texts):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_class_ids = torch.argmax(logits, dim=-1).tolist()
        results = []
        for text, class_id in zip(texts, predicted_class_ids):
            results.append({
                "text": text,
                "predicted_class": class_id,
                "predicted_label": self.label_map.get(class_id, f"Classe desconhecida: {class_id}")
            })
        return results