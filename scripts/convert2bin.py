# convert_to_bin.py

from transformers import AutoModelForSequenceClassification
import torch

# Caminho onde o modelo safetensors já está OK
MODEL_ID = "output/"

# Baixa o modelo com suporte a safetensors
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, trust_remote_code=True)

# Salva como .bin real
torch.save(model.state_dict(), "pytorch_model.bin")

print("Modelo convertido com sucesso para pytorch_model.bin")
