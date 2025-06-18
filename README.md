# 📰 ZMB Classifiers

Classificador de matérias jornalísticas com pipeline de treinamento modular e configurável.

---

## 📦 Instalação

Clone o repositório e instale com:

```bash
pip install .
```

---

## ⚙️ Uso via Terminal

Execute o pipeline com o arquivo de configuração YAML:

```bash
zmb-train --config config.yaml
```

---

## 🧠 Uso via Python

```python
from zmb.pipeline import run_pipeline
run_pipeline("config.yaml")
```

---

## 🗂️ Estrutura do Projeto

```
zmb-classifiers/
├── zmb/                      # Pacote com os módulos principais
│   ├── __init__.py
│   ├── preprocess.py         # Pré-processamento dos dados
│   ├── train.py              # Treinamento do modelo
│   ├── evaluate.py           # Avaliação do modelo
│   ├── pipeline.py           # Orquestrador geral
├── config.yaml               # Arquivo de configuração
├── cli.py                    # Interface de linha de comando
├── setup.py                  # Script de instalação
├── data/                     # Dados de entrada (não versionados)
```

---

## 🔧 Exemplo de `config.yaml`

```yaml
data:
  input: "./data/03-ready_4_training/classifier-dataset.csv"
  target: "label"

model:
  type: "logistic_regression"
  save_path: "./models/zmb_model.pkl"

training:
  test_size: 0.2
  random_state: 42
```

---

## 📋 Requisitos

- Python 3.7+
- scikit-learn
- pandas
- PyYAML
- joblib

Instalados automaticamente com `pip install .`

---

## 👨‍💻 Autor

Jefferson O. Silva — [silvajo@pucsp.br]