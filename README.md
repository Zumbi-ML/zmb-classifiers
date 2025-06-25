# 📰 ZMB Classifiers

[![Hugging Face](https://img.shields.io/badge/model-hub-%23FF6A6A?logo=huggingface)](https://huggingface.co/j3ffsilva/zmb-classifier-model)

**Pipeline modular para classificação binária de matérias jornalísticas com referências raciais ou culturais (branca / negra / cultura negra).**

---

## 📚 Uso como biblioteca Python

Se você deseja usar o classificador diretamente no seu código Python sem executar o pipeline completo:

```python
from zmb_classifiers.inference import ZmbClassifier

clf = ZmbClassifier()
resultado = clf.predict("A capoeira é um símbolo da resistência negra no Brasil.")

print(resultado)
# Saída esperada:
# {
#   'text': 'A capoeira é um símbolo da resistência negra no Brasil.',
#   'predicted_class': 1,
#   'predicted_label': 'Com referência racial'
# }
````

> ⚠️ Na primeira execução, o modelo será baixado automaticamente do Hugging Face e armazenado em cache local.

---

## 🎯 Objetivo

Classificar textos jornalísticos como:

* **`0`** – Sem referências a raça/cultura branca, raça/cultura negra ou cultura negra.
* **`1`** – Com **pelo menos uma** referência a:

  * Raça branca (ex.: *"um homem branco"*, *"supremacia branca"*).
  * Raça negra (ex.: *"uma mulher negra"*, *"uma pessoa preta"*).
  * Cultura negra (ex.: *"capoeira"*, *"Lei Áurea"*).

---

## 📦 Instalação

```bash
pip install zmb-classifiers
```

> 🔄 Alternativamente, para uso em desenvolvimento:

```bash
git clone https://github.com/Zumbi-ML/zmb-classifiers.git
cd zmb-classifiers
pip install -e .
```

### 📌 Dependência obrigatória

Instale também o [zmb-newslink-extractor](https://github.com/Zumbi-ML/zmb-newslink-extractor), responsável pela extração dos textos jornalísticos:

```bash
git clone https://github.com/Zumbi-ML/zmb-newslink-extractor.git
cd zmb-newslink-extractor
pip install -e .
```

---

## 📥 Download automático do modelo

Este projeto utiliza um modelo treinado salvo em:
➡️ [https://huggingface.co/j3ffsilva/zmb-classifier-model](https://huggingface.co/j3ffsilva/zmb-classifier-model)

O modelo é baixado automaticamente na primeira execução do `ZmbClassifier`.
Ele é armazenado em cache local em:

```bash
~/.cache/zmb_classifier_model/
```

---

## 🚩 Pipeline de Uso

### 1. **Preparação das URLs e Labels**

Crie um arquivo Excel em `data/01-newslinks-labelled/` com o seguinte formato:

| URL                          | label |
| ---------------------------- | ----- |
| `https://.../materia1.shtml` | 1     |
| `https://.../materia2.shtml` | 0     |

> ✅ Colunas obrigatórias: `URL` e `label`

---

### 2. **Extração dos Textos** (usando `zmb-newslink-extractor`)

```bash
zmb-extract \
  --file data/01-newslinks-labelled/seus_dados.xlsx \
  --sheet "Sheetname" \
  --label-column "label" \
  --export-path data/03-jsonified/ \
  --output materias_extraidas
```

**Saída:** Arquivos JSON em `data/03-jsonified/`, um por matéria.

---

### 3. **Geração do Dataset de Treinamento (CSV)**

```bash
zmb-clf make-dataset --config config.yaml
```

**Saída:** Um CSV consolidado em `data/04-ready_4_training/classifier_dataset.csv`.

---

### 4. **Treinamento do Modelo**

```bash
zmb-clf train --config config.yaml
```

**Saída:** Modelo salvo em `models/zmb_model/`.

---

### 5. **(Opcional) Avaliação da Performance**

```bash
zmb-clf evaluate --config config.yaml
```

**Saída:** Métricas como precisão, recall e F1-score em `evaluation/resultados_avaliacao.csv`.

---

## 🗂️ Estrutura do Projeto

```
zmb-classifiers/
├── data/
│   ├── 01-newslinks-labelled/   # URLs + labels
│   ├── 03-jsonified/            # Matérias extraídas (JSON)
│   └── 04-ready_4_training/     # Dataset final (CSV)
├── models/                      # Modelos treinados
├── evaluation/                  # Avaliações
├── zmb_classifiers/             # Código-fonte do classificador
├── config.yaml                  # Configuração
├── requirements.txt             # Dependências
└── README.md                    # Este arquivo
```

---

## 🔧 Exemplo de Configuração (`config.yaml`)

```yaml
paths:
  raw_json_dir: "./data/03-jsonified"
  dataset_csv: "./data/04-ready_4_training/classifier_dataset.csv"
  model_output_dir: "./models/zmb_model"
  evaluation_results: "./evaluation/resultados_avaliacao.csv"

training:
  test_size: 0.2
  random_state: 42
```

---

## 📋 Requisitos

* Python 3.8+
* transformers >= 4.46.0
* datasets >= 3.1.0
* torch >= 2.4.0
* pandas >= 2.0.0
* scikit-learn >= 1.3.0
* PyYAML >= 6.0
* tqdm
* psutil
* huggingface\_hub
* safetensors

> Instale as dependências com:

```bash
pip install -r requirements.txt
```

---

## 📖 Como Citar o Projeto>

Se você utilizar o classificador Zumbi em sua pesquisa, por favor cite-o da seguinte forma:

> SILVA, JEFFERSON O. **Negro ou Branco? Um modelo para detectar referências raciais em matérias jornalísticas**. 2025. Disponível em: [https://github.com/Zumbi-ML/zmb-classifiers](https://github.com/Zumbi-ML/zmb-classifiers). Acesso em: \[data de acesso].

Citação em BibTeX:

```bibtex
@misc{silva2025zmbclassifiers,
  author       = {Jefferson Oliveira Silva},
  title        = {Negro ou Branco? Um modelo para detectar referências raciais em matérias jornalísticas},
  year         = {2025},
  howpublished = {\url{https://github.com/Zumbi-ML/zmb-classifiers}},
  note         = {Acesso em: \today}
}
```

---

## 👨‍💻 Autor

Jefferson O. Silva – [silvajo@pucsp.br](mailto:silvajo@pucsp.br)





