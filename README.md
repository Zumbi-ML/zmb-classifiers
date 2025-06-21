# 📰 ZMB Classifiers

**Pipeline modular para classificação binária de matérias jornalísticas com referências raciais ou culturais (branca / negra / cultura negra).**

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
git clone git@github.com:Zumbi-ML/zmb-classifiers.git
cd zmb-classifiers
pip install .
```

### 📌 Dependência obrigatória

Instale também o [zmb-newslink-extractor](https://github.com/seu_usuario/zmb-newslink-extractor), responsável pela extração dos textos jornalísticos:

```bash
git clone git@github.com:Zumbi-ML/zmb-newslink-extractor.git
cd zmb-newslink-extractor
pip install .
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

**Exemplo de JSON:**

```json
[
  {
    "title": "Título da matéria",
    "source": "URL da matéria",
    "text": "Texto completo da matéria...",
    "interest": 1
  }
]
```

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
├── zmb/                         # Código-fonte
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

* Python 3.7+
* transformers >= 4.46.0
* datasets >= 3.1.0
* torch >= 2.4.0
* pandas >= 2.0.0
* scikit-learn >= 1.3.0
* PyYAML >= 6.0
* tqdm
* psutil
* zmb-newslink-extractor (projeto separado)

> Instale as dependências com:

```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Autor

Jefferson O. Silva – [silvajo@pucsp.br](mailto:silvajo@pucsp.br)