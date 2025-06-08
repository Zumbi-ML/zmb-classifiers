# ZMB Classifiers

Classificador para detecção de referências à população negra ou branca em textos jornalísticos.

## Visão Geral

Este projeto tem como objetivo treinar um modelo de classificação que identifica se um texto jornalístico contém referências à população negra ou branca. O pipeline completo inclui:

1. Extração de conteúdo de URLs de matérias jornalísticas
2. Pré-processamento dos textos
3. Treinamento do modelo classificador
4. Avaliação e predição

## Estrutura do Projeto

```
.
├── data
│   ├── 01-raw                      # Dados brutos (planilha com URLs)
│   ├── 02-jsonified                # Matérias extraídas em formato JSON
│   └── 03-ready_4_training         # Dataset pré-processado para treinamento
├── metrics                         # Métricas de avaliação dos modelos
├── models                          # Versões salvas dos modelos de classificação
├── notebooks                       # Jupyter notebooks para análise
├── README.md                       # Este arquivo
├── requirements.txt                # Dependências do projeto
└── src
    ├── evaluate.py                 # Script para avaliação do modelo
    ├── predict.py                  # Script para fazer predições
    ├── preprocess.py               # Script de pré-processamento
    ├── train.py                    # Script de treinamento
    └── utils                       # Utilitários auxiliares
```

## Pré-requisitos

- Python 3.6+
- pip
- Git

## Instalação

1. Clone o repositório:
```bash
git clone github
cd zmb-classifiers
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Instale o extrator de notícias (necessário para a etapa de extração):
```bash
git clone [URL_DO_REPOSITORIO_zmb-newslink-extractor]
cd zmb-newslink-extractor
pip install -e .
cd ..
```

## Fluxo de Trabalho

### 1. Extração de Matérias Jornalísticas

Extraia o conteúdo das URLs listadas na planilha:

```bash
zmb-extract --file data/01-raw/zmb-URLs_4_classifier_training.xlsx \
            --sheet "FSP-Escravo" \
            --label-column "Ref Negros ,Brancos Como Raça ou a Elementos da Cultura?" \
            --export-path data/02-jsonified \
            --output FSP-escravo
```

Repita o processo para todas as abas/planilhas necessárias.

### 2. Pré-processamento dos Dados

Combine todos os JSONs extraídos em um único dataset para treinamento:

```bash
python src/preprocess.py
```

### 3. Treinamento do Modelo

Execute o treinamento do classificador:

```bash
python src/train.py \
    --dataset-path data/03-ready_4_training/classifier-dataset.csv \
    --output-dir models/classifier_v1 \
    --num-epochs 5 \
    --learning-rate 2e-5 \
    --batch-size 8
```

Parâmetros:
- `--dataset-path`: Caminho para o arquivo CSV com os dados de treinamento
- `--output-dir`: Diretório para salvar o modelo treinado
- `--num-epochs`: Número de épocas de treinamento
- `--learning-rate`: Taxa de aprendizado
- `--batch-size`: Tamanho do batch

### 4. Avaliação do Modelo

Para avaliar o modelo treinado:

```bash
python src/evaluate.py --model-path models/classifier_v1 --test-data data/03-ready_4_training/classifier-dataset.csv
```

### 5. Predição

Para classificar novos textos:

```bash
python src/predict.py --model-path models/classifier_v1 --text "Texto a ser classificado"
```

## Contribuição

Contribuições são bem-vindas! Siga os passos:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Licença

MIT

## Contato

silvajo@pucsp.br