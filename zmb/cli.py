import argparse
from zmb.pipeline import run_pipeline
from zmb.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="ZMB Classifiers - Treinamento e Avaliação")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcomando: train
    train_parser = subparsers.add_parser("train", help="Executa o pipeline completo de treinamento")
    train_parser.add_argument("--config", type=str, default="config.yaml", help="Arquivo de configuração YAML")

    # Subcomando: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Avalia o modelo salvo com base no dataset de teste")
    eval_parser.add_argument("--config", type=str, default="config.yaml", help="Arquivo de configuração YAML")

    args = parser.parse_args()

    if args.command == "train":
        run_pipeline(args.config)
    elif args.command == "evaluate":
        evaluate_model(args.config)

if __name__ == "__main__":
    main()
