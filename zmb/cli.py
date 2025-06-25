import argparse
import yaml

def main():
    parser = argparse.ArgumentParser(description="ZMB Classifiers - Treinamento, Avaliação, Dataset e Inferência")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    train_parser = subparsers.add_parser("train", help="Executa o pipeline completo de treinamento")
    train_parser.add_argument("--config", type=str, default="config.yaml", help="Arquivo de configuração YAML")

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Avalia o modelo salvo com base no dataset de teste")
    eval_parser.add_argument("--config", type=str, default="config.yaml", help="Arquivo de configuração YAML")

    # make-dataset
    make_ds_parser = subparsers.add_parser("make-dataset", help="Gera CSV de treinamento a partir de JSONs")
    make_ds_parser.add_argument("--config", type=str, default="config.yaml", help="Arquivo de configuração YAML")

    # predict
    predict_parser = subparsers.add_parser("predict", help="Faz a inferência de um texto usando o modelo final salvo")
    predict_parser.add_argument("--text", type=str, required=True, help="Texto da matéria jornalística para classificar")

    args = parser.parse_args()

    if args.command == "train":
        from zmb.pipeline import run_pipeline
        run_pipeline(args.config)

    elif args.command == "evaluate":
        from zmb.evaluate import evaluate_model
        evaluate_model(args.config)

    elif args.command == "make-dataset":
        from zmb.make_ds import process_jsons
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        input_dir = config["paths"]["raw_json_dir"]
        output_file = config["paths"]["dataset_csv"]
        process_jsons(input_dir, output_file)

    elif args.command == "predict":
        import json
        from zmb.inference import ZMBClassifier
        clf = ZMBClassifier(model_path="./output")
        result = clf.predict(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2))