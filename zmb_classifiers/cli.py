import argparse
import yaml

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="ZMB Classifiers - Treinamento, Avaliação, Dataset e Inferência")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ["train", "evaluate", "make-dataset"]:
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", type=str, default="config.yaml")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--text", type=str, required=True)

    args = parser.parse_args()

    if args.command in ["train", "evaluate", "make-dataset"]:
        config = load_config(args.config)

    if args.command == "train":
        from zmb_classifiers.pipeline import run_pipeline
        run_pipeline(config)

    elif args.command == "evaluate":
        from zmb_classifiers.evaluate import evaluate_model
        evaluate_model()

    elif args.command == "make-dataset":
        from zmb_classifiers.make_ds import process_jsons
        input_dir = config["paths"]["raw_json_dir"]
        output_file = config["paths"]["dataset_csv"]
        process_jsons(input_dir, output_file)

    elif args.command == "predict":
        import json
        from zmb_classifiers.inference import ZmbClassifier
        from zmb_classifiers.config import CONFIG
        clf = ZmbClassifier(model_path=CONFIG["paths"]["best_model_dir"])
        result = clf.predict(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2))