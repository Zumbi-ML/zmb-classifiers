import json
import csv
import os
import argparse
from glob import glob

def map_label(value):
    """Converte o campo interest para 1 ou 0"""
    if value.strip().lower() == "sim":
        return "1"
    else:
        return "0"

def process_jsons(input_dir, output_file, text_field="text", label_field="interest"):
    rows = []

    for filepath in glob(os.path.join(input_dir, "*.json")):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for item in data:
                    text = item.get(text_field, "").replace("\n", " ").replace('"', "'").strip()
                    raw_label = item.get(label_field, "").strip()
                    label = map_label(raw_label)
                    rows.append([text, label])
            except json.JSONDecodeError:
                print(f"❌ Erro ao ler JSON: {filepath}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print(f"✅ Dataset salvo em: {output_file} - Total de exemplos: {len(rows)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera dataset CSV a partir de vários JSONs")
    parser.add_argument("--input_dir", type=str, required=True, help="Diretório de entrada com JSONs")
    parser.add_argument("--output_file", type=str, required=True, help="Arquivo CSV de saída")
    parser.add_argument("--text_field", type=str, default="text", help="Campo usado como texto (default: 'text')")
    parser.add_argument("--label_field", type=str, default="interest", help="Campo usado como label (default: 'interest')")

    args = parser.parse_args()

    process_jsons(args.input_dir, args.output_file, args.text_field, args.label_field)