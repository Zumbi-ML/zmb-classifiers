import yaml
from pathlib import Path

def load_config(path="config.yaml"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f)

CONFIG = load_config()