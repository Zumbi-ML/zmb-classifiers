import yaml
from importlib import resources

def load_config():
    with resources.open_text("zmb_classifiers", "config.yaml") as f:
        return yaml.safe_load(f)

CONFIG = load_config()
