import yaml

def load_config(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
