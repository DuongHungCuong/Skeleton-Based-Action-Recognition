import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        configs = yaml.full_load(f)

    return configs