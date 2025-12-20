import os
import yaml

# (поднимаемся на 2 уровня вверх от текущего файла: src/utils/ -> src/ -> root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
REQ_PATH = os.path.join(BASE_DIR, "requirements.txt")


def load_requirements(path: str = REQ_PATH) -> list:
    """Читает requirements.txt и возвращает список пакетов."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Requirements file not found at: {path}")
    
    with open(path, "r") as f:
        return list(filter(None, f.read().splitlines()))


def load_yaml_config(path: str = CONFIG_PATH) -> dict:
    """Читает YAML конфиг и возвращает словарь."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at: {path}")
        
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {path}")
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {path}") from e


PIPELINE_PACKAGES = load_requirements()
LOCAL_CONFIG = load_yaml_config()