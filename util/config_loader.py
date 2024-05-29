import yaml


class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        try:
            with open(self.config_path, "r") as stream:
                return yaml.safe_load(stream)
        except FileNotFoundError:
            raise Exception(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise Exception(f"Error parsing YAML file: {e}")

    def get_config(self):
        return self.config
