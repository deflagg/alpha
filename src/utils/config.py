import yaml
import os
from pathlib import Path
from typing import Any, Dict

class Config:
    def __init__(self, config_dict: Dict[str, Any], config_path: str = None):
        self._config = config_dict
        self.config_path = config_path
        
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict, path)

    def __getattr__(self, name: str) -> Any:
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return Config(value, self.config_path)
            return value
        raise AttributeError(f"'Config' object has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def to_dict(self) -> Dict[str, Any]:
        return self._config

    def resolve_paths(self, base_dir: Path):
        """Recursively resolve all paths relative to base_dir if they are strings and contain 'dir' or 'path' in key."""
        def _resolve(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    _resolve(v)
                elif isinstance(v, str) and ('dir' in k or 'path' in k or 'file' in k):
                    # Only resolve if it's not an absolute path already
                    if not os.path.isabs(v):
                        d[k] = str((base_dir / v).resolve())
        _resolve(self._config)

def load_config(config_path: str) -> Config:
    config = Config.from_yaml(config_path)
    # Resolve paths relative to the project root (assumed to be parent of 'src')
    project_root = Path(__file__).parent.parent.parent
    config.resolve_paths(project_root)
    return config
