import yaml
from pathlib import Path
from typing import Optional


def load_config(config_path: Optional[str] = None) -> dict:
    """Load YAML config.
    """
    if config_path:
        p = Path(config_path)
    else:
        # repo_root/ config/config.yaml  (utils/ -> repo_root)
        p = Path(__file__).resolve().parents[1] / "config" / "config.yaml"

    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)