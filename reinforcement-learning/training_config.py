from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "training_config.yaml"


def load_training_config(path, section, include_common=True):
    """Load and merge the common and entry-point-specific YAML settings."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    if not isinstance(config, dict):
        raise ValueError("Training config must contain a YAML mapping")

    common = config.get("common", {}) if include_common else {}
    specific = config.get(section, {})
    if not isinstance(common, dict) or not isinstance(specific, dict):
        raise ValueError("'common' and entry-point sections must be YAML mappings")

    structured = {}
    for name in ("model", "optimizer", "replay", "checkpoint", "runtime"):
        values = config.get(name, {})
        if not isinstance(values, dict):
            raise ValueError(f"'{name}' must be a YAML mapping")
        structured.update(values)

    # Null values mean "use the script's automatic/default value".
    merged = {
        key: value
        for key, value in {**structured, **common, **specific}.items()
        if value is not None
    }
    for key in (
        "checkpoint_dir",
        "resume",
        "model_out",
        "replay_path",
        "data_dir",
        "plot_path",
        "init_model",
        "sl_data_dir",
        "midgame_fens",
    ):
        if key in merged:
            value = Path(merged[key])
            if not value.is_absolute():
                value = config_path.resolve().parent / value
            merged[key] = str(value)
    return merged
