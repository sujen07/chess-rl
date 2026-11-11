"""Load a flat YAML run config (configs/*.yaml) as argparse defaults."""

from pathlib import Path

import yaml


RL_DIR = Path(__file__).resolve().parent
CONFIG_DIR = RL_DIR / "configs"

# Relative values for these keys are resolved against reinforcement-learning/.
PATH_KEYS = (
    "checkpoint_dir", "resume", "model_out", "replay_path", "data_dir",
    "plot_path", "init_model", "sl_data_dir", "midgame_fens",
)


def load_config(path):
    """Return the config's settings; null values mean "use the script default"."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a flat YAML mapping: {config_path}")

    settings = {key: value for key, value in config.items() if value is not None}
    for key in PATH_KEYS:
        if key in settings and not Path(settings[key]).is_absolute():
            settings[key] = str(RL_DIR / settings[key])
    return settings


def apply_config(parser, config_path):
    """Use the config's settings as defaults for `parser`; CLI flags still win."""
    try:
        settings = load_config(config_path)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    valid = {action.dest for action in parser._actions}
    unknown = sorted(set(settings) - valid)
    if unknown:
        parser.error(f"unknown keys in {config_path}: {', '.join(unknown)}")
    parser.set_defaults(**settings)
