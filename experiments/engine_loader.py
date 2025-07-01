"""
Configuration loader for YAML-based model configurations.
"""

import glob
import os
from typing import Any, Dict, List

import yaml
from rich import print as _print

from agent.src.typedefs import (
    AnthropicParams,
    EngineParams,
    EngineType,
    GeminiParams,
    HuggingFaceParams,
    OpenAIParams,
)


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Load a single YAML configuration file.

    Args:
        file_path (str): Path to the YAML file

    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def load_engine_params_from_yaml(file_path: str) -> EngineParams:
    """
    Load EngineParams from a YAML file, using provider-specific validation.

    Args:
        file_path (str): Path to the YAML file

    Returns:
        EngineParams: Validated engine parameters (provider-specific subclass)
    """
    config_dict = load_yaml_config(file_path)

    # Get engine type to determine which specific class to use
    engine_type_str = config_dict.get("engine_type")
    if not engine_type_str:
        raise ValueError(f"Missing 'engine_type' in config file: {file_path}")

    try:
        engine_type = EngineType(engine_type_str)
    except ValueError:
        raise ValueError(f"Invalid engine_type '{engine_type_str}' in {file_path}")

    # Dispatch to provider-specific class for validation
    if engine_type == EngineType.OPENAI:
        return OpenAIParams.from_dict(config_dict)
    elif engine_type == EngineType.ANTHROPIC:
        return AnthropicParams.from_dict(config_dict)
    elif engine_type == EngineType.GEMINI:
        return GeminiParams.from_dict(config_dict)
    elif engine_type == EngineType.HUGGINGFACE:
        return HuggingFaceParams.from_dict(config_dict)
    else:
        # Fallback to generic EngineParams
        _print(f"[yellow]Using generic EngineParams for {file_path}")
        return EngineParams.from_dict(config_dict)


def load_all_model_engine_params(
    config_dir: str = "config/models",
    include: List[str] = None,
    exclude: List[str] = None,
) -> List[EngineParams]:
    """
    Load all YAML configuration files from a directory, excluding files that start with '_' or '.'.

    Args:
        config_dir (str): Directory containing YAML config files
        include (List[str], optional): Only include config files with these names (without extension)
        exclude (List[str], optional): Exclude config files with these names (without extension)

    Returns:
        List[EngineParams]: List of engine parameter configurations
    """
    yaml_files = glob.glob(os.path.join(config_dir, "*.yaml")) + glob.glob(
        os.path.join(config_dir, "*.yml")
    )

    configs = []
    for yaml_file in sorted(yaml_files):
        filename = os.path.basename(yaml_file)
        if filename.startswith(("_", ".")):
            continue

        # Get filename without extension for filtering
        config_name = os.path.splitext(filename)[0]

        # Apply include filter
        if include is not None and config_name not in include:
            continue

        # Apply exclude filter
        if exclude is not None and config_name in exclude:
            continue

        try:
            config = load_engine_params_from_yaml(yaml_file)
            configs.append(config)
            print(
                f"Loaded config from {yaml_file}: {config.engine_type}/{config.model}"
            )
        except Exception as e:
            print(f"Error loading config from {yaml_file}: {e}")
            continue

    if not configs:
        raise ValueError(f"No valid YAML configuration files found in {config_dir}")

    return configs
