import csv
import time
from _warnings import warn
from pathlib import Path
from typing import Dict

import pandas as pd
from rich import print as _print

from agent.src.config import get_config


def _collect_model_experiment_data(model_dir: Path) -> pd.DataFrame:
    """
    Collect experiment_data.csv files from a single model directory.
    
    Expected directory structure:
        model_dir/  (e.g., Provider_ModelName/)
            product_name/
                journey_id/
                    experiment_data.csv
                    engine_params.txt
                    ...
    
    Args:
        model_dir (Path): The model directory containing product subdirectories
        
    Returns:
        pd.DataFrame: Combined DataFrame from all journeys in this model
    """
    journey_data = []
    model_name = model_dir.name
    
    # Iterate through each product directory
    for product_dir in model_dir.iterdir():
        if product_dir.is_dir():
            product_name = product_dir.name
            
            # Iterate through each journey directory under the product
            for journey_dir in product_dir.iterdir():
                if journey_dir.is_dir():
                    journey_id = journey_dir.name
                    experiment_csv_path = journey_dir / "experiment_data.csv"
                    
                    # Process experiment_data.csv if it exists
                    if experiment_csv_path.is_file():
                        try:
                            df = pd.read_csv(experiment_csv_path)
                            # Add metadata columns
                            df["model_name_dir"] = model_name  # From directory structure
                            df["product_name"] = product_name
                            df["journey_id"] = journey_id
                            
                            journey_data.append(df)
                        except Exception as e:
                            warn(f"Error reading {experiment_csv_path}: {e}")
    
    # Combine all journeys for this model
    if journey_data:
        return pd.concat(journey_data, ignore_index=True)
    else:
        return pd.DataFrame()


def _collect_run_experiment_data(run_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Collect experiment data from all models in a run directory.
    
    Expected directory structure:
        run_dir/  (e.g., run_20241210_143022/)
            Provider_ModelName_1/
                product_name/
                    journey_id/
                        experiment_data.csv
            Provider_ModelName_2/
                product_name/
                    journey_id/
                        experiment_data.csv
    
    Args:
        run_dir (Path): The run directory containing model subdirectories
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with keys as model names and values as combined DataFrames
    """
    model_data = {}
    all_journey_data = []
    
    # Iterate through each model directory
    for model_dir in run_dir.iterdir():
        if model_dir.is_dir() and "_" in model_dir.name:  # Model directories have format Provider_ModelName
            model_name = model_dir.name
            
            # Collect data for this model
            model_df = _collect_model_experiment_data(model_dir)
            
            if not model_df.empty:
                model_data[model_name] = model_df
                all_journey_data.append(model_df)
    
    # Add combined data across all models
    if all_journey_data:
        model_data["_ALL_MODELS"] = pd.concat(all_journey_data, ignore_index=True)
    
    return model_data


def aggregate_model_data(model_path: Path):
    """
    Aggregate experiment data for a single model.
    
    This is called from runner.py with the model-specific directory.
    Creates: {model_dir}/aggregated_experiment_data.csv
    
    Args:
        model_path (Path): The model directory (e.g., run_dir/Provider_ModelName/)
    """
    if not model_path.exists():
        warn(f"Model directory does not exist: {model_path}")
        return
    
    # Collect experiment data for this model
    model_df = _collect_model_experiment_data(model_path)
    
    if model_df.empty:
        warn(f"No experiment data found in {model_path}")
        return
    
    # Save aggregated data for this model
    try:
        output_path = model_path / "aggregated_experiment_data.csv"
        model_df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)  # type: ignore[arg-type]
        _print(f"[green]Model aggregated data saved to[/] [blue underline]{output_path}")
    except Exception as e:
        warn(f"Error saving model aggregated data: {e}")


def aggregate_run_data(run_dir: str):
    """
    Aggregate experiment data across all models in a run directory.
    
    Creates:
    - Per-model aggregated files: {run_dir}/{model_name}/aggregated_experiment_data.csv
    - Run-level aggregated file: {run_dir}/run_aggregated_experiment_data.csv
    
    Args:
        run_dir (str): The run directory containing model subdirectories
    """
    run_path = Path(run_dir)
    
    if not run_path.exists():
        warn(f"Run directory does not exist: {run_dir}")
        return
    
    # Collect data from all models in this run
    model_data = _collect_run_experiment_data(run_path)
    
    if not model_data:
        warn(f"No experiment data found in {run_dir}")
        return
    
    # Save per-model aggregated data (if not already done)
    for model_name, df in model_data.items():
        if model_name == "_ALL_MODELS":
            continue  # Skip the combined data for now
            
        try:
            model_dir = run_path / model_name
            if model_dir.exists():
                output_path = model_dir / "aggregated_experiment_data.csv"
                # Only create if it doesn't exist (runner.py should have already created it)
                if not output_path.exists():
                    df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)  # type: ignore[arg-type]
                    _print(f"[green]Model aggregated data saved to[/] [blue underline]{output_path}")
        except Exception as e:
            warn(f"Error saving model aggregated data for {model_name}: {e}")
    
    # Save run-level aggregated data
    if "_ALL_MODELS" in model_data:
        try:
            run_output_path = run_path / "run_aggregated_experiment_data.csv"
            model_data["_ALL_MODELS"].to_csv(run_output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)  # type: ignore[arg-type]
            _print(f"[green]Run aggregated data saved to[/] [blue underline]{run_output_path}")
        except Exception as e:
            warn(f"Error saving run aggregated data: {e}")


def continue_experiment_or_exit(delay: int = None):
    """Handle keyboard interrupt with delay before continuing."""
    if delay is None:
        delay = get_config().continue_or_exit_wait_delay
    try:
        _print("[bold red]Keyboard interrupt detected. "
               f"Use Ctrl+C in {delay} seconds to cancel subsequent experiments...")
        for countdown in range(delay, 0, -1):
            _print(f"[bold red]Restarting in {countdown} seconds...")
            time.sleep(1)
        _print("Moving on to the next experiment...")
    except KeyboardInterrupt:
        _print("[bold red]Experiment cancelled.")
        exit(1)
