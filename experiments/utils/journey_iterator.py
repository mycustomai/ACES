import json
import warnings
from pathlib import Path
from typing import Iterator, Optional, Any, Sequence

import pandas as pd


def iterate_journey_directories(results_path: Path, query: Optional[str] = None, model_name: Optional[str] = None) -> Iterator[Path]:
    """
    Iterates through all journey directories for a specified dataset results path.
    
    Yields Path objects for directories matching the pattern:
    experiment_logs/{dataset_name}/{model_name}/{query}/{experiment_id}/
    
    Args:
        results_path: Path to the results directory (e.g., "experiment_logs/rating_sanity_check_dataset")
        query: Optional query name (e.g., "mousepad")
        model_name: Optional case-insensitive model name. Can be a snippet e.g., "2.5-flash"
        
    Yields:
        Path: Directory paths for individual experiment journeys
    """
    if not results_path.exists() or not results_path.is_dir():
        raise ValueError(f"Results path does not exist or is not a directory: {results_path}")
            
    # Iterate through model directories
    for model_dir in results_path.iterdir():
        if not model_dir.is_dir():
            continue
        elif model_name and model_name.lower() not in model_dir.name.lower():
            continue

        # Iterate through query directories
        for query_dir in model_dir.iterdir():
            if not query_dir.is_dir():
                continue
            elif query and query.lower() not in query_dir.name.lower():
                continue
                
            # Iterate through experiment_id directories
            for experiment_dir in query_dir.iterdir():
                if not experiment_dir.is_dir():
                    continue
                    
                yield experiment_dir


def extract_reasoning_for_journey(journey_dir: Path) -> str:
    """Extract reasoning from the agent_response.json file in a specific journey directory."""
    if not journey_dir.is_dir():
        raise ValueError(f"Journey directory does not exist or is not a directory: {journey_dir}")

    agent_response_file = journey_dir / "agent_response.json"
    if agent_response_file.exists():
        with open(agent_response_file, "r") as f:
            data = json.load(f)
            return data["text"]
    else:
        raise ValueError(f"Missing agent_response.json file in {journey_dir}")


def _evenly_sample_list[T](target_list: Sequence[T], sample_frequency: float) -> Sequence[T]:
    if sample_frequency <= 0:
        raise ValueError("Sample frequency must be greater than 0.")
    elif sample_frequency >= 1:
        raise ValueError("Sample frequency must be less than 1.")
    elif sample_frequency == 1:
        return target_list

    num_items = len(target_list)
    num_samples = int(num_items * sample_frequency)
    if num_items < 25:
        warnings.warn(f"Target list only contains {num_items} items. Consider increasing sample frequency.", stacklevel=0)
    elif num_samples == 0:
        raise ValueError("No samples returned. Increase sample frequency.")
    if num_samples > 0 and num_items > 0:
        step = num_items / num_samples
        indices = [int(i * step) for i in range(num_samples)]
        return [target_list[i] for i in indices]
    else:
        raise ValueError("Invalid result returned from sampling. Unknown error.")


def sample_journey_reasoning(results_path: Path, *, query: Optional[str], model_name: Optional[str] = None, sample_frequency: Optional[float]) -> list[str]:
    """Iterate through journey directories and extract reasoning for each, without regard to the selected product.

    Args:
        results_path: Path to the results directory (e.g., "experiment_logs/rating_sanity_check_dataset")
        query: Optional query name (e.g., "mousepad")
        model_name: Optional case-insensitive model name. Can be a snippet e.g., "2.5-flash"
        sample_frequency: Fraction of journeys to sample (e.g., 0.1 for 10% of journeys). When provided, samples are evenly distributed across all journeys.
    """
    target_journey_dirs = [d for d in iterate_journey_directories(results_path, query, model_name)]

    if sample_frequency:
        target_journey_dirs = _evenly_sample_list(target_journey_dirs, sample_frequency)

    return [extract_reasoning_for_journey(d) for d in target_journey_dirs]


def _is_incorrect_rationality_check(journey_dir: Path) -> bool:
    """Iterate through journey directories for rationality checks and extract reasoning for each, only considering the selected product.

    Args:
        journey_dir: Path to the journey directory (e.g., "experiment_logs/rating_sanity_check_dataset/2.5-flash/mousepad/1/journey_1")

    Returns:
        True if the journey is an incorrect rationality check, False otherwise.
    """
    if not journey_dir.is_dir():
        raise ValueError(f"Journey directory does not exist or is not a directory: {journey_dir}")

    experiment_data_file = journey_dir / "experiment_data.csv"
    if not experiment_data_file.exists():
        warnings.warn(f"Missing experiment_data.csv file in {journey_dir}", stacklevel=0)
    experiment_df = pd.read_csv(experiment_data_file)

    if "desired_choice" not in experiment_df.columns:
        raise ValueError(f"Missing desired_choice column in '{experiment_data_file}'. Experiment is not a rationality check.")
    elif experiment_df['selected'].sum() < 0:  # do not hardcode an expected sum for flexibility. see logger for more information.
        return True
    else:
        return False


def extract_incorrect_rationality_check_reasoning(results_path: Path, *, query: Optional[str], model_name: Optional[str] = None, sample_frequency: Optional[float]) -> list[str]:
    """Iterate through journey directories for rationality checks and extract reasoning for each, only considering the selected product.

    Args:
        results_path: Path to the results directory (e.g., "experiment_logs/rating_sanity_check_dataset")
        query: Optional query name (e.g., "mousepad")
        model_name: Optional case-insensitive model name. Can be a snippet e.g., "2.5-flash"
        sample_frequency: Fraction of journeys to sample (e.g., 0.1 for 10% of journeys). When provided, samples are evenly distributed across all journeys which contain the selected product.

    Note:
        Since the incorrect responses for rationality checks are typically small, a warning is logged if the total number of valid journeys is less than 50 if sample_frequency is provided.
    """
    valid_journey_dirs = [
        d for d in iterate_journey_directories(results_path, query, model_name)
        if _is_incorrect_rationality_check(d)
    ]

    if sample_frequency:
        valid_journey_dirs = _evenly_sample_list(valid_journey_dirs, sample_frequency)

    return [extract_reasoning_for_journey(d) for d in valid_journey_dirs]


def _is_specific_product_selection(journey_dir: Path, product_substr: str) -> bool:
    """Check if a journey selected a product containing the given substring.
    
    Args:
        journey_dir: Path to the journey directory
        product_substr: Substring to search for in the selected product
        
    Returns:
        True if the selected product contains the substring, False otherwise
    """
    if not journey_dir.is_dir():
        raise ValueError(f"Journey directory does not exist or is not a directory: {journey_dir}")
    
    selected_product_file = journey_dir / "selected_product.txt"
    if not selected_product_file.exists():
        warnings.warn(f"Missing selected_product.txt file in {journey_dir}", stacklevel=0)
        return False
    
    with open(selected_product_file, "r") as f:
        selected_product = f.read().strip()
    
    return product_substr.lower() in selected_product.lower()


def extract_specific_product_journey_reasoning(results_path: Path, *, query: Optional[str], model_name: Optional[str] = None, product_substr: Optional[str] = None, inverse: bool = False, sample_frequency: Optional[float] = None) -> list[str]:
    """Iterate through journey directories for specific products and extract reasoning for each.

    Args:
        results_path: Path to the results directory (e.g., "experiment_logs/rating_sanity_check_dataset")
        query: Optional query name (e.g., "mousepad")
        model_name: Optional case-insensitive model name. Can be a snippet e.g., "2.5-flash"
        product_substr: Substring to search for in selected products
        inverse: If True, select journeys that DO NOT contain the product substring
        sample_frequency: Fraction of journeys to sample (e.g., 0.1 for 10% of journeys)
        
    Returns:
        List of reasoning strings from matching journeys
    """
    if product_substr is None:
        raise ValueError("product_substr must be provided")
    
    valid_journey_dirs = []
    for d in iterate_journey_directories(results_path, query, model_name):
        is_match = _is_specific_product_selection(d, product_substr)
        if (is_match and not inverse) or (not is_match and inverse):
            valid_journey_dirs.append(d)
    
    if sample_frequency:
        valid_journey_dirs = _evenly_sample_list(valid_journey_dirs, sample_frequency)
    
    return [extract_reasoning_for_journey(d) for d in valid_journey_dirs]
