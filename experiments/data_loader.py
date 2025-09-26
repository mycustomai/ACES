from typing import Iterable, Optional
import pandas as pd
from datasets import load_dataset, Image

from experiments.config import ExperimentData, InstructionConfig


def experiments_iter(combined_df: pd.DataFrame, dataset_name: str) -> Iterable[ExperimentData]:
    """
    This function yields grouped dataframes for unique combinations of query, experiment label,
    and experiment number from the provided dataframe.

    Parameters:
    combined_df (pd.DataFrame): A pandas DataFrame containing the columns 'query',
        'experiment label', and 'experiment number'.

    Yields:
    Tuple[Tuple[str, str, int], pd.DataFrame]: A tuple where the first element is another tuple containing
        ``(query: str, experiment label: str, experiment number: int)``.
        The second element is a pandas DataFrame, representing the subset of rows grouped by the keys
        'query', 'experiment label', and 'experiment number'.

    Raises:
    AssertionError: If the provided DataFrame does not contain the required columns
        'query', 'experiment label', and 'experiment number'.
    """
    assert "query" in combined_df.columns, "Missing 'query' column in combined_df"
    assert "experiment_label" in combined_df.columns, "Missing 'experiment label' column in combined_df"
    assert "experiment_number" in combined_df.columns, "Missing 'experiment number' column in combined_df"

    grouped_df = combined_df.groupby(["query", "experiment_label", "experiment_number"])
    for (query, experiment_label, experiment_number), group_df in grouped_df:

        df_copy = group_df.copy()
        df_copy.sort_values(by=["assigned_position"], inplace=True)

        # Check for instruction columns and create InstructionConfig
        instruction_config = None
        instruction_columns = ["brand", "color", "cheapest", "budget"]
        present_instruction_columns = [col for col in instruction_columns if col in df_copy.columns]
        
        if present_instruction_columns:
            # Create InstructionConfig with values from all present instruction columns
            instruction_config = InstructionConfig(
                brand=df_copy["brand"].dropna().iloc[0] if "brand" in present_instruction_columns and not df_copy["brand"].dropna().empty else None,
                color=df_copy["color"].dropna().iloc[0] if "color" in present_instruction_columns and not df_copy["color"].dropna().empty else None,
                cheapest=df_copy["cheapest"].dropna().any() if "cheapest" in present_instruction_columns and not df_copy["cheapest"].dropna().empty else None,
                budget=df_copy["budget"].dropna().iloc[0] if "budget" in present_instruction_columns and not df_copy["budget"].dropna().empty else None
            )

        yield ExperimentData(
            query=query,
            experiment_label=experiment_label,
            experiment_number=experiment_number,
            experiment_df=df_copy,
            dataset_name=dataset_name,
            instruction_config=instruction_config,
        )

def load_experiment_data(csv_path: str) -> pd.DataFrame:
    """
    Load experiment data from a CSV file.
    
    Parameters:
    csv_path (str): Path to the CSV file containing experiment data.
    
    Returns:
    pd.DataFrame: The loaded experiment data.
    """
    return pd.read_csv(csv_path)


def hf_experiments_iter(dataset_name: str, subset: Optional[str] = None, split: str = "data") -> Iterable[ExperimentData]:
    """
    Load a HuggingFace dataset and iterate over experiments.
    
    In HF datasets, each row represents one experiment with:
    - Scalar columns: experiment_label, query, experiment_number
    - Sequence columns: All other columns contain lists of values for the experiment
    
    Parameters:
        dataset_name (str): Name of the dataset on HuggingFace Hub
        subset (str): Subset/configuration of the dataset to load
        split (str): Split to load (default: "data")

    Yields:
        ExperimentData: Experiment configuration with expanded DataFrame
    """
    # Load the dataset
    if subset is None:
        dataset = load_dataset(dataset_name, split=split)
    else:
        dataset = load_dataset(dataset_name, subset, split=split)
    
    for row in dataset:
        # Extract scalar values
        query = row['query']
        experiment_label = row['experiment_label']
        experiment_number = row['experiment_number']
        
        # Find sequence columns (those that contain lists)
        sequence_columns = {}
        scalar_columns = {'query': query, 'experiment_label': experiment_label, 'experiment_number': experiment_number}
        
        for col_name, col_value in row.items():
            if col_name not in scalar_columns and isinstance(col_value, list):
                sequence_columns[col_name] = col_value
        
        # Determine the number of products in this experiment
        num_products = len(next(iter(sequence_columns.values()))) if sequence_columns else 0
        
        # Create a DataFrame from the sequences
        df_data = {
            'query': [query] * num_products,
            'experiment_label': [experiment_label] * num_products,
            'experiment_number': [experiment_number] * num_products
        }
        
        # Add all sequence columns
        for col_name, col_values in sequence_columns.items():
            df_data[col_name] = col_values
        
        experiment_df = pd.DataFrame(df_data)
        
        # Sort by assigned_position if it exists
        if 'assigned_position' in experiment_df.columns:
            experiment_df.sort_values(by=['assigned_position'], inplace=True)
        
        # Check for instruction columns and create InstructionConfig
        instruction_config = None
        instruction_columns = ["brand", "color", "cheapest", "budget"]
        present_instruction_columns = [col for col in instruction_columns if col in experiment_df.columns]
        
        if present_instruction_columns:
            instruction_config = InstructionConfig(
                brand=experiment_df["brand"].dropna().iloc[0] if "brand" in present_instruction_columns and not experiment_df["brand"].isna().all() else None,
                color=experiment_df["color"].dropna().iloc[0] if "color" in present_instruction_columns and not experiment_df["color"].isna().all() else None,
                cheapest=experiment_df["cheapest"].dropna().any() if "cheapest" in present_instruction_columns and not experiment_df["cheapest"].isna().all() else None,
                budget=float(experiment_df["budget"].dropna().iloc[0]) if "budget" in present_instruction_columns and not experiment_df["budget"].isna().all() else None
            )
        
        screenshot: Image = row.get('screenshot')
        
        experiment_data = ExperimentData(
            query=query,
            experiment_label=experiment_label,
            experiment_number=experiment_number,
            experiment_df=experiment_df,
            dataset_name=dataset_name,
            instruction_config=instruction_config,
            screenshot=screenshot,
        )
        
        yield experiment_data