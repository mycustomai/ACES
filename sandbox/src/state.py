from typing import List, Dict, Any
import pandas as pd

# In-memory state for experiment data
_current_experiment_data: List[Dict[str, Any]] = []

def set_experiment_data(data: pd.DataFrame | List[Dict[str, Any]]):
    """Set the current experiment data for the server to use."""
    global _current_experiment_data
    if isinstance(data, pd.DataFrame):
        _current_experiment_data = data.to_dict('records')
    else:
        _current_experiment_data = data

def get_experiment_data() -> List[Dict[str, Any]]:
    """Get the current experiment data."""
    return _current_experiment_data

def clear_experiment_data():
    """Clear the current experiment data."""
    global _current_experiment_data
    _current_experiment_data = []