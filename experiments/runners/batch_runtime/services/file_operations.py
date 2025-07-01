"""
File operations service for batch processing.

Centralizes all file I/O operations used across batch providers and runtime.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich import print as _print


@dataclass
class BatchDirectories:
    """Standard directory structure for batch operations."""
    batch_metadata: Path
    batch_results: Path
    batch_inputs: Path
    
    def create_all(self):
        """Create all directories if they don't exist."""
        for directory in [self.batch_metadata, self.batch_results, self.batch_inputs]:
            directory.mkdir(parents=True, exist_ok=True)


class FileOperationsService:
    """Centralized file operations for batch processing."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.directories = BatchDirectories(
            batch_metadata=self.output_dir / "batch_metadata",
            batch_results=self.output_dir / "batch_results", 
            batch_inputs=self.output_dir / "batch_inputs"
        )
        self.directories.create_all()
    
    def write_jsonl_file(self, data: List[Dict[str, Any]], file_path: Path) -> Path:
        """
        Write data to a JSONL file with error handling.
        
        Args:
            data: List of dictionaries to write
            file_path: Path to write the file
            
        Returns:
            Path to the written file
            
        Raises:
            IOError: If file writing fails
        """
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
            
            _print(f"[bold blue]Wrote {len(data)} items to {file_path}")
            return file_path
            
        except Exception as e:
            _print(f"[bold red]Error writing JSONL file {file_path}: {e}")
            raise IOError(f"Failed to write JSONL file: {e}")
    
    def read_json_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Read JSON file with error handling.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            _print(f"[bold yellow]JSON file not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            _print(f"[bold red]Invalid JSON in file {file_path}: {e}")
            raise
    
    def write_json_file(self, data: Dict[str, Any], file_path: Path) -> Path:
        """
        Write JSON file with error handling.
        
        Args:
            data: Dictionary to write
            file_path: Path to write the file
            
        Returns:
            Path to the written file
        """
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return file_path
            
        except Exception as e:
            _print(f"[bold red]Error writing JSON file {file_path}: {e}")
            raise IOError(f"Failed to write JSON file: {e}")
    
    def read_jsonl_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Read JSONL file with error handling.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List of parsed JSON objects
        """
        results = []
        
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        if line.strip():  # Skip empty lines
                            results.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        _print(f"[bold yellow]Skipping invalid JSON on line {line_num} in {file_path}: {e}")
            
            return results
            
        except FileNotFoundError:
            _print(f"[bold yellow]JSONL file not found: {file_path}")
            return []
        except Exception as e:
            _print(f"[bold red]Error reading JSONL file {file_path}: {e}")
            return []
    
    def generate_unique_filename(self, base_name: str, config_name: str, 
                                extension: str = ".jsonl", timestamp: Optional[int] = None) -> str:
        """
        Generate unique filename with timestamp and config name.
        
        Args:
            base_name: Base name for the file
            config_name: Configuration name to include
            extension: File extension
            timestamp: Optional timestamp (uses current time if None)
            
        Returns:
            Unique filename
        """
        if timestamp is None:
            timestamp = int(time.time() * 1000)
        
        return f"{config_name}_{base_name}_{timestamp}{extension}"
    
    def ensure_directory_exists(self, directory: Path) -> Path:
        """
        Ensure directory exists, creating it if necessary.
        
        Args:
            directory: Directory path to create
            
        Returns:
            The directory path
        """
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    
    def get_batch_input_file(self, config_name: str, chunk_name: str) -> Path:
        """Get path for batch input file."""
        filename = f"{config_name}_{chunk_name}_requests.jsonl"
        return self.directories.batch_inputs / filename
    
    def get_batch_results_file(self, config_name: str, batch_id: str, timestamp: Optional[int] = None) -> Path:
        """Get path for batch results file."""
        filename = self.generate_unique_filename("results", config_name, ".jsonl", timestamp)
        if batch_id:
            # Include batch_id in filename for uniqueness
            filename = f"{config_name}_{batch_id}_{timestamp or int(time.time() * 1000)}_results.jsonl"
        return self.directories.batch_results / filename
    
    def get_batch_metadata_file(self, config_name: str) -> Path:
        """Get path for batch metadata file."""
        return self.directories.batch_metadata / f"{config_name}_batch_metadata.json"
    
    def get_experiment_mapping_file(self) -> Path:
        """Get path for centralized experiment batch mapping file."""
        return self.directories.batch_metadata / "experiment_batch_mapping.json"