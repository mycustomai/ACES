"""
Experiment tracking service for batch processing.

Centralizes all experiment status tracking and batch mapping operations.
"""

import asyncio
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from rich import print as _print

from agent.src.typedefs import EngineParams
from experiments.config import ExperimentData
from experiments.data_loader import experiments_iter


class ExperimentStatus(Enum):
    """Status of an individual experiment."""
    NOT_SUBMITTED = "not_submitted"
    SUBMITTED = "submitted" 
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExperimentStatusInfo:
    """Detailed status information for an experiment."""
    experiment_id: str
    status: ExperimentStatus
    batch_id: Optional[str] = None
    submitted_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    
    @property
    def is_outstanding(self) -> bool:
        """True if experiment needs processing (not submitted or not completed)."""
        return self.status in [ExperimentStatus.NOT_SUBMITTED, ExperimentStatus.SUBMITTED, ExperimentStatus.FAILED]


class ExperimentTrackingService:
    """Centralized experiment status tracking and batch mapping."""
    
    def __init__(self, output_dir: Path, dataset_name: str):
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name

        self.batch_metadata_dir = self.output_dir / "batch_metadata"
        self.batch_metadata_dir.mkdir(parents=True, exist_ok=True)
        
        self.mapping_file = self.batch_metadata_dir / "experiment_batch_mapping.json"
        self.submitted_experiments_file = self.batch_metadata_dir / "submitted_experiments.json"
        
        # Async lock for thread-safe operations
        self._lock = asyncio.Lock()
    
    async def get_experiment_status(self, config_name: str, experiments_df, engine_params: Optional[EngineParams] = None) -> Dict[str, ExperimentStatusInfo]:
        """
        Get comprehensive status for all experiments for a given configuration.
        
        Args:
            config_name: Configuration name
            experiments_df: DataFrame containing experiments
            engine_params: Optional engine parameters for better path construction
            
        Returns:
            Dictionary mapping experiment_id to ExperimentStatusInfo
        """
        async with self._lock:
            # Get all experiments from the dataset
            dataset_experiments = {data.experiment_id for data in experiments_iter(experiments_df, dataset_name=self.dataset_name)}
            
            # Load submission status from tracking files
            _print(f"[dim]Getting experiment status for {config_name}...")
            submission_status = self._load_submission_status(config_name, dataset_experiments)
            
            # Check completion status by examining filesystem
            _print(f"[dim]Checking completion status for {config_name}...")
            completion_status = await self._check_completion_status(config_name, experiments_df, engine_params)
            
            # Load batch mapping data once for all experiments (optimization)
            _print(f"[dim]Loading batch mapping data for {config_name}...")
            batch_mapping = self._load_batch_mapping_data(config_name)
            
            # Combine into comprehensive status
            _print(f"[dim]Combining status for {len(dataset_experiments)} experiments...")
            comprehensive_status = {}
            
            for exp_id in dataset_experiments:
                submitted = submission_status.get(exp_id, False)
                completed = completion_status.get(exp_id, False)
                
                if completed:
                    status = ExperimentStatus.COMPLETED
                elif submitted:
                    status = ExperimentStatus.SUBMITTED
                else:
                    status = ExperimentStatus.NOT_SUBMITTED
                
                # Get batch_id from cached mapping data
                batch_id = batch_mapping.get(exp_id)
                
                comprehensive_status[exp_id] = ExperimentStatusInfo(
                    experiment_id=exp_id,
                    status=status,
                    batch_id=batch_id
                )
            
            return comprehensive_status
    
    async def mark_experiments_submitted(self, config_name: str, experiment_ids: List[str], 
                                       batch_ids: List[str]) -> None:
        """
        Mark experiments as submitted with thread safety.
        
        Args:
            config_name: Configuration name
            experiment_ids: List of experiment IDs that were submitted
            batch_ids: List of batch IDs they were submitted to
        """
        if not experiment_ids or not batch_ids:
            _print(f"[bold yellow]Warning: Empty experiment IDs or batch IDs for {config_name}")
            return
        
        async with self._lock:
            # Update the detailed experiment-to-batch mapping
            await self._update_experiment_batch_mapping(config_name, experiment_ids, batch_ids)
            
            # Update the simple submitted experiments tracking
            await self._update_submitted_experiments_tracking(config_name, experiment_ids)
    
    async def get_outstanding_experiments(self, config_name: str, all_experiments: List[ExperimentData],
                                        experiments_df, force_submit: bool = False, engine_params: Optional[EngineParams] = None) -> List[ExperimentData]:
        """
        Get list of experiments that need processing.
        
        Args:
            config_name: Configuration name
            all_experiments: List of all experiments
            experiments_df: DataFrame containing experiments
            force_submit: If True, only skip completed experiments
            engine_params: Optional engine parameters for better completion checking
            
        Returns:
            List of experiments that need processing
        """
        status_map = await self.get_experiment_status(config_name, experiments_df, engine_params)
        _print(f"[dim]Finished getting experiment status for {config_name}")
        outstanding_experiments = []
        
        for experiment in all_experiments:
            exp_status = status_map.get(experiment.experiment_id)
            
            if exp_status is None:
                # Unknown experiment, include it
                outstanding_experiments.append(experiment)
                continue
            
            if force_submit:
                # When force_submit is enabled, only skip completed experiments
                if exp_status.status != ExperimentStatus.COMPLETED:
                    outstanding_experiments.append(experiment)
            else:
                # Normal behavior: include outstanding experiments
                if exp_status.is_outstanding:
                    outstanding_experiments.append(experiment)
        
        return outstanding_experiments
    
    def print_status_report(self, config_name: str, status_map: Dict[str, ExperimentStatusInfo]) -> None:
        """Print a detailed status report for experiments."""
        _print(f"\\n[bold blue]══════ Experiment Status Report for {config_name} ══════")
        
        total = len(status_map)
        submitted = sum(1 for s in status_map.values() if s.status == ExperimentStatus.SUBMITTED)
        completed = sum(1 for s in status_map.values() if s.status == ExperimentStatus.COMPLETED)
        outstanding = sum(1 for s in status_map.values() if s.is_outstanding)
        
        _print(f"[bold green]Total Experiments: {total}")
        _print(f"[bold yellow]Submitted: {submitted}")
        _print(f"[bold green]Completed: {completed}")
        _print(f"[bold red]Outstanding: {outstanding}")
        _print(f"[bold blue]══════ End Status Report ══════\\n")
    
    def _load_submission_status(self, config_name: str, dataset_experiments: Set[str]) -> Dict[str, bool]:
        """Load submission status from tracking files."""
        if not self.mapping_file.exists():
            return {exp_id: False for exp_id in dataset_experiments}
        
        try:
            with open(self.mapping_file, 'r') as f:
                mapping_data = json.load(f)
            
            if config_name not in mapping_data:
                return {exp_id: False for exp_id in dataset_experiments}
            
            provider_data = mapping_data[config_name]
            experiment_status = {}
            
            for exp_id in dataset_experiments:
                if exp_id in provider_data.get("experiments", {}):
                    exp_data = provider_data["experiments"][exp_id]
                    # Check if experiment has valid batch_id
                    batch_id = exp_data.get("batch_id")
                    if batch_id and isinstance(batch_id, str) and batch_id.strip():
                        experiment_status[exp_id] = True
                    else:
                        experiment_status[exp_id] = False
                else:
                    experiment_status[exp_id] = False
            
            return experiment_status
            
        except (json.JSONDecodeError, KeyError) as e:
            _print(f"[bold yellow]Warning: Could not load submission status for {config_name}: {e}")
            return {exp_id: False for exp_id in dataset_experiments}
    
    async def _check_completion_status(self, config_name: str, experiments_df, engine_params: Optional[EngineParams] = None) -> Dict[str, bool]:
        """Check completion status by examining filesystem for experiment_data.csv files using batch operations."""
        completion_status = {}
        
        try:
            # Collect all experiments for batch processing
            experiments_list = list(experiments_iter(experiments_df, self.dataset_name))
            
            # Use batch filesystem checking for better performance with large datasets
            completion_status = await self._check_completion_status_batch(config_name, experiments_list, engine_params)
                
        except Exception as e:
            _print(f"[bold yellow]Warning: Could not check completion status for {config_name}: {e}")
            # Return False for all experiments if we can't check
            for data in experiments_iter(experiments_df, self.dataset_name):
                completion_status[data.experiment_id] = False
            
        return completion_status
    
    async def _check_completion_status_batch(self, config_name: str, experiments_list: List[ExperimentData],
                                     engine_params: Optional[EngineParams] = None) -> Dict[str, bool]:
        """
        Batch check completion status for multiple experiments efficiently.
        
        This method builds all file paths first, then checks their existence in batch
        for improved performance with large datasets.
        """
        completion_status = {}
        path_to_experiment_mapping = {}
        
        # Build all expected paths in batch
        for experiment_data in experiments_list:
            try:
                if engine_params:
                    journey_output_dir = experiment_data.journey_dir(self.output_dir, engine_params)
                    expected_path = journey_output_dir / "experiment_data.csv"
                else:
                    # Fallback: reverse-engineer the provider and model from config_name
                    provider_model = config_name  # e.g., "Gemini_gemini-2.0-flash-001"
                    product_name = experiment_data.query
                    experiment_folder = f"{experiment_data.experiment_label}_{experiment_data.experiment_number}"
                    expected_path = (
                        self.output_dir / 
                        provider_model / 
                        product_name / 
                        experiment_folder / 
                        "experiment_data.csv"
                    )
                
                path_to_experiment_mapping[expected_path] = experiment_data.experiment_id
                
            except Exception as e:
                _print(f"[bold yellow]Warning: Could not build path for experiment {experiment_data.experiment_id}: {e}")
                completion_status[experiment_data.experiment_id] = False
        
        # Batch check file existence using thread pool for non-blocking operations
        completed_count = 0
        
        def check_path_exists(path_exp_pair):
            """Synchronous helper to check if a path exists."""
            expected_path, experiment_id = path_exp_pair
            try:
                is_completed = expected_path.exists()
                return experiment_id, is_completed
            except Exception as e:
                _print(f"[bold yellow]Warning: Could not check completion for experiment {experiment_id}: {e}")
                return experiment_id, False
        
        # Process filesystem checks in chunks using thread pool
        path_items = list(path_to_experiment_mapping.items())
        chunk_size = 100  # Process 100 files at a time
        
        for i in range(0, len(path_items), chunk_size):
            chunk = path_items[i:i + chunk_size]

            try:
                # Use asyncio.to_thread to run filesystem operations in thread pool
                tasks = [asyncio.to_thread(check_path_exists, item) for item in chunk]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        _print(f"[bold yellow]Warning: Exception in filesystem check: {result}")
                        continue
                    experiment_id, is_completed = result
                    completion_status[experiment_id] = is_completed
                    if is_completed:
                        completed_count += 1
                        
            except Exception as e:
                _print(f"[bold yellow]Warning: Chunk processing failed, falling back to synchronous: {e}")
                # Fallback to synchronous processing for this chunk
                for expected_path, experiment_id in chunk:
                    try:
                        is_completed = expected_path.exists()
                        completion_status[experiment_id] = is_completed
                        if is_completed:
                            completed_count += 1
                    except Exception:
                        completion_status[experiment_id] = False
        
        total_experiments = len(experiments_list)
        _print(f"[bold cyan]Completion check for {config_name}: {completed_count}/{total_experiments} experiments completed")
        
        return completion_status
    
    def _check_experiment_completion(self, config_name: str, experiment_data: ExperimentData, engine_params: Optional[EngineParams] = None) -> bool:
        """
        Check if a specific experiment is completed by looking for experiment_data.csv.
        
        Use the standard output directory structure:
        output_dir/{Provider}_{ModelName}/{product_name}/{experiment_label}_{experiment_number}/experiment_data.csv
        """
        try:
            # Use engine_params if available for more accurate path construction
            if engine_params:
                # Use the proper journey_dir method from ExperimentData
                journey_output_dir = experiment_data.journey_dir(self.output_dir, engine_params)
                expected_path = journey_output_dir / "experiment_data.csv"
            else:
                # Fallback: reverse-engineer the provider and model from config_name
                provider_model = config_name  # e.g., "Gemini_gemini-2.0-flash-001"
                
                # Build path: output_dir/{Provider}_{ModelName}/{product_name}/{experiment_label}_{experiment_number}/
                product_name = experiment_data.query  # The search query is used as product name
                experiment_folder = f"{experiment_data.experiment_label}_{experiment_data.experiment_number}"
                
                # Construct the expected path
                expected_path = (
                    self.output_dir / 
                    provider_model / 
                    product_name / 
                    experiment_folder / 
                    "experiment_data.csv"
                )
            
            # Check if the completion file exists
            is_completed = expected_path.exists()
            
            if is_completed:
                _print(f"[bold green]✓ Found completion file: {expected_path}")
            else:
                _print(f"[dim]Missing completion file: {expected_path}")
                
            return is_completed
            
        except Exception as e:
            _print(f"[bold yellow]Warning: Could not check completion for experiment {experiment_data.experiment_id}: {e}")
            return False
    
    def _load_batch_mapping_data(self, config_name: str) -> Dict[str, str]:
        """Load batch mapping data once for all experiments to avoid repeated file I/O."""
        if not self.mapping_file.exists():
            return {}
        
        try:
            with open(self.mapping_file, 'r') as f:
                mapping_data = json.load(f)
            
            if config_name in mapping_data:
                experiments = mapping_data[config_name].get("experiments", {})
                # Return a dict mapping experiment_id -> batch_id
                return {exp_id: exp_data.get("batch_id") for exp_id, exp_data in experiments.items() if exp_data.get("batch_id")}
                    
        except Exception:
            pass
        
        return {}
    
    def _get_experiment_batch_id(self, config_name: str, experiment_id: str) -> Optional[str]:
        """Get batch_id for a specific experiment."""
        if not self.mapping_file.exists():
            return None
        
        try:
            with open(self.mapping_file, 'r') as f:
                mapping_data = json.load(f)
            
            if config_name in mapping_data:
                experiments = mapping_data[config_name].get("experiments", {})
                if experiment_id in experiments:
                    return experiments[experiment_id].get("batch_id")
                    
        except Exception:
            pass
        
        return None
    
    async def _update_experiment_batch_mapping(self, config_name: str, experiment_ids: List[str], 
                                             batch_ids: List[str]) -> None:
        """Update the detailed experiment-to-batch mapping file."""
        if not experiment_ids or not batch_ids:
            return
        
        # Load existing mapping
        mapping_data = {}
        if self.mapping_file.exists():
            try:
                with open(self.mapping_file, 'r') as f:
                    mapping_data = json.load(f)
            except json.JSONDecodeError:
                _print(f"[bold yellow]Warning: Could not parse existing mapping file")
        
        # Initialize provider entry if needed
        if config_name not in mapping_data:
            mapping_data[config_name] = {
                "batches": {},
                "experiments": {}
            }
        
        # Record batch information
        for batch_id in batch_ids:
            if batch_id not in mapping_data[config_name]["batches"]:
                mapping_data[config_name]["batches"][batch_id] = {
                    "experiment_ids": [],
                    "submitted_at": asyncio.get_event_loop().time(),
                    "status": "submitted"
                }
            
            # Add experiments to this batch (avoid duplicates)
            existing_experiments = set(mapping_data[config_name]["batches"][batch_id]["experiment_ids"])
            new_experiments = [exp_id for exp_id in experiment_ids if exp_id not in existing_experiments]
            mapping_data[config_name]["batches"][batch_id]["experiment_ids"].extend(new_experiments)
        
        # Record reverse lookup: experiment -> batch
        for experiment_id in experiment_ids:
            mapping_data[config_name]["experiments"][experiment_id] = {
                "batch_id": batch_ids[0],  # Use first batch ID
                "submitted_at": asyncio.get_event_loop().time()
            }
        
        # Write updated mapping
        with open(self.mapping_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)
    
    async def _update_submitted_experiments_tracking(self, config_name: str, experiment_ids: List[str]) -> None:
        """Update simple submitted experiments tracking file."""
        submitted_data = {}
        
        # Load existing data
        if self.submitted_experiments_file.exists():
            try:
                with open(self.submitted_experiments_file, 'r') as f:
                    submitted_data = json.load(f)
            except json.JSONDecodeError:
                pass
        
        # Update data for this provider
        if config_name not in submitted_data:
            submitted_data[config_name] = []
        
        # Add new experiment IDs (avoid duplicates)
        existing_ids = set(submitted_data[config_name])
        new_ids = [exp_id for exp_id in experiment_ids if exp_id not in existing_ids]
        submitted_data[config_name].extend(new_ids)
        
        # Write updated data
        with open(self.submitted_experiments_file, 'w') as f:
            json.dump(submitted_data, f, indent=2)
    
    async def get_submitted_batch_ids(self, config_name: str) -> List[str]:
        """
        Get all submitted batch IDs for a given configuration.
        
        Args:
            config_name: Configuration name to get batch IDs for
            
        Returns:
            List of batch IDs that have been submitted but may need status refresh
        """
        async with self._lock:
            if not self.mapping_file.exists():
                return []
            
            try:
                with open(self.mapping_file, 'r') as f:
                    mapping_data = json.load(f)
                
                if config_name not in mapping_data:
                    return []
                
                provider_data = mapping_data[config_name]
                batch_info = provider_data.get('batches', {})
                
                # Return all batch IDs that are currently tracked
                return list(batch_info.keys())
                
            except Exception as e:
                _print(f"[bold yellow]Warning: Could not load batch IDs for {config_name}: {e}")
                return []
    
    async def update_batch_status(self, config_name: str, batch_id: str, new_status: str) -> None:
        """
        Update the status of a batch in the metadata.
        
        Args:
            config_name: Configuration name
            batch_id: Batch ID to update
            new_status: New status for the batch (e.g., 'completed', 'failed')
        """
        async with self._lock:
            if not self.mapping_file.exists():
                return
            
            try:
                with open(self.mapping_file, 'r') as f:
                    mapping_data = json.load(f)
                
                if config_name in mapping_data and 'batches' in mapping_data[config_name]:
                    if batch_id in mapping_data[config_name]['batches']:
                        mapping_data[config_name]['batches'][batch_id]['status'] = new_status
                        mapping_data[config_name]['batches'][batch_id]['updated_at'] = asyncio.get_event_loop().time()
                        
                        # Write updated mapping
                        with open(self.mapping_file, 'w') as f:
                            json.dump(mapping_data, f, indent=2)
                        
                        _print(f"[bold cyan]Updated batch {batch_id} status to: {new_status}")
                
            except Exception as e:
                _print(f"[bold yellow]Warning: Could not update batch status for {batch_id}: {e}")
    
    async def validate_metadata_against_provider(self, config_name: str, provider) -> Dict[str, Any]:
        """
        Validate local metadata against provider-side batch information.
        
        This method checks if batches recorded in local metadata actually exist
        on the provider side and provides a detailed validation report.
        
        Args:
            config_name: Configuration name to validate
            provider: The batch provider instance to query
            
        Returns:
            Dictionary containing validation results and recommendations
        """
        async with self._lock:
            validation_report = {
                'config_name': config_name,
                'validation_timestamp': asyncio.get_event_loop().time(),
                'batches_in_metadata': 0,
                'valid_batches': [],
                'invalid_batches': [],
                'orphaned_experiments': [],
                'recommendations': []
            }
            
            # Load existing mapping data
            if not self.mapping_file.exists():
                validation_report['recommendations'].append("No metadata file found - no validation needed")
                return validation_report
            
            try:
                with open(self.mapping_file, 'r') as f:
                    mapping_data = json.load(f)
                
                if config_name not in mapping_data:
                    validation_report['recommendations'].append(f"No metadata found for {config_name}")
                    return validation_report
                
                provider_data = mapping_data[config_name]
                batch_info = provider_data.get('batches', {})
                validation_report['batches_in_metadata'] = len(batch_info)
                
                _print(f"[bold blue]Validating metadata for {config_name}: {len(batch_info)} batches to check...")
                
                # Use batch monitoring for efficient validation of large datasets
                batch_ids = list(batch_info.keys())
                
                try:
                    # Try to use the new concurrent monitoring method for efficiency
                    if hasattr(provider, 'monitor_batches_concurrent'):
                        status_map = await provider.monitor_batches_concurrent(batch_ids)
                    elif hasattr(provider, 'monitor_batches'):
                        status_map = await provider.monitor_batches(batch_ids)
                    else:
                        # Fallback to individual status checks
                        status_map = {}
                        for batch_id in batch_ids:
                            try:
                                if hasattr(provider, '_get_batch_status'):
                                    status_map[batch_id] = await provider._get_batch_status(batch_id)
                                else:
                                    status_map[batch_id] = 'unknown'
                            except Exception as e:
                                status_map[batch_id] = 'error'
                                _print(f"[bold red]✗ Error checking batch {batch_id}: {e}")
                    
                    # Process the batched results
                    for batch_id, batch_data in batch_info.items():
                        status = status_map.get(batch_id, 'error')
                        
                        if status and status not in ['error', 'unknown']:
                            validation_report['valid_batches'].append({
                                'batch_id': batch_id,
                                'status': status,
                                'experiment_count': len(batch_data.get('experiment_ids', []))
                            })
                            _print(f"[bold green]✓ Batch {batch_id} is valid (status: {status})")
                        elif status == 'unknown':
                            # Provider doesn't support batch validation
                            validation_report['valid_batches'].append({
                                'batch_id': batch_id,
                                'status': 'unknown',
                                'experiment_count': len(batch_data.get('experiment_ids', []))
                            })
                            _print(f"[bold yellow]Provider doesn't support batch validation for {batch_id}")
                        else:
                            validation_report['invalid_batches'].append({
                                'batch_id': batch_id,
                                'status': status,
                                'experiment_ids': batch_data.get('experiment_ids', []),
                                'submitted_at': batch_data.get('submitted_at')
                            })
                            _print(f"[bold red]✗ Batch {batch_id} is invalid (status: {status})")
                    
                except Exception as e:
                    # Fallback to individual validation if batch monitoring fails
                    _print(f"[bold yellow]Batch monitoring failed, falling back to individual checks: {e}")
                    
                    for batch_id, batch_data in batch_info.items():
                        try:
                            if hasattr(provider, '_get_batch_status'):
                                status = await provider._get_batch_status(batch_id)
                                if status and status != 'error':
                                    validation_report['valid_batches'].append({
                                        'batch_id': batch_id,
                                        'status': status,
                                        'experiment_count': len(batch_data.get('experiment_ids', []))
                                    })
                                    _print(f"[bold green]✓ Batch {batch_id} is valid (status: {status})")
                                else:
                                    validation_report['invalid_batches'].append({
                                        'batch_id': batch_id,
                                        'status': status,
                                        'experiment_ids': batch_data.get('experiment_ids', []),
                                        'submitted_at': batch_data.get('submitted_at')
                                    })
                                    _print(f"[bold red]✗ Batch {batch_id} is invalid (status: {status})")
                            else:
                                validation_report['valid_batches'].append({
                                    'batch_id': batch_id,
                                    'status': 'unknown',
                                    'experiment_count': len(batch_data.get('experiment_ids', []))
                                })
                                _print(f"[bold yellow]Provider doesn't support batch validation for {batch_id}")
                        
                        except Exception as e:
                            validation_report['invalid_batches'].append({
                                'batch_id': batch_id,
                                'error': str(e),
                                'experiment_ids': batch_data.get('experiment_ids', []),
                                'submitted_at': batch_data.get('submitted_at')
                            })
                            _print(f"[bold red]✗ Error validating batch {batch_id}: {e}")
                
                # Identify orphaned experiments (experiments with invalid batch_ids)
                for invalid_batch in validation_report['invalid_batches']:
                    validation_report['orphaned_experiments'].extend(
                        invalid_batch.get('experiment_ids', [])
                    )
                
                # Generate recommendations
                if validation_report['invalid_batches']:
                    validation_report['recommendations'].append(
                        f"Found {len(validation_report['invalid_batches'])} invalid batches - consider cleaning up metadata"
                    )
                    validation_report['recommendations'].append(
                        f"Found {len(validation_report['orphaned_experiments'])} orphaned experiments - consider resubmitting"
                    )
                
                if validation_report['valid_batches']:
                    validation_report['recommendations'].append(
                        f"Found {len(validation_report['valid_batches'])} valid batches - metadata is mostly accurate"
                    )
                
                _print(f"[bold cyan]Validation complete for {config_name}:")
                _print(f"  Valid batches: {len(validation_report['valid_batches'])}")
                _print(f"  Invalid batches: {len(validation_report['invalid_batches'])}")
                _print(f"  Orphaned experiments: {len(validation_report['orphaned_experiments'])}")
                
            except Exception as e:
                validation_report['error'] = str(e)
                validation_report['recommendations'].append(f"Error during validation: {e}")
                _print(f"[bold red]Error validating metadata for {config_name}: {e}")
            
            return validation_report
    
    async def cleanup_invalid_metadata(self, config_name: str, validation_report: Dict[str, Any]) -> bool:
        """
        Clean up invalid batch metadata based on validation report.
        
        Args:
            config_name: Configuration name to clean up
            validation_report: Validation report from validate_metadata_against_provider
            
        Returns:
            True if cleanup was successful, False otherwise
        """
        if not validation_report.get('invalid_batches'):
            _print(f"[bold green]No invalid metadata to clean up for {config_name}")
            return True
        
        async with self._lock:
            try:
                _print(f"[bold blue]Cleaning up invalid metadata for {config_name}...")
                
                # Load existing mapping data
                if not self.mapping_file.exists():
                    return True
                
                with open(self.mapping_file, 'r') as f:
                    mapping_data = json.load(f)
                
                if config_name not in mapping_data:
                    return True
                
                # Remove invalid batches from metadata
                provider_data = mapping_data[config_name]
                batches_to_remove = [batch['batch_id'] for batch in validation_report['invalid_batches']]
                experiments_to_remove = validation_report['orphaned_experiments']
                
                # Clean up batches section
                for batch_id in batches_to_remove:
                    if batch_id in provider_data.get('batches', {}):
                        del provider_data['batches'][batch_id]
                        _print(f"[bold yellow]Removed invalid batch {batch_id} from metadata")
                
                # Clean up experiments section
                for exp_id in experiments_to_remove:
                    if exp_id in provider_data.get('experiments', {}):
                        del provider_data['experiments'][exp_id]
                        _print(f"[bold yellow]Removed orphaned experiment {exp_id} from metadata")
                
                # Write updated mapping data
                with open(self.mapping_file, 'w') as f:
                    json.dump(mapping_data, f, indent=2)
                
                # Also clean up submitted experiments file
                if self.submitted_experiments_file.exists():
                    with open(self.submitted_experiments_file, 'r') as f:
                        submitted_data = json.load(f)
                    
                    if config_name in submitted_data:
                        original_count = len(submitted_data[config_name])
                        submitted_data[config_name] = [
                            exp_id for exp_id in submitted_data[config_name] 
                            if exp_id not in experiments_to_remove
                        ]
                        removed_count = original_count - len(submitted_data[config_name])
                        
                        with open(self.submitted_experiments_file, 'w') as f:
                            json.dump(submitted_data, f, indent=2)
                        
                        if removed_count > 0:
                            _print(f"[bold yellow]Removed {removed_count} orphaned experiments from submitted list")
                
                _print(f"[bold green]✓ Metadata cleanup completed for {config_name}")
                return True
                
            except Exception as e:
                _print(f"[bold red]Error during metadata cleanup for {config_name}: {e}")
                return False