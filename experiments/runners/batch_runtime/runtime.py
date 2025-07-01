"""
Simplified BatchEvaluationRuntime using composed services.

This runtime orchestrates batch processing by delegating responsibilities
to specialized services rather than handling everything internally.
"""

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from rich import print as _print

from agent.src.typedefs import EngineParams
from experiments.config import ExperimentData
from experiments.data_loader import experiments_iter, load_experiment_data
from experiments.runners.simple_runtime import BaseEvaluationRuntime

from ..services.gcs_upload import GCSUploadService
from ..services.screenshot_validation import ScreenshotValidationService
from .providers.anthropic import AnthropicBatchProvider
from .providers.gemini import GeminiBatchProvider
from .providers.openai import OpenAIBatchProvider
from .services.batch_operations import BatchOperationsService
from .services.experiment_tracking import ExperimentTrackingService
from .services.file_operations import FileOperationsService


class BatchEvaluationRuntime(BaseEvaluationRuntime):
    """
    Simplified batch evaluation runtime using composed services.
    
    This runtime orchestrates batch processing by delegating to specialized services
    rather than handling all operations internally.
    """
    
    def __init__(
        self, 
        local_dataset_path: str,
        engine_params_list: List[EngineParams], 
        output_dir_override: Optional[str] = None,
        experiment_count_limit: Optional[int] = None,
        experiment_label_filter: Optional[str] = None,
        debug_mode: bool = False,
        force_submit: bool = False,
        remote: bool = False
    ):
        """Initialize the simplified BatchEvaluationRuntime."""
        # Extract dataset name from path
        dataset_filename = os.path.splitext(os.path.basename(local_dataset_path))[0]
        dataset_name = dataset_filename.replace('_dataset', '')
        super().__init__(dataset_name, output_dir_override, debug_mode)
        
        # Store configuration
        self.local_dataset_path = local_dataset_path
        self.engine_params_list = engine_params_list
        self.experiment_count_limit = experiment_count_limit
        self.experiment_label_filter = experiment_label_filter
        self.force_submit = force_submit
        self.remote = remote
        
        # Load dataset
        self.experiments_df = load_experiment_data(local_dataset_path)
        
        # Initialize services
        self.file_ops = FileOperationsService(self.run_output_dir)
        self.tracking_service = ExperimentTrackingService(self.run_output_dir, self.dataset_name)
        
        # Set up screenshots directory
        dataset_dir = Path(local_dataset_path).parent
        self.screenshots_dir = dataset_dir / "screenshots" / self.dataset_name
        self.screenshot_service = ScreenshotValidationService(self.screenshots_dir, dataset_name)
        
        # Initialize GCS upload service if remote mode is enabled
        self.gcs_upload_service = None
        if self.remote:
            try:
                self.gcs_upload_service = GCSUploadService()
                _print(f"[bold green]✓ GCS upload service initialized for remote mode")
            except Exception as e:
                _print(f"[bold yellow]Warning: Could not initialize GCS upload service: {e}")
                _print(f"[bold yellow]Remote mode disabled")
        
        # Create providers
        self.providers = self._create_providers()
        
        # Create batch operations service
        self.batch_service = BatchOperationsService(
            providers=self.providers,
            tracking_service=self.tracking_service,
            file_ops=self.file_ops,
            dataset_name=self.dataset_name,
            # TODO: there has to be a more efficient method of passing the path.
            #  This is also tightly coupled with the fs env runtime / incompat w hf ds
            screenshots_dir=self.screenshots_dir,
            remote=self.remote,
            engine_params_list=engine_params_list,
        )
        
        # Filter supported engines
        self.supported_engines = [ep for ep in engine_params_list if ep.config_name in self.providers]
        
        if not self.supported_engines:
            raise ValueError("No supported engines found for batch processing")
        
        _print(f"[bold green]Initialized BatchEvaluationRuntime with {len(self.supported_engines)} supported engines")
    
    @property
    def experiments_iter(self) -> Iterable[ExperimentData]:
        """Return an iterator over experiments from the local dataset."""
        return experiments_iter(self.experiments_df, self.dataset_name)
    
    def _create_providers(self) -> Dict[str, Any]:
        """Create batch providers for supported engine types."""
        providers = {}
        
        for engine in self.engine_params_list:
            engine_type = engine.engine_type.lower()
            config_name = engine.config_name
            
            try:
                if engine_type == "openai":
                    providers[config_name] = OpenAIBatchProvider(
                        file_ops=self.file_ops,
                        dataset_name=self.dataset_name,
                        remote=self.remote
                    )
                elif engine_type == "anthropic":
                    providers[config_name] = AnthropicBatchProvider(
                        file_ops=self.file_ops,
                        dataset_name=self.dataset_name,
                        remote=self.remote
                    )
                elif engine_type == "gemini":
                    providers[config_name] = GeminiBatchProvider(
                        file_ops=self.file_ops,
                        screenshots_dir=self.screenshots_dir,
                        dataset_name=self.dataset_name,
                        skip_upload=self.remote  # Skip upload in Gemini if batch runtime handles it
                    )
                else:
                    _print(f"[bold yellow]Unsupported engine type: {engine_type}")
                    
            except Exception as e:
                _print(f"[bold yellow]Failed to create provider for {config_name}: {e}")
        
        return providers
    
    async def _validate_metadata_at_startup(self):
        """Validate metadata against providers at startup for each supported engine."""
        _print(f"[bold blue]Validating metadata against providers at startup...")
        
        validation_results = {}
        for engine_params in self.supported_engines:
            config_name = engine_params.config_name
            
            try:
                _print(f"[bold cyan]Validating metadata for {config_name}...")
                validation_report = await self.batch_service.validate_and_rectify_metadata(
                    config_name, auto_cleanup=True  # Auto-cleanup invalid metadata at startup
                )
                
                validation_results[config_name] = validation_report
                
                # Log summary
                if 'summary' in validation_report:
                    summary = validation_report['summary']
                    success_rate = summary.get('validation_success_rate', 0)
                    
                    if success_rate == 100:
                        _print(f"[bold green]✓ {config_name}: All metadata valid (100%)")
                    elif success_rate >= 80:
                        _print(f"[bold yellow]⚠ {config_name}: Mostly valid metadata ({success_rate:.1f}%)")
                    else:
                        _print(f"[bold red]✗ {config_name}: Poor metadata quality ({success_rate:.1f}%)")
                    
                    if validation_report.get('cleanup_performed'):
                        _print(f"[bold yellow]  Cleaned up {summary['invalid_batches']} invalid batches")
                        _print(f"[bold yellow]  Recovered {summary['orphaned_experiments']} orphaned experiments")
                
            except Exception as e:
                _print(f"[bold red]Failed to validate metadata for {config_name}: {e}")
                validation_results[config_name] = {'error': str(e)}
        
        # Summary of all validations
        total_providers = len(self.supported_engines)
        successful_validations = sum(1 for result in validation_results.values() if 'error' not in result)
        
        _print(f"[bold cyan]Metadata validation complete: {successful_validations}/{total_providers} providers validated")
        
        if successful_validations < total_providers:
            _print(f"[bold yellow]Some providers had validation errors - proceeding with caution")
    
    async def _refresh_existing_batches_at_startup(self):
        """
        Check for existing submitted batches and process any that have completed.
        
        This method refreshes the status of previously submitted batches and processes
        any newly completed batches that weren't previously downloaded, focusing on
        Gemini provider as requested.
        """
        _print(f"[bold blue]Refreshing existing batches at startup...")
        
        batch_refresh_results = {}
        
        for engine_params in self.supported_engines:
            config_name = engine_params.config_name
            provider = self.providers.get(config_name)
            
            if not provider:
                continue
            
            try:
                _print(f"[bold cyan]Refreshing existing batches for {config_name}...")
                
                # Get all submitted batch IDs for this provider
                existing_batch_ids = await self.tracking_service.get_submitted_batch_ids(config_name)
                
                if not existing_batch_ids:
                    _print(f"[bold green]No existing batches found for {config_name}")
                    batch_refresh_results[config_name] = {'existing_batches': 0, 'newly_completed': 0}
                    continue
                
                _print(f"[bold blue]Found {len(existing_batch_ids)} existing batches for {config_name}")
                
                # Check current status of all existing batches
                status_map = await provider.monitor_batches(existing_batch_ids)
                
                if not status_map:
                    _print(f"[bold yellow]Could not get status for existing batches in {config_name}")
                    batch_refresh_results[config_name] = {'existing_batches': len(existing_batch_ids), 'newly_completed': 0, 'error': 'status_check_failed'}
                    continue
                
                # Process newly completed batches
                newly_completed_batches = []
                
                for batch_id, status in status_map.items():
                    # Update batch status in metadata
                    await self.tracking_service.update_batch_status(config_name, batch_id, status)
                    
                    if status in ['completed', 'ended']:
                        # Check if this batch has already been processed by looking for results
                        if await self._is_batch_unprocessed(batch_id, config_name):
                            newly_completed_batches.append(batch_id)
                
                # Process the newly completed batches
                processed_count = 0
                for batch_id in newly_completed_batches:
                    try:
                        _print(f"[bold cyan]Processing newly completed batch {batch_id} for {config_name}...")
                        success = await self.batch_service._process_completed_batch(
                            provider, batch_id, config_name, 
                            self.run_output_dir, self.experiments_df
                        )
                        if success:
                            processed_count += 1
                            _print(f"[bold green]✓ Successfully processed completed batch {batch_id}")
                        else:
                            _print(f"[bold red]Failed to process completed batch {batch_id} - will retry later")
                    except Exception as e:
                        _print(f"[bold red]Error processing completed batch {batch_id}: {e}")
                
                batch_refresh_results[config_name] = {
                    'existing_batches': len(existing_batch_ids),
                    'newly_completed': processed_count,
                    'completed_batches': len([s for s in status_map.values() if s in ['completed', 'ended']]),
                    'in_progress_batches': len([s for s in status_map.values() if s in ['in_progress', 'running', 'pending']]),
                    'failed_batches': len([s for s in status_map.values() if s in ['failed', 'error', 'cancelled']])
                }
                
                _print(f"[bold green]✓ Batch refresh completed for {config_name}:")
                _print(f"  Total existing batches: {batch_refresh_results[config_name]['existing_batches']}")
                _print(f"  Newly processed batches: {batch_refresh_results[config_name]['newly_completed']}")
                _print(f"  Status distribution: {batch_refresh_results[config_name]['completed_batches']} completed, "
                      f"{batch_refresh_results[config_name]['in_progress_batches']} in progress, "
                      f"{batch_refresh_results[config_name]['failed_batches']} failed")
                
            except Exception as e:
                _print(f"[bold red]Error refreshing batches for {config_name}: {e}")
                batch_refresh_results[config_name] = {'error': str(e)}
        
        # Summary
        total_existing = sum(r.get('existing_batches', 0) for r in batch_refresh_results.values())
        total_processed = sum(r.get('newly_completed', 0) for r in batch_refresh_results.values())
        
        _print(f"[bold cyan]Batch refresh summary: Found {total_existing} existing batches, processed {total_processed} newly completed batches")
        
        if total_processed > 0:
            _print(f"[bold green]✓ Successfully processed {total_processed} previously unprocessed completed batches")
    
    async def _is_batch_unprocessed(self, batch_id: str, config_name: str) -> bool:
        """
        Check if a completed batch still needs processing by looking for results files.
        
        Args:
            batch_id: The batch ID to check
            config_name: Configuration name
            
        Returns:
            True if the batch needs processing, False if already processed
        """
        try:
            # Check if results file exists for this batch
            results_file = self.file_ops.get_batch_results_file(config_name, batch_id)
            
            if not results_file.exists():
                # No results file means we haven't processed this batch yet
                return True
            
            # If results file exists but is empty, it needs reprocessing
            if results_file.stat().st_size == 0:
                return True
            
            # Additional check: see if any experiments from this batch have been processed
            # by checking for experiment output directories
            # This is a more thorough check for Gemini batches
            
            return False  # Assume processed if results file exists and has content
            
        except Exception as e:
            _print(f"[bold yellow]Warning: Could not check processing status for batch {batch_id}: {e}")
            # If we can't determine, assume it needs processing to be safe
            return True
    
    async def _upload_screenshots_to_gcs(self):
        """Upload screenshots to GCS for remote mode processing."""
        if not self.remote or not self.gcs_upload_service:
            return
        
        
        try:
            results = self.gcs_upload_service.upload_screenshots_batch(
                screenshots_dir=self.screenshots_dir,
                dataset_name=self.dataset_name,
                experiments_df=self.experiments_df,
                max_workers=6,
                skip_existing=True
            )
            
            if results.get("success"):
                uploaded = results.get("uploaded", 0)
                failed = results.get("failed", 0)

                if uploaded > 0:
                    _print(f"[bold green]✓ Uploaded {uploaded} screenshots to GCS")
                
                if failed > 0:
                    _print(f"[bold yellow]Warning: {failed} screenshots failed to upload")
                    for failed_file in results.get("failed_files", []):
                        _print(f"  Failed: {failed_file}")
            else:
                error = results.get("error", "Unknown error")
                _print(f"[bold red]✗ Screenshot upload failed: {error}")
                raise RuntimeError(f"Screenshot upload failed: {error}")
                
        except Exception as e:
            _print(f"[bold red]✗ Critical error during screenshot upload: {e}")
            _print(f"[bold red]Cannot proceed with batch processing without screenshots in GCS")
            raise RuntimeError(f"Screenshot upload to GCS failed: {e}")
    
    async def run(self):
        """
        Main entry point for simplified batch processing.
        
        This method orchestrates the entire batch processing workflow using services.
        """
        # Validate screenshots
        if not self.screenshot_service.validate_all_screenshots(self.experiments_df, self.local_dataset_path):
            raise RuntimeError("Screenshot validation failed. Cannot proceed with batch processing.")
        
        # Upload screenshots to GCS if remote mode is enabled
        await self._upload_screenshots_to_gcs()
        
        # Validate metadata against providers at startup
        await self._validate_metadata_at_startup()
        
        # Refresh existing batches and process any completed ones
        await self._refresh_existing_batches_at_startup()
        
        _print(f"[bold blue]Starting batch processing for {len(self.supported_engines)} engines...")
        _print(f"[bold blue]Output directory: {self.run_output_dir}")
        
        if self.force_submit:
            _print(f"[bold yellow]Force submit mode enabled")
        
        # Submit all batches
        batch_mapping = await self.batch_service.submit_all_batches(
            self.supported_engines, self.experiments_df, self.run_output_dir, self.force_submit
        )
        
        if not batch_mapping:
            _print("[bold yellow]No batches were submitted")
            return
        
        # Monitor and process results
        await self.batch_service.monitor_and_process_results(
            batch_mapping, self.run_output_dir, self.experiments_df
        )
        
        _print("[bold green]✓ Batch processing completed successfully!")