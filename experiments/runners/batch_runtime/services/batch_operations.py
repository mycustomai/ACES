"""
Service to orchestrate batch operations across all providers.

This service coordinates batch submission, monitoring, and result processing
across different AI providers (OpenAI, Anthropic, Gemini).
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import backoff
import pandas as pd
from rich import print as _print

from agent.src.core.tools import AddToCartInput
from agent.src.logger import create_logger
from agent.src.shopper import SimulatedShopper
from agent.src.typedefs import EngineParams
from experiments.config import ExperimentData
from experiments.data_loader import experiments_iter
from experiments.filesystem_environment import FilesystemShoppingEnvironment
from experiments.results import aggregate_model_data
from experiments.runners.batch_runtime.providers.base import BaseBatchProvider

from .experiment_tracking import ExperimentTrackingService
from .file_operations import FileOperationsService


class BatchOperationsService:
    """Service to orchestrate batch operations across all providers."""
    
    def __init__(self, providers: Dict[str, BaseBatchProvider], tracking_service: ExperimentTrackingService,
                 file_ops: FileOperationsService, screenshots_dir: Path,
                 dataset_name: str,
                 remote: bool = False,
                 engine_params_list: Optional[List[EngineParams]] = None):
        self.providers = providers
        self.tracking_service = tracking_service
        self.file_ops = file_ops
        self.screenshots_dir = screenshots_dir
        self.dataset_name = dataset_name
        self.remote = remote
        # Store engine params mapping for later use
        self.engine_params_map: Dict[str, EngineParams] = {}
        
        # Initialize engine params map if provided
        if engine_params_list:
            for engine_params in engine_params_list:
                self.engine_params_map[engine_params.config_name] = engine_params

    async def submit_all_batches(self, supported_engines: List[EngineParams], 
                               experiments_df, run_output_dir: Path, 
                               force_submit: bool = False) -> Dict[str, List[str]]:
        """Submit batches for all supported engines."""
        all_submitted_batches = {}
        
        # Store engine params for later use
        for engine_params in supported_engines:
            self.engine_params_map[engine_params.config_name] = engine_params
        
        # Submit batches for each engine in parallel
        submission_tasks = []
        for engine_params in supported_engines:
            task = asyncio.create_task(
                self._submit_batches_for_engine(engine_params, experiments_df, 
                                              run_output_dir, force_submit)
            )
            submission_tasks.append((engine_params.config_name, task))
        
        # Wait for all submissions to complete
        for config_name, task in submission_tasks:
            try:
                batch_ids = await task
                if batch_ids:
                    all_submitted_batches[config_name] = batch_ids
                    _print(f"[bold green]✓ Submitted {len(batch_ids)} batches for {config_name}")
                else:
                    _print(f"[bold yellow]No batches submitted for {config_name}")
            except Exception as e:
                _print(f"[bold red]Failed to submit batches for {config_name}: {e}")
        
        return all_submitted_batches
    
    async def monitor_and_process_results(self, batch_mapping: Dict[str, List[str]], 
                                        run_output_dir: Path, experiments_df: pd.DataFrame):
        """Monitor batches and process results as they complete."""
        if not batch_mapping:
            _print("[bold yellow]No batches to monitor")
            return
        
        _print(f"[bold blue]Monitoring {sum(len(batches) for batches in batch_mapping.values())} batches across {len(batch_mapping)} providers...")
        
        # Start monitoring tasks for each provider
        monitoring_tasks = []
        for config_name, batch_ids in batch_mapping.items():
            if batch_ids:
                provider = self.providers.get(config_name)
                if provider:
                    task = asyncio.create_task(
                        self._monitor_provider_batches(provider, batch_ids, config_name, 
                                                     run_output_dir, experiments_df)
                    )
                    monitoring_tasks.append(task)
        
        # Wait for all monitoring to complete
        if monitoring_tasks:
            await asyncio.gather(*monitoring_tasks, return_exceptions=True)
        
        _print("[bold green]✓ All batch monitoring completed")
    
    async def _submit_batches_for_engine(self, engine_params: EngineParams, experiments_df: pd.DataFrame,
                                       run_output_dir: Path, force_submit: bool) -> List[str]:
        """Submit batches for a single engine."""
        config_name = engine_params.config_name
        provider = self.providers.get(config_name)
        
        if not provider:
            _print(f"[bold red]No provider found for {config_name}")
            return []
        
        # Get outstanding experiments
        all_experiments = list(experiments_iter(experiments_df, self.dataset_name))
        outstanding_experiments = await self.tracking_service.get_outstanding_experiments(
            config_name, all_experiments, experiments_df, force_submit, engine_params
        )
        
        if not outstanding_experiments:
            _print(f"[bold green]All experiments already completed for {config_name}")
            return []
        
        # Generate batch requests
        batch_requests = []
        experiments_in_batch = []
        
        for data in outstanding_experiments:
            try:
                request = self._generate_batch_request(data, engine_params)
                batch_requests.append(request)
                experiments_in_batch.append(data.experiment_id)
            except Exception as e:
                _print(f"[bold red]Error generating request for {data.experiment_id}: {e}")
        
        if not batch_requests:
            return []
        
        # Submit to provider with retry logic
        _print(f"[bold blue]Submitting {len(batch_requests)} requests for {config_name}...")
        
        submission_context = {
            'experiments_to_run': outstanding_experiments,
            'engine_params': engine_params,
            'batch_runtime': self,
            'experiment_ids': experiments_in_batch
        }
        
        submitted_ids = await self._submit_with_retry(
            provider, batch_requests, config_name, run_output_dir, submission_context
        )
        
        # Update tracking only if submission was successful
        if submitted_ids and experiments_in_batch:
            # Validate submitted batches before updating metadata
            valid_batch_ids = await self._validate_submitted_batches(provider, submitted_ids, config_name)
            if valid_batch_ids:
                await self.tracking_service.mark_experiments_submitted(
                    config_name, experiments_in_batch, valid_batch_ids
                )
                _print(f"[bold green]✓ Updated metadata for {len(valid_batch_ids)} successfully submitted batches")
            else:
                _print(f"[bold red]No valid batch submissions found - metadata not updated")
                return []
        
        return submitted_ids
    
    @backoff.on_exception(
        backoff.expo,
        (Exception,),
        max_tries=3,
        max_time=300,  # 5 minutes max
        on_backoff=lambda details: _print(f"[bold yellow]Retrying batch submission (attempt {details['tries']}) after {details['wait']:.1f}s..."),
        on_giveup=lambda details: _print(f"[bold red]Failed to submit batch after {details['tries']} attempts")
    )
    async def _submit_with_retry(self, provider, batch_requests, config_name, run_output_dir, submission_context):
        """Submit batches with retry logic and exponential backoff."""
        try:
            return await provider.upload_and_submit_batches(
                batch_requests, config_name, run_output_dir, submission_context
            )
        except Exception as e:
            _print(f"[bold red]Batch submission error for {config_name}: {e}")
            raise
    
    @backoff.on_exception(
        backoff.expo,
        (Exception,),
        max_tries=3,
        max_time=180,  # 3 minutes max
        on_backoff=lambda details: _print(f"[bold yellow]Retrying batch download (attempt {details['tries']}) after {details['wait']:.1f}s..."),
        on_giveup=lambda details: _print(f"[bold red]Failed to download batch results after {details['tries']} attempts")
    )
    async def _download_with_retry(self, provider, batch_id, results_file):
        """Download batch results with retry logic and exponential backoff."""
        try:
            return await provider.download_batch_results(batch_id, results_file)
        except Exception as e:
            _print(f"[bold red]Batch download error for {batch_id}: {e}")
            raise
    
    @backoff.on_exception(
        backoff.expo,
        (Exception,),
        max_tries=3,
        max_time=60,  # 1 minute max for monitoring
        on_backoff=lambda details: _print(f"[bold yellow]Retrying batch monitoring (attempt {details['tries']}) after {details['wait']:.1f}s..."),
        on_giveup=lambda details: _print(f"[bold red]Failed to monitor batches after {details['tries']} attempts")
    )
    async def _monitor_with_retry(self, provider, batch_ids):
        """Monitor batch statuses with retry logic and exponential backoff."""
        try:
            return await provider.monitor_batches(batch_ids)
        except Exception as e:
            _print(f"[bold red]Batch monitoring error: {e}")
            raise
    
    async def _validate_submitted_batches(self, provider: BaseBatchProvider, batch_ids: List[str], config_name: str) -> List[str]:
        """
        Validate that submitted batches actually exist on the provider side.
        
        This method queries the provider to confirm that the submitted batch IDs
        are valid and exist, focusing on the Gemini provider as requested.
        
        Args:
            provider: The batch provider instance
            batch_ids: List of batch IDs that were supposedly submitted
            config_name: Configuration name for logging
            
        Returns:
            List of validated batch IDs that actually exist on the provider
        """
        if not batch_ids:
            return []
        
        _print(f"[bold blue]Validating {len(batch_ids)} submitted batches for {config_name}...")
        valid_batch_ids = []
        
        try:
            # For Gemini provider, we can check if batches exist by querying their status
            if hasattr(provider, '_get_batch_status'):
                for batch_id in batch_ids:
                    try:
                        status = await provider._get_batch_status(batch_id)
                        if status and status != 'error':
                            valid_batch_ids.append(batch_id)
                            _print(f"[bold green]✓ Validated batch {batch_id} (status: {status})")
                        else:
                            _print(f"[bold red]✗ Invalid batch {batch_id} (status: {status})")
                    except Exception as e:
                        _print(f"[bold red]✗ Failed to validate batch {batch_id}: {e}")
            
            # For other providers, try to use the monitor_batches method
            elif hasattr(provider, 'monitor_batches'):
                try:
                    status_map = await provider.monitor_batches(batch_ids)
                    for batch_id, status in status_map.items():
                        if status and status != 'error':
                            valid_batch_ids.append(batch_id)
                            _print(f"[bold green]✓ Validated batch {batch_id} (status: {status})")
                        else:
                            _print(f"[bold red]✗ Invalid batch {batch_id} (status: {status})")
                except Exception as e:
                    _print(f"[bold red]Error validating batches via monitor_batches: {e}")
                    # Fallback: assume all batches are valid if we can't validate
                    valid_batch_ids = batch_ids
            else:
                # Fallback: assume all batches are valid if provider doesn't support validation
                _print(f"[bold yellow]Provider {config_name} doesn't support batch validation - assuming all valid")
                valid_batch_ids = batch_ids
            
            validation_rate = len(valid_batch_ids) / len(batch_ids) * 100 if batch_ids else 0
            _print(f"[bold cyan]Batch validation complete: {len(valid_batch_ids)}/{len(batch_ids)} valid ({validation_rate:.1f}%)")
            
        except Exception as e:
            _print(f"[bold red]Error during batch validation for {config_name}: {e}")
            # In case of validation error, don't assume batches are valid
            valid_batch_ids = []
        
        return valid_batch_ids
    
    async def validate_and_rectify_metadata(self, config_name: str, auto_cleanup: bool = False) -> Dict[str, Any]:
        """
        Public method to validate metadata against provider and optionally clean up inconsistencies.
        
        This method provides a comprehensive validation of local batch metadata against
        the actual provider state, with optional automatic cleanup of invalid entries.
        
        Args:
            config_name: Configuration name to validate
            auto_cleanup: If True, automatically clean up invalid metadata entries
            
        Returns:
            Dictionary containing validation results and actions taken
        """
        provider = self.providers.get(config_name)
        if not provider:
            return {
                'error': f'No provider found for configuration: {config_name}',
                'config_name': config_name
            }
        
        _print(f"[bold blue]Starting metadata validation and rectification for {config_name}...")
        
        try:
            # Perform validation
            validation_report = await self.tracking_service.validate_metadata_against_provider(
                config_name, provider
            )
            
            # Optionally perform cleanup
            if auto_cleanup and validation_report.get('invalid_batches'):
                _print(f"[bold yellow]Auto-cleanup enabled - removing invalid metadata...")
                cleanup_success = await self.tracking_service.cleanup_invalid_metadata(
                    config_name, validation_report
                )
                validation_report['cleanup_performed'] = cleanup_success
                validation_report['cleanup_timestamp'] = asyncio.get_event_loop().time()
            else:
                validation_report['cleanup_performed'] = False
            
            # Add summary
            validation_report['summary'] = {
                'total_batches': validation_report['batches_in_metadata'],
                'valid_batches': len(validation_report['valid_batches']),
                'invalid_batches': len(validation_report['invalid_batches']),
                'orphaned_experiments': len(validation_report['orphaned_experiments']),
                'validation_success_rate': (
                    len(validation_report['valid_batches']) / validation_report['batches_in_metadata'] * 100
                    if validation_report['batches_in_metadata'] > 0 else 100
                )
            }
            
            _print(f"[bold green]✓ Metadata validation completed for {config_name}")
            _print(f"  Success rate: {validation_report['summary']['validation_success_rate']:.1f}%")
            
            return validation_report
            
        except Exception as e:
            error_report = {
                'error': f'Validation failed: {str(e)}',
                'config_name': config_name,
                'validation_timestamp': asyncio.get_event_loop().time()
            }
            _print(f"[bold red]Metadata validation failed for {config_name}: {e}")
            return error_report
    
    async def _monitor_provider_batches(self, provider: BaseBatchProvider, batch_ids: List[str], config_name: str,
                                      run_output_dir: Path, experiments_df: pd.DataFrame):
        """Monitor batches for a specific provider."""
        _print(f"[bold blue]Monitoring {len(batch_ids)} batches for {config_name}...")
        
        completed_batches = set()
        check_interval = 5  # Reduced from 10 to 5 seconds for faster monitoring
        
        while True:
            try:
                # Check batch statuses with retry
                status_map = await self._monitor_with_retry(provider, batch_ids)
                
                if not status_map:
                    _print(f"[bold yellow]No status received for {config_name} batches")
                    await asyncio.sleep(check_interval)
                    continue
                
                # Process newly completed batches
                newly_completed = []
                for batch_id, status in status_map.items():
                    if status in ['completed', 'ended'] and batch_id not in completed_batches:
                        completed_batches.add(batch_id)
                        newly_completed.append(batch_id)
                
                # Download and process results for completed batches
                if newly_completed:
                    for batch_id in newly_completed:
                        await self._process_completed_batch(provider, batch_id, config_name, 
                                                          run_output_dir, experiments_df)
                
                # Check if all batches are done
                completed_count = sum(1 for status in status_map.values() 
                                    if status in ['completed', 'ended'])
                failed_count = sum(1 for status in status_map.values() 
                                 if status in ['failed', 'error', 'cancelled'])
                in_progress_count = len(status_map) - completed_count - failed_count
                
                _print(f"[bold cyan]{config_name} status: {completed_count} completed, {in_progress_count} in progress, {failed_count} failed")
                
                if in_progress_count == 0:
                    _print(f"[bold green]All batches completed for {config_name}")
                    # Run final aggregation
                    await self._run_final_aggregation_for_provider(config_name, run_output_dir, experiments_df)
                    break
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                _print(f"[bold red]Error monitoring {config_name}: {e}")
                await asyncio.sleep(check_interval)
    
    async def _process_completed_batch(self, provider: BaseBatchProvider, batch_id: str, config_name: str,
                                     run_output_dir: Path, experiments_df) -> bool:
        """Process a completed batch by downloading and processing results.
        
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            _print(f"[bold cyan]Processing completed batch {batch_id} for {config_name}...")
            
            # Download results with retry
            results_file = self.file_ops.get_batch_results_file(config_name, batch_id)
            downloaded_path = await self._download_with_retry(provider, batch_id, results_file)
            
            if downloaded_path:
                # Process the results file
                await self._process_batch_results_file(downloaded_path, config_name, 
                                                     provider, run_output_dir, experiments_df)
                _print(f"[bold green]✓ Processed batch {batch_id} for {config_name}")
                return True
            else:
                _print(f"[bold red]Failed to download results for batch {batch_id}")
                return False
                
        except Exception as e:
            _print(f"[bold red]Error processing batch {batch_id}: {e}")
            return False
    
    async def _process_batch_results_file(self, results_file_path: str, config_name: str,
                                        provider, run_output_dir: Path, experiments_df):
        """Process a batch results file and create experiment outputs."""
        results = self.file_ops.read_jsonl_file(Path(results_file_path))
        
        if not results:
            _print(f"[bold yellow]No results found in {results_file_path}")
            return
        
        _print(f"[bold blue]Processing {len(results)} results from {results_file_path}")
        
        # Process each result
        processed_count = 0
        for i, result in enumerate(results):
            try:
                await self._process_single_result(result, provider, config_name, 
                                                run_output_dir, experiments_df)
                processed_count += 1
            except Exception as e:
                _print(f"[bold red]Error processing single result {i+1}: {e}")
                import traceback
                traceback.print_exc()
        
        _print(f"[bold green]Successfully processed {processed_count}/{len(results)} results")
    
    async def _process_single_result(self, result: Dict[str, Any], provider: BaseBatchProvider, config_name: str,
                                   run_output_dir: Path, experiments_df):
        """Process a single batch result and create experiment output."""
        # Get custom_id and parse experiment info
        custom_id = result.get('custom_id')
        if not custom_id:
            return
        
        # For Anthropic providers, resolve hashed custom_id back to original
        original_custom_id = custom_id
        if hasattr(provider, 'resolve_custom_id'):
            resolved_id = provider.resolve_custom_id(custom_id, config_name)
            if resolved_id:
                original_custom_id = resolved_id
            else:
                _print(f"[bold yellow]Could not resolve custom_id {custom_id} for Anthropic provider")
                return
        
        # Parse custom_id to get experiment info
        if '|' not in original_custom_id:
            _print(f"[bold yellow]Invalid custom_id format: {original_custom_id}")
            return
        
        parts = original_custom_id.split('|')
        if len(parts) < 4:
            _print(f"[bold yellow]Insufficient custom_id parts: {original_custom_id}")
            return
        
        query, experiment_label, experiment_number, _ = parts[:4]
        
        # Find corresponding experiment data
        experiment_data = None
        for data in experiments_iter(experiments_df, self.dataset_name):
            if (data.query == query and 
                data.experiment_label == experiment_label and 
                str(data.experiment_number) == experiment_number):
                experiment_data = data
                break
        
        if not experiment_data:
            _print(f"[bold yellow]Could not find experiment data for original_custom_id: {original_custom_id}")
            _print(f"[bold yellow]  Parsed: query='{query}', experiment_label='{experiment_label}', experiment_number='{experiment_number}'")
            return
        
        # Check if response was successful
        if not provider.is_response_successful(result):
            error_msg = provider.get_error_message(result)
            _print(f"[bold red]Batch request failed for {custom_id}: {error_msg}")
            return
        
        # Extract response and tool calls
        response_body = provider.get_response_body_from_result(result)
        tool_calls = provider.parse_tool_calls_from_response(response_body)
        
        # Find add_to_cart tool call
        add_to_cart_call = None
        for tool_call in tool_calls:
            tool_name = tool_call.get('function', {}).get('name')
            if tool_name == 'add_to_cart':
                add_to_cart_call = tool_call
                break
        
        if not add_to_cart_call:
            _print(f"[bold yellow]No add_to_cart tool call found for {original_custom_id}")
            _print(f"[bold yellow]Available tool calls: {[tc.get('function', {}).get('name') for tc in tool_calls]}")
            return
        
        # Parse tool call arguments
        try:
            args_str = add_to_cart_call['function'].get('arguments', '{}')
            args_dict = json.loads(args_str) if isinstance(args_str, str) else args_str
            add_to_cart_input = AddToCartInput.model_validate(args_dict)
        except Exception as e:
            _print(f"[bold red]Failed to parse add_to_cart arguments for {custom_id}: {e}")
            return
        
        # Create experiment output
        await self._create_experiment_output(experiment_data, config_name, add_to_cart_input, 
                                           response_body, provider, run_output_dir)
    
    async def _create_experiment_output(self, data: ExperimentData, config_name: str,
                                      add_to_cart_input: AddToCartInput, response_body: Dict[str, Any],
                                      provider: BaseBatchProvider, run_output_dir: Path):
        """Create experiment output directory and files."""
        try:
            # Get engine params from stored mapping
            engine_params = self.engine_params_map.get(config_name)
            
            if not engine_params:
                _print(f"[bold yellow]Could not find engine params for {config_name}")
                _print(f"[bold yellow]Available engine params: {list(self.engine_params_map.keys())}")
                return
            
            experiment_df = data.experiment_df
            model_output_dir = data.model_output_dir(run_output_dir, engine_params)
            
            # Create logger and record experiment results
            
            with create_logger(
                data.query,
                output_dir=model_output_dir,
                experiment_df=experiment_df,
                engine_params=engine_params,
                experiment_label=data.experiment_label,
                experiment_number=data.experiment_number,
                silent=True
            ) as logger:
                # Record the cart item
                logger.record_cart_item(add_to_cart_input)
                
                # Create mock AIMessage for the response
                from langchain_core.messages import AIMessage
                tool_call_dict = {
                    'name': 'add_to_cart',
                    'args': add_to_cart_input.model_dump(),
                    'id': 'batch_result'
                }
                
                response_content = provider.extract_response_content(response_body)
                
                ai_message = AIMessage(
                    content=response_content,
                    tool_calls=[tool_call_dict]
                )
                
                # Record the agent interaction
                logger.record_agent_interaction(ai_message)
                
        except Exception as e:
            _print(f"[bold red]Error creating experiment output for {data.experiment_id}: {e}")
    
    async def _run_final_aggregation_for_provider(self, config_name: str, run_output_dir: Path, experiments_df):
        """Run final aggregation for a provider."""
        try:
            _print(f"[bold blue]Running final aggregation for {config_name}...")
            
            # Get engine params for this config
            engine_params = self.engine_params_map.get(config_name)
            if not engine_params:
                _print(f"[bold yellow]Could not find engine params for {config_name}")
                return
            
            # Get the model output directory using the proper method
            model_output_dir = ExperimentData.model_output_dir(run_output_dir, engine_params)
            
            # Check if the directory exists and has data
            if model_output_dir.exists():
                try:
                    aggregate_model_data(model_output_dir)
                    _print(f"[bold green]✓ Aggregated data for {model_output_dir}")
                except Exception as e:
                    _print(f"[bold red]Error aggregating {model_output_dir}: {e}")
            else:
                _print(f"[bold yellow]Model output directory does not exist: {model_output_dir}")
            
            _print(f"[bold green]✓ Final aggregation completed for {config_name}")
            
        except Exception as e:
            _print(f"[bold red]Error during final aggregation for {config_name}: {e}")
    
    def _generate_batch_request(self, data: ExperimentData, engine_params: EngineParams) -> dict:
        """Generate a batch request for the given experiment."""
        # remote=True for Gemini always, or if batch runtime remote flag is set
        is_gemini = engine_params.engine_type.lower() == "gemini"
        use_remote = is_gemini or self.remote
        
        environment = FilesystemShoppingEnvironment(
            screenshots_dir=self.screenshots_dir,
            query=data.query,
            experiment_label=data.experiment_label,
            experiment_number=data.experiment_number,
            dataset_name=self.dataset_name,
            remote=use_remote
        )
        
        shopper = SimulatedShopper(
            initial_message=data.prompt_template,
            engine_params=engine_params,
            environment=environment,
            logger=None,
        )
        
        raw_messages = shopper.get_batch_request()
        
        tools = []
        for tool in shopper.agent.tools:
            if hasattr(tool, 'args_schema') and tool.args_schema:
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.args_schema.model_json_schema()
                    }
                }
                tools.append(tool_def)
        
        # Use provider to create request
        provider = self.providers.get(engine_params.config_name)
        if provider:
            custom_id = f"{data.query}|{data.experiment_label}|{data.experiment_number}|{engine_params.config_name}"
            return provider.create_batch_request(data, engine_params, raw_messages, custom_id, tools)
        else:
            raise ValueError(f"No provider found for {engine_params.config_name}")
