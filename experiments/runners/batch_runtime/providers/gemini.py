"""
Gemini batch provider implementation using enhanced base classes.

Simplified by delegating common operations to base classes while handling
Gemini-specific requirements like GCS integration and tool schema conversion.
"""

import asyncio
import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from rich import print as _print

try:
    import vertexai
    from google import genai
    from google.cloud import storage
    from google.genai.types import (CreateBatchJobConfig, HttpOptions,
                                    JobState, ListBatchJobsConfig)
    GENAI_AVAILABLE = True
    GCS_AVAILABLE = True
except ImportError as e:
    GENAI_AVAILABLE = False
    GCS_AVAILABLE = False
    _print(f"[bold yellow]Warning: Google Cloud libraries not available: {e}")
    _print("[bold yellow]Install with: pip install google-cloud-storage google-generativeai")

from agent.src.typedefs import EngineParams
from experiments.config import ExperimentData
from experiments.runners.services.gcs_upload import GCSUploadService

from ..services.file_operations import FileOperationsService
from .base import BaseBatchProvider, BatchProviderConfig, ChunkingStrategy


class GeminiBatchProvider(BaseBatchProvider):
    """Gemini-specific batch API implementation using enhanced base classes."""

    def __init__(self, file_ops: FileOperationsService, screenshots_dir: Optional[Path] = None,
                 project_id: Optional[str] = None, location: str = "us-central1",
                 bucket_name: Optional[str] = None, dataset_name: Optional[str] = None,
                 skip_upload: bool = False):
        """
        Initialize GeminiBatchProvider.

        Args:
            file_ops: File operations service
            screenshots_dir: Directory containing screenshots for upload
            project_id: Google Cloud project ID
            location: Google Cloud location
            bucket_name: GCS bucket name for screenshots
            dataset_name: Name of the dataset for unique file naming
            skip_upload: Skip automatic screenshot upload (when batch runtime handles it)
        """
        # Create configuration
        config = BatchProviderConfig(
            api_key=os.getenv('GOOGLE_API_KEY'),
            project_id=project_id or os.getenv('GOOGLE_CLOUD_PROJECT'),
            location=location,
            bucket_name=bucket_name or os.getenv('GCS_BUCKET_NAME'),
            dataset_name=dataset_name
        )

        # Initialize base class
        super().__init__(config, file_ops)

        self.screenshots_dir = Path(screenshots_dir) if screenshots_dir else None
        self.skip_upload = skip_upload
        
        # Validate Google Cloud configuration
        errors = self._validate_google_cloud_config()
        if errors:
            _print(f"[bold yellow]Google Cloud configuration warnings: {', '.join(errors)}")

        # Initialize clients lazily
        self._batch_client = None
        self._storage_client = None
        self._uploaded_screenshots: Dict[str, str] = {}  # local_path -> gcs_url
        self._batch_output_mappings: Dict[str, str] = {}  # batch_id -> output_uri
        self._bulk_upload_completed = False
        
        # Initialize GCS upload service (only if not skipping upload)
        self._gcs_upload_service = None
        if self.config.bucket_name and not self.skip_upload:
            try:
                self._gcs_upload_service = GCSUploadService(
                    bucket_name=self.config.bucket_name,
                    project_id=self.config.project_id
                )
                _print(f"[bold green]✓ GCS upload service initialized")
            except Exception as e:
                _print(f"[bold yellow]Warning: Could not initialize GCS upload service: {e}")
        elif self.skip_upload:
            _print(f"[bold blue]Skipping GCS upload service initialization (upload handled by batch runtime)")

    def create_batch_request(self, data: ExperimentData, engine_params: EngineParams,
                           raw_messages: List[Dict[str, Any]], custom_id: str,
                           tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create Gemini batch request object."""
        # Convert raw_messages to Gemini GenerateContentRequest format
        contents = []
        system_instruction = None

        for message in raw_messages:
            # Handle system messages - Gemini uses systemInstruction instead of system role
            if message.get('role') == 'system':
                if isinstance(message.get('content'), str):
                    system_instruction = message['content']
                elif isinstance(message.get('content'), list):
                    # Extract text from system message parts
                    text_parts = []
                    for part in message['content']:
                        if part.get('type') == 'text':
                            text_parts.append(part['text'])
                    if text_parts:
                        system_instruction = '\n'.join(text_parts)
                continue  # Skip adding system messages to contents
            
            # Handle different message formats and convert to Gemini contents structure
            if isinstance(message.get('content'), str):
                # Simple text message
                contents.append({
                    "role": 'model' if message.get('role') == 'assistant' else message.get('role', 'user'),
                    "parts": [{"text": message['content']}]
                })
            elif isinstance(message.get('content'), list):
                # Complex message with multiple parts (text + images)
                parts = []
                for part in message['content']:
                    if part.get('type') == 'text':
                        parts.append({"text": part['text']})
                    elif part.get('type') == 'image_url':
                        image_url = part['image_url']['url']
                        
                        # Handle GCS URLs only - Gemini batch API doesn't support base64 images
                        if image_url.startswith('https://storage.googleapis.com/'):
                            # Convert GCS HTTPS URL to GCS URI format
                            gcs_uri = image_url.replace('https://storage.googleapis.com/', 'gs://')
                            parts.append({
                                "file_data": {
                                    "mime_type": "image/png", 
                                    "file_uri": gcs_uri
                                }
                            })
                        elif image_url.startswith('gs://'):
                            # GCS URI format - use directly
                            parts.append({
                                "file_data": {
                                    "mime_type": "image/png",
                                    "file_uri": image_url
                                }
                            })
                        else:
                            # For batch processing, we should only receive GCS URLs
                            if image_url.startswith('data:image/'):
                                _print(f"[bold red]Error: Gemini batch API doesn't support base64 images")
                                _print(f"[bold yellow]Hint: Ensure FilesystemShoppingEnvironment uses remote=True for Gemini")
                            else:
                                _print(f"[bold red]Error: Unexpected image format for Gemini batch API: {image_url[:100]}...")
                            continue  # Skip this image
                
                if parts:  # Only add if we have valid parts
                    contents.append({
                        "role": 'model' if message.get('role') == 'assistant' else message.get('role', 'user'),
                        "parts": parts
                    })
        
        # Create GenerateContentRequest structure
        request: Dict[str, str | list | dict] = {
            "contents": contents,
            "model": engine_params.model,
            "custom_id": custom_id
        }
        
        # Add system instruction if present
        if system_instruction:
            request["system_instruction"] = {
                "parts": [{"text": system_instruction}]
            }
        
        # Add generation config
        request["generationConfig"] = {
            "maxOutputTokens": getattr(engine_params, 'max_tokens', getattr(engine_params, 'max_new_tokens', 1000)),
            "temperature": getattr(engine_params, 'temperature', 0.7)
        }
        
        # Add tools if provided
        if tools:
            # Convert tools to Gemini function calling format
            gemini_tools = []
            for tool in tools:
                if tool.get('type') == 'function' and 'function' in tool:
                    gemini_tools.append({
                        "function_declarations": [{
                            "name": tool['function']['name'],
                            "description": tool['function']['description'],
                            "parameters": tool['function']['parameters']
                        }]
                    })
            
            if gemini_tools:
                request["tools"] = gemini_tools
        
        return request

    def get_request_chunks(self, requests: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Gemini uses count-based chunking with 100 requests per batch for better manageability."""
        return self.chunk_requests(requests, ChunkingStrategy.BY_COUNT, chunk_size=100)

    async def _submit_chunk(self, chunk: List[Dict[str, Any]], config_name: str, 
                           chunk_index: int, submission_context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Submit a single chunk to Gemini API."""
        try:
            # Ensure batch client is initialized
            if not await self._ensure_batch_client():
                return None

            # Generate unique chunk name
            chunk_name = self.generate_unique_batch_name(config_name, chunk_index)

            _print(f"[bold blue]Submitting Gemini batch '{chunk_name}' with {len(chunk)} requests...")

            # Write chunk to input file
            input_file = self.write_batch_input_file(chunk, config_name, f"chunk_{chunk_index}")

            # Submit to API
            response = await self._submit_batch_to_gemini_api(input_file, chunk_name)
            
            if response and 'name' in response:
                batch_id = response['name']
                _print(f"[bold green]✓ Gemini batch '{chunk_name}' submitted successfully. Batch ID: {batch_id}")
                return batch_id
            else:
                _print(f"[bold red]✗ Failed to submit Gemini batch '{chunk_name}': Invalid response")
                return None

        except Exception as e:
            self.handle_api_error(f"Gemini batch submission", e, 
                                {"chunk_index": chunk_index, "config": config_name})
            return None

    async def monitor_batches(self, batch_ids: List[str]) -> Dict[str, str]:
        """Monitor Gemini batch processing status using efficient bulk monitoring."""
        if not batch_ids:
            return {}
            
        # Use bulk monitoring for better performance
        try:
            return await self._monitor_batches_bulk(batch_ids)
        except Exception as e:
            _print(f"[bold yellow]Bulk monitoring failed, falling back to individual checks: {e}")
            # Fallback to individual monitoring if bulk fails
            return await self._monitor_batches_individual(batch_ids)
    
    async def _monitor_batches_individual(self, batch_ids: List[str]) -> Dict[str, str]:
        """Monitor batches individually (fallback method)."""
        if not await self._ensure_batch_client():
            return {batch_id: 'error' for batch_id in batch_ids}

        status_map = {}

        for batch_id in batch_ids:
            try:
                # Get batch status from API
                status = await self._get_batch_status(batch_id)
                status_map[batch_id] = status
            except Exception as e:
                self.handle_api_error(f"Gemini status check", e, {"batch_id": batch_id})
                status_map[batch_id] = 'error'

        return status_map

    async def _monitor_batches_bulk(self, batch_ids: List[str]) -> Dict[str, str]:
        """
        Monitor multiple batches efficiently using list_batches API.
        
        This method leverages Gemini's bulk listing capability to check status
        of multiple batches in fewer API calls, improving performance for large datasets.
        
        Args:
            batch_ids: List of batch IDs to monitor
            
        Returns:
            Dict mapping batch_id to status
        """
        if not batch_ids:
            return {}
            
        if not await self._ensure_batch_client():
            return {batch_id: 'error' for batch_id in batch_ids}
        
        # Convert batch_ids to a set for faster lookup
        target_batch_ids = set(batch_ids)
        status_map = {}
        
        try:
            _print(f"[bold blue]Using bulk monitoring for {len(batch_ids)} Gemini batches...")
            
            # Use list_batches to get all batches, then filter for our target IDs
            # This is more efficient than individual API calls for each batch
            # Use larger page size and limit to only what we need for maximum efficiency
            max_needed = min(len(batch_ids) * 3, 2000)  # Get a bit more than needed, but cap at 2000
            all_batches = await self.list_batches(page_size=2000, max_batches=max_needed)
            
            for batch_info in all_batches:
                batch_name = batch_info['name']
                if batch_name in target_batch_ids:
                    # Map JobState to consistent status strings
                    state = batch_info['state']
                    if state == JobState.JOB_STATE_SUCCEEDED:
                        status_map[batch_name] = 'completed'
                    elif state == JobState.JOB_STATE_FAILED:
                        status_map[batch_name] = 'failed'
                    elif state == JobState.JOB_STATE_CANCELLED:
                        status_map[batch_name] = 'cancelled'
                    elif state in [JobState.JOB_STATE_RUNNING, JobState.JOB_STATE_PENDING]:
                        status_map[batch_name] = 'in_progress'
                    elif state == JobState.JOB_STATE_PAUSED:
                        status_map[batch_name] = 'paused'
                    else:
                        status_map[batch_name] = str(state).lower()
                    
                    # Early exit optimization: if we found all target batches, stop processing
                    if len(status_map) == len(target_batch_ids):
                        break
            
            # Handle any batch IDs that weren't found in the list
            for batch_id in target_batch_ids:
                if batch_id not in status_map:
                    status_map[batch_id] = 'not_found'
            
            _print(f"[bold green]✓ Bulk monitoring completed: found {len(status_map)} of {len(batch_ids)} batches")
                
        except Exception as e:
            self.handle_api_error("Bulk batch monitoring", e, {"batch_count": len(batch_ids)})
            _print(f"[bold yellow]Bulk monitoring failed, falling back to individual checks...")
            # Fall back to individual monitoring if bulk fails
            return await self.monitor_batches(batch_ids)
        
        return status_map

    async def download_batch_results(self, batch_id: str, output_path: Path) -> Optional[str]:
        """Download batch results for a completed Gemini batch."""
        if not await self._ensure_batch_client():
            return None

        try:
            _print(f"[bold blue]Downloading Gemini results for batch {batch_id}...")

            # Get batch results from API
            results = await self._download_batch_results_from_api(batch_id)
            
            if not results:
                _print(f"[bold red]No results found for Gemini batch {batch_id}")
                return None

            # Write results to file in JSONL format
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.file_ops.write_jsonl_file(results, output_path)
            
            _print(f"[bold green]✓ Downloaded Gemini results to {output_path}")
            return str(output_path)

        except Exception as e:
            self.handle_api_error(f"Gemini batch download", e, {"batch_id": batch_id})
            return None

    def parse_tool_calls_from_response(self, response_body: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse tool calls from Gemini batch response."""
        tool_calls = []

        # Gemini format: candidates[0].content.parts contain functionCall blocks
        candidates = response_body.get('candidates', [])
        if candidates:
            content = candidates[0].get('content', {})
            parts = content.get('parts', [])
            
            for part in parts:
                if 'functionCall' in part:
                    func_call = part['functionCall']
                    tool_calls.append({
                        'function': {
                            'name': func_call.get('name'),
                            'arguments': json.dumps(func_call.get('args', {}))
                        }
                    })

        return tool_calls

    def extract_response_content(self, response_body: Dict[str, Any]) -> str:
        """Extract response content from Gemini response."""
        try:
            candidates = response_body.get('candidates', [])
            
            for candidate in candidates:
                content = candidate.get('content', {})
                for part in content.get('parts', []):
                    if 'text' in part:
                        return part['text']
        except (KeyError, TypeError) as e:
            _print(f"[bold red]Error extracting response content from Gemini response: {e}")
        return ""

    def is_response_successful(self, result: Dict[str, Any]) -> bool:
        """Check if Gemini batch response is successful."""
        # Check for errors in the result
        if 'error' in result:
            return False
        
        # Check if response has valid candidates
        response = result.get('response', result)
        candidates = response.get('candidates', [])
        
        return len(candidates) > 0

    def get_error_message(self, result: Dict[str, Any]) -> str:
        """Extract error message from Gemini batch response."""
        error = result.get('error', {})
        if isinstance(error, dict):
            return error.get('message', 'Unknown error')
        return str(error)

    def get_response_body_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the response body from a single batch result item."""
        # For Gemini batch responses, the response is under the 'response' key
        return result.get('response', {})

    # Screenshot management methods
    
    def upload_screenshots_for_experiments(self, experiments_df: pd.DataFrame, 
                                          max_workers: int = 6, skip_existing: bool = True) -> Dict[str, Any]:
        """
        Upload screenshots for experiments to GCS using the dedicated upload service.
        
        Args:
            experiments_df: DataFrame containing experiment information
            max_workers: Number of concurrent upload threads
            skip_existing: Skip files that already exist in GCS
            
        Returns:
            Dictionary with upload results and statistics
        """
        if self.skip_upload:
            _print(f"[bold blue]Skipping screenshot upload (handled by batch runtime)")
            return {"success": True, "uploaded": 0, "failed": 0, "skipped": 0, "note": "Upload skipped"}
        
        if not self._gcs_upload_service:
            return {"success": False, "error": "GCS upload service not initialized"}
        
        if not self.screenshots_dir:
            return {"success": False, "error": "Screenshots directory not specified"}
        
        if not self.config.dataset_name:
            return {"success": False, "error": "Dataset name not specified for GCS upload"}
        
        _print(f"[bold blue]Uploading screenshots for Gemini batch processing...")
        
        results = self._gcs_upload_service.upload_screenshots_batch(
            screenshots_dir=self.screenshots_dir,
            dataset_name=self.config.dataset_name,
            experiments_df=experiments_df,
            max_workers=max_workers,
            skip_existing=skip_existing
        )
        
        if results.get("success"):
            self._bulk_upload_completed = True
            _print(f"[bold green]✓ Screenshot upload completed for Gemini processing")
        
        return results
    
    def get_gcs_screenshot_url(self, query: str, experiment_label: str, experiment_number: int) -> Optional[str]:
        """
        Get the GCS URL for a specific screenshot.
        
        Args:
            query: Search query
            experiment_label: Experiment label
            experiment_number: Experiment number
            
        Returns:
            GCS URL for the screenshot, or None if not available
        """
        if not self._gcs_upload_service or not self.config.dataset_name:
            return None
        
        # Create a temporary ExperimentData object to use standardized path methods
        import pandas as pd

        from experiments.config import ExperimentData
        
        temp_experiment_data = ExperimentData(
            experiment_label=experiment_label,
            experiment_number=experiment_number,
            experiment_df=pd.DataFrame(),  # Empty DataFrame, not needed for path construction
            query=query,
            dataset_name=self.config.dataset_name
        )
        
        # Use ExperimentData method for standardized GCS path
        gcs_path = temp_experiment_data.get_gcs_screenshot_path()
        
        # Check if the file exists in GCS
        if self._gcs_upload_service.check_screenshot_exists_in_gcs(gcs_path):
            return self._gcs_upload_service.get_gcs_url(gcs_path)
        
        return None

    # Private helper methods

    def _validate_google_cloud_config(self) -> List[str]:
        """Validate Google Cloud configuration."""
        errors = []
        
        if not self.config.api_key:
            errors.append("GOOGLE_API_KEY not set")
        if not self.config.project_id:
            errors.append("GOOGLE_CLOUD_PROJECT not set")
        if not self.config.bucket_name:
            errors.append("GCS_BUCKET_NAME not set")
            
        return errors

    async def _ensure_batch_client(self) -> bool:
        """Ensure batch client is initialized."""
        if self._batch_client is None:
            if not GENAI_AVAILABLE:
                _print("[bold red]Google AI SDK not available. Please install google-generativeai.")
                return False
            
            try:
                # Set environment variables for genai client to use Vertex AI
                os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'True'
                if 'GOOGLE_CLOUD_LOCATION' not in os.environ:
                    os.environ['GOOGLE_CLOUD_LOCATION'] = self.config.location
                
                vertexai.init(project=self.config.project_id, location=self.config.location)
                self._batch_client = genai.Client(http_options=HttpOptions(api_version="v1"))
                _print(f"[bold green]✓ Vertex AI and GenAI clients initialized for project '{self.config.project_id}' in '{self.config.location}'")
            except Exception as e:
                self.handle_api_error("Gemini client initialization", e)
                return False
        return True

    async def _ensure_storage_client(self) -> bool:
        """Ensure GCS storage client is initialized."""
        if self._storage_client is None:
            if not GCS_AVAILABLE:
                _print("[bold red]Google Cloud Storage not available. Please install google-cloud-storage.")
                return False
            
            try:
                self._storage_client = storage.Client(project=self.config.project_id)
                self._bucket = self._storage_client.bucket(self.config.bucket_name)
                _print(f"[bold green]✓ GCS client initialized for bucket: {self.config.bucket_name}")
            except Exception as e:
                self.handle_api_error("GCS client initialization", e)
                return False
        return True

    def _convert_messages_to_gemini_format(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI message format to Gemini format."""
        gemini_messages = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            # Map roles
            if role == 'assistant':
                gemini_role = 'model'
            else:
                gemini_role = 'user'
            
            gemini_message = {
                "role": gemini_role,
                "parts": []
            }
            
            # Handle content (text or multimodal)
            if isinstance(content, str):
                gemini_message["parts"].append({"text": content})
            elif isinstance(content, list):
                for part in content:
                    if part.get('type') == 'text':
                        gemini_message["parts"].append({"text": part.get('text', '')})
                    elif part.get('type') == 'image_url':
                        # Handle image content - would need to convert to Gemini format
                        image_url = part.get('image_url', {}).get('url', '')
                        if image_url:
                            gemini_message["parts"].append({
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": image_url  # This would need proper conversion
                                }
                            })
            
            gemini_messages.append(gemini_message)
        
        return gemini_messages

    def _convert_openai_tools_to_gemini(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool schema format to Gemini tool format."""
        gemini_tools = []
        
        for tool in tools:
            if tool.get('type') != 'function':
                continue
            
            function_def = tool.get('function', {})
            gemini_tool = {
                "function_declarations": [{
                    "name": function_def.get('name'),
                    "description": function_def.get('description'),
                    "parameters": function_def.get('parameters', {})
                }]
            }
            
            gemini_tools.append(gemini_tool)
        
        return gemini_tools

    async def _submit_batch_to_gemini_api(self, input_file: Path, chunk_name: str) -> Optional[Dict[str, Any]]:
        """Submit batch to Gemini API."""
        if not await self._ensure_batch_client() or not await self._ensure_storage_client():
            return None
        
        try:
            # Read requests from input file
            requests = self.file_ops.read_jsonl_file(input_file)
            
            # Wrap requests in the expected format for Gemini batch
            wrapped_requests = []
            for request in requests:
                # Extract custom_id and wrap request
                custom_id = request.pop('custom_id', None)
                wrapped_requests.append({
                    "custom_id": custom_id,
                    "request": request
                })
            
            # Write wrapped requests back to file
            self.file_ops.write_jsonl_file(wrapped_requests, input_file)
            
            # Upload file to GCS
            timestamp = int(time.time())
            dataset_part = f"_{self.config.dataset_name}" if self.config.dataset_name else ""
            gcs_input_path = f"batch_inputs/{chunk_name}{dataset_part}_{timestamp}.jsonl"
            blob = self._bucket.blob(gcs_input_path)
            blob.upload_from_filename(str(input_file))
            gcs_input_uri = f"gs://{self.config.bucket_name}/{gcs_input_path}"
            
            _print(f"[bold green]✓ Uploaded batch input to {gcs_input_uri}")
            
            # Get model name from first request
            model_id = requests[0].get('model', 'gemini-2.0-flash-001')
            model_resource_name = f"publishers/google/models/{model_id}"
            
            # Create unique output directory
            gcs_output_uri = f"gs://{self.config.bucket_name}/batch_outputs/{chunk_name}{dataset_part}_{timestamp}/"
            
            _print(f"[bold blue]Submitting Vertex AI batch prediction job...")
            _print(f"[bold blue]  Model: {model_resource_name}")
            _print(f"[bold blue]  Input: {gcs_input_uri}")
            _print(f"[bold blue]  Output: {gcs_output_uri}")
            
            # Submit batch job using genai client
            batch_job = self._batch_client.batches.create(
                model=model_resource_name,
                src=gcs_input_uri,
                config=CreateBatchJobConfig(dest=gcs_output_uri)
            )
            
            # Store output mapping for later retrieval
            self._batch_output_mappings[batch_job.name] = gcs_output_uri
            
            _print(f"[bold green]✓ Successfully submitted batch job: {batch_job.name}")
            return {"name": batch_job.name}
            
        except Exception as e:
            self.handle_api_error("Gemini batch submission", e, {
                "chunk_name": chunk_name,
                "input_file": str(input_file)
            })
            return None

    async def _get_batch_status(self, batch_id: str) -> str:
        """Get status of a Gemini batch."""
        if not await self._ensure_batch_client():
            return 'error'
        
        try:
            job = self._batch_client.batches.get(name=batch_id)
            state = job.state
            
            # Map genai JobState to consistent batch provider states
            if state == JobState.JOB_STATE_SUCCEEDED:
                return 'completed'
            elif state == JobState.JOB_STATE_FAILED:
                return 'failed'
            elif state == JobState.JOB_STATE_CANCELLED:
                return 'cancelled'
            elif state in [JobState.JOB_STATE_RUNNING, JobState.JOB_STATE_PENDING]:
                return 'in_progress'
            elif state == JobState.JOB_STATE_PAUSED:
                return 'paused'
            else:
                return str(state).lower()
            
        except Exception as e:
            self.handle_api_error("Gemini status check", e, {"batch_id": batch_id})
            return 'error'

    async def _download_batch_results_from_api(self, batch_id: str) -> Optional[List[Dict[str, Any]]]:
        """Download batch results from Gemini API."""
        if not await self._ensure_batch_client() or not await self._ensure_storage_client():
            return None
        
        try:
            job = self._batch_client.batches.get(name=batch_id)
            
            # Try to get output directory using multiple methods
            output_dir_uri = None
            
            # Method 1: Use stored mapping
            if batch_id in self._batch_output_mappings:
                output_dir_uri = self._batch_output_mappings[batch_id]
                _print(f"[bold blue]Using stored output URI: {output_dir_uri}")
            else:
                # Method 2: Try to get from job config
                try:
                    if hasattr(job, 'config') and hasattr(job.config, 'dest'):
                        output_dir_uri = job.config.dest
                        _print(f"[bold blue]Found output URI from job config: {output_dir_uri}")
                except Exception:
                    pass
                
                # Method 3: Search for output directories that might contain this batch's results
                if not output_dir_uri:
                    try:
                        # Extract the numeric batch ID from the full resource name
                        # Format: projects/1032599528497/locations/us-central1/batchPredictionJobs/7649984177546199040
                        numeric_batch_id = batch_id.split('/')[-1]
                        
                        # Search for output directories in batch_outputs/ that might contain results
                        bucket = self._storage_client.bucket(self.config.bucket_name)
                        prefix = "batch_outputs/"
                        
                        # Look for directories containing the numeric batch ID
                        # or check for recent directories that might contain results
                        potential_dirs = []
                        for blob in bucket.list_blobs(prefix=prefix, delimiter='/'):
                            if blob.name.endswith('/'):
                                dir_name = blob.name.rstrip('/')
                                # Check if directory name contains the numeric batch ID
                                if numeric_batch_id in dir_name:
                                    potential_dirs.append(dir_name)
                        
                        # If we found potential directories, check if they contain results
                        for dir_name in potential_dirs:
                            test_uri = f"gs://{self.config.bucket_name}/{dir_name}/"
                            # Check if this directory contains any results files
                            test_blobs = list(bucket.list_blobs(prefix=dir_name + "/", max_results=5))
                            if test_blobs:
                                output_dir_uri = test_uri
                                _print(f"[bold blue]Found output URI by searching: {output_dir_uri}")
                                break
                                
                    except Exception as e:
                        _print(f"[bold yellow]Could not search for output directory: {e}")
            
            if not output_dir_uri:
                _print(f"[bold red]No output directory found for batch {batch_id}")
                return None
            
            # Parse bucket and prefix from GCS URI
            bucket_name, prefix = output_dir_uri.replace("gs://", "").split("/", 1)
            bucket = self._storage_client.bucket(bucket_name)
            blobs = list(bucket.list_blobs(prefix=prefix))
            
            _print(f"[bold blue]Checking for result files in {output_dir_uri}")
            _print(f"[bold blue]Found {len(blobs)} total files in output directory")
            
            if not blobs:
                _print(f"[bold red]No result files found in {output_dir_uri}")
                return None
            
            # Look for predictions.jsonl (Gemini's standard output)
            result_blobs = []
            predictions_blobs = []
            other_jsonl_blobs = []
            
            for blob in blobs:
                if blob.size > 0 and blob.name.endswith('.jsonl'):
                    if blob.name.endswith('predictions.jsonl'):
                        predictions_blobs.append(blob)
                    else:
                        other_jsonl_blobs.append(blob)
                        
            # Prefer predictions.jsonl files
            if predictions_blobs:
                result_blobs = predictions_blobs
                _print(f"[bold green]Found {len(predictions_blobs)} predictions.jsonl files")
            elif other_jsonl_blobs:
                result_blobs = other_jsonl_blobs
                _print(f"[bold yellow]No predictions.jsonl found, using {len(other_jsonl_blobs)} other JSONL files")
            else:
                _print(f"[bold red]No result files found")
                # List all files for debugging
                _print(f"[bold yellow]Available files in {output_dir_uri}:")
                for blob in blobs:
                    _print(f"  - {blob.name} ({blob.size} bytes)")
                return None
            
            # Download and parse results
            results = []
            for blob in result_blobs:
                content = blob.download_as_text()
                for line in content.strip().split('\n'):
                    if line:
                        try:
                            result = json.loads(line)
                            results.append(result)
                        except json.JSONDecodeError:
                            continue
            
            _print(f"[bold green]Downloaded {len(results)} results from batch {batch_id}")
            return results
            
        except Exception as e:
            self.handle_api_error("Gemini batch download", e, {"batch_id": batch_id})
            return None

    async def list_batches(self, page_size: int = 1000, max_batches: Optional[int] = None) -> List[Dict[str, Any]]:
        """List batch prediction jobs using the genai client.
        
        Args:
            page_size: Number of batches to fetch per page (default: 1000 for maximum efficiency)
            max_batches: Maximum number of batches to return (default: None for all)
            
        Returns:
            List of batch job information dictionaries
        """
        try:
            if not await self._ensure_batch_client():
                return []
                
            batches = []
            batch_count = 0
            
            _print(f"[bold blue]Listing batch prediction jobs (page_size={page_size})...")
            
            # Use the genai client to list batches with pagination
            for job in self._batch_client.batches.list(config=ListBatchJobsConfig(page_size=page_size)):
                if max_batches and batch_count >= max_batches:
                    break
                    
                batch_info = {
                    "name": job.name,
                    "state": job.state,
                    "create_time": getattr(job, 'create_time', None),
                    "update_time": getattr(job, 'update_time', None),
                    "model": getattr(job, 'model', None),
                    "config": getattr(job, 'config', None)
                }
                batches.append(batch_info)
                batch_count += 1
                
                # Log progress for large lists
                if batch_count % 50 == 0:
                    _print(f"[cyan]Retrieved {batch_count} batches...")
            
            _print(f"[bold green]✓ Found {len(batches)} batch prediction jobs")
            return batches
            
        except Exception as e:
            _print(f"[bold red]Failed to list batch prediction jobs: {e}")
            return []
    
    async def delete_batch(self, batch_name: str) -> bool:
        """Delete a batch prediction job using the genai client.
        
        Args:
            batch_name: Name/ID of the batch to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            if not await self._ensure_batch_client():
                return False
                
            self._batch_client.batches.delete(name=batch_name)
            _print(f"[bold green]✓ Deleted batch: {batch_name}")
            return True
            
        except Exception as e:
            _print(f"[bold red]Failed to delete batch {batch_name}: {e}")
            return False