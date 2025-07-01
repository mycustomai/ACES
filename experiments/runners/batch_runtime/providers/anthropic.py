"""
Anthropic batch provider implementation using enhanced base classes.

Simplified by delegating common operations to base classes while handling
Anthropic-specific requirements like custom_id hashing and tool schema conversion.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import anthropic
except ImportError:
    anthropic = None

from rich import print as _print

from agent.src.typedefs import EngineParams
from experiments.config import ExperimentData

from ..services.file_operations import FileOperationsService
from .base import BaseBatchProvider, BatchProviderConfig, ChunkingStrategy


class AnthropicBatchProvider(BaseBatchProvider):
    """Anthropic-specific batch API implementation using enhanced base classes."""

    def __init__(self, file_ops: FileOperationsService, api_key: Optional[str] = None, 
                 dataset_name: Optional[str] = None, remote: bool = False):
        """
        Initialize AnthropicBatchProvider.

        Args:
            file_ops: File operations service
            api_key: Anthropic API key. If not provided, will use ANTHROPIC_API_KEY environment variable.
            dataset_name: Name of the dataset for unique file naming.
            remote: Whether to use remote GCS URLs instead of base64 image data.
                   NOTE: Anthropic's batch API requires publicly accessible URLs for images.
                   Base64 images will cause "Unable to download the file" errors.
        """
        # Check for anthropic package
        if anthropic is None:
            raise ImportError("anthropic package is required for Anthropic batch processing. Install with: pip install anthropic")

        # Create configuration
        config = BatchProviderConfig(
            api_key=api_key or os.getenv('ANTHROPIC_API_KEY'),
            dataset_name=dataset_name
        )

        super().__init__(config, file_ops)
        self.remote = remote

        if not self.config.api_key:
            _print("[bold yellow]Warning: No Anthropic API key configured. Batch submission will fail.")
            _print("[bold yellow]Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")

        self.client = anthropic.Anthropic(api_key=self.config.api_key) if self.config.api_key else None

        # Store custom_id mappings for Anthropic (required due to hashing)
        self._custom_id_mappings: Dict[str, str] = {}  # hashed_id -> original_id

    def create_batch_request(self, data: ExperimentData, engine_params: EngineParams,
                           raw_messages: List[Dict[str, Any]], custom_id: str,
                           tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create Anthropic batch request object."""
        params = {
            "model": engine_params.model,
            "max_tokens": getattr(engine_params, 'max_tokens', 1000),
            "messages": raw_messages
        }

        if hasattr(engine_params, 'temperature'):
            params["temperature"] = engine_params.temperature

        # Add tools if provided (convert from OpenAI format to Anthropic format)
        if tools:
            anthropic_tools = self._convert_openai_tools_to_anthropic(tools)
            if anthropic_tools:
                params["tools"] = anthropic_tools

        # Anthropic has stricter custom_id requirements (1-64 chars, specific pattern)
        # Generate a shorter, compliant custom_id by hashing the original
        hashed_custom_id = hashlib.md5(custom_id.encode()).hexdigest()[:32]
        
        # Store mapping for later resolution
        self._custom_id_mappings[hashed_custom_id] = custom_id

        # Create the batch request
        batch_request = {
            "custom_id": hashed_custom_id,
            "params": params
        }

        return batch_request

    def get_request_chunks(self, requests: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Anthropic uses count-based chunking with 10,000 requests per batch."""
        return self.chunk_requests(requests, ChunkingStrategy.BY_COUNT, chunk_size=10000)

    async def _submit_chunk(self, chunk: List[Dict[str, Any]], config_name: str, 
                           chunk_index: int, submission_context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Submit a single chunk to Anthropic API."""
        if not self.client:
            _print("[bold red]Anthropic client not configured. Cannot submit batch.")
            return None

        # Generate unique chunk name
        chunk_name = self.generate_unique_batch_name(config_name, chunk_index)

        try:
            _print(f"[bold blue]Submitting Anthropic batch '{chunk_name}' with {len(chunk)} requests...")

            # Submit directly to API (no file upload needed for Anthropic)
            response = self.client.beta.messages.batches.create(requests=chunk)
            
            batch_id = response.id
            _print(f"[bold green]✓ Anthropic batch '{chunk_name}' submitted successfully. Batch ID: {batch_id}")

            # Save custom_id mappings for this chunk
            await self._save_custom_id_mappings(config_name, chunk_index)

            # Update experiment tracking if context provided
            if submission_context:
                await self._update_anthropic_experiment_tracking(
                    config_name, chunk, batch_id, submission_context
                )

            return batch_id

        except Exception as e:
            self.handle_api_error(f"Anthropic batch submission", e, 
                                {"chunk_index": chunk_index, "config": config_name})
            return None

    async def monitor_batches(self, batch_ids: List[str]) -> Dict[str, str]:
        """Monitor Anthropic batch processing status using bulk list API with limit=100."""
        if not self.client:
            return {batch_id: 'error' for batch_id in batch_ids}

        status_map = {}
        batch_ids_set = set(batch_ids)  # For faster lookup
        
        try:
            _print(f"[dim]Fetching batch status for {len(batch_ids)} batches using bulk list API (limit=100)...")
            
            # Pagination with smaller limit for better performance
            found_batches = 0
            after_id = None
            page_count = 0
            
            while found_batches < len(batch_ids):
                page_count += 1
                
                # Use the list API with limit=100 for faster responses
                list_params = {"limit": 100}
                if after_id:
                    list_params["after_id"] = after_id
                
                batches_response = await asyncio.to_thread(
                    self.client.beta.messages.batches.list,
                    **list_params
                )
                
                # Process the response and extract status for requested batch IDs
                page_found = 0
                last_batch_id = None
                
                for batch in batches_response.data:
                    last_batch_id = batch.id
                    if batch.id in batch_ids_set:
                        status_map[batch.id] = batch.processing_status
                        found_batches += 1
                        page_found += 1
                
                _print(f"[dim]Page {page_count}: found {page_found} matching batches ({found_batches}/{len(batch_ids)} total)")
                
                # Check if we should continue pagination
                if len(batches_response.data) < 100:
                    # We've reached the end of available batches
                    _print(f"[dim]Reached end of batch list (page size: {len(batches_response.data)})")
                    break
                
                if found_batches == len(batch_ids):
                    # We've found all requested batches
                    _print(f"[dim]Found all requested batches")
                    break
                
                # Set up for next page
                after_id = last_batch_id
                
                # Safety limit to prevent excessive pagination
                if page_count >= 50:
                    _print(f"[bold yellow]Warning: Reached pagination limit of 50 pages, stopping search")
                    break
            
            _print(f"[dim]Bulk list completed: found {found_batches}/{len(batch_ids)} batches across {page_count} pages")
            
            # For any batches not found in the list response, mark as 'not_found'
            for batch_id in batch_ids:
                if batch_id not in status_map:
                    status_map[batch_id] = 'not_found'
            
            return status_map
            
        except Exception as e:
            _print(f"[bold yellow]Warning: Bulk batch monitoring failed, falling back to individual calls: {e}")
            self.handle_api_error(f"Anthropic bulk status check", e, {"batch_count": len(batch_ids)})
            
            # Fallback to individual batch monitoring
            return await self._monitor_batches_individual_fallback(batch_ids)

    async def _monitor_batches_individual_fallback(self, batch_ids: List[str]) -> Dict[str, str]:
        """Fallback method for individual batch monitoring when bulk API fails."""
        status_map = {}
        
        def check_single_batch(batch_id: str) -> Tuple[str, str]:
            """Check status of a single batch."""
            try:
                batch = self.client.beta.messages.batches.retrieve(batch_id)
                return batch_id, batch.processing_status
            except Exception as e:
                self.handle_api_error(f"Anthropic status check", e, {"batch_id": batch_id})
                return batch_id, 'error'

        # Process batches in small chunks for fallback
        chunk_size = 10
        
        _print(f"[dim]Fallback: monitoring {len(batch_ids)} batches with {chunk_size} concurrent calls per chunk...")
        
        for i in range(0, len(batch_ids), chunk_size):
            chunk = batch_ids[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            total_chunks = (len(batch_ids) + chunk_size - 1) // chunk_size
            
            try:
                tasks = [asyncio.to_thread(check_single_batch, batch_id) for batch_id in chunk]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        continue
                    batch_id, status = result
                    status_map[batch_id] = status
                        
            except Exception as e:
                _print(f"[bold yellow]Warning: Fallback chunk {chunk_num} failed, using sequential: {e}")
                for batch_id in chunk:
                    try:
                        batch = self.client.beta.messages.batches.retrieve(batch_id)
                        status_map[batch_id] = batch.processing_status
                    except Exception as e:
                        self.handle_api_error(f"Anthropic status check", e, {"batch_id": batch_id})
                        status_map[batch_id] = 'error'

        return status_map


    async def download_batch_results(self, batch_id: str, output_path: Path) -> Optional[str]:
        """Download batch results for a completed Anthropic batch."""
        if not self.client:
            _print("[bold red]Anthropic client not configured. Cannot download batch results.")
            return None

        try:
            _print(f"[bold blue]Downloading Anthropic results for batch {batch_id}...")

            # Get the batch results
            results = self.client.beta.messages.batches.results(batch_id)
            
            # Write results to file in JSONL format
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                for result in results:
                    # Convert result to dict if needed
                    if hasattr(result, 'model_dump'):
                        result_dict = result.model_dump()
                    elif hasattr(result, '__dict__'):
                        result_dict = result.__dict__
                    else:
                        result_dict = result
                    
                    f.write(json.dumps(result_dict) + '\n')

            _print(f"[bold green]✓ Downloaded Anthropic results to {output_path}")
            return str(output_path)

        except Exception as e:
            self.handle_api_error(f"Anthropic batch download", e, {"batch_id": batch_id})
            return None

    def parse_tool_calls_from_response(self, response_body: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse tool calls from Anthropic batch response."""
        tool_calls = []

        # Anthropic format: content contains tool_use blocks
        content = response_body.get('content', [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'tool_use':
                    # Convert Anthropic tool format to OpenAI-compatible format
                    tool_calls.append({
                        'id': block.get('id'),
                        'type': 'function',
                        'function': {
                            'name': block.get('name'),
                            'arguments': json.dumps(block.get('input', {}))
                        }
                    })

        return tool_calls

    def extract_response_content(self, response_body: Dict[str, Any]) -> str:
        """Extract response content from Anthropic response."""
        content = response_body.get('content', [])
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'text':
                    text_parts.append(block.get('text', ''))
            return '\n'.join(text_parts)
        return str(content)

    def is_response_successful(self, result: Dict[str, Any]) -> bool:
        """Check if Anthropic batch response is successful."""
        # Check if result has error at top level
        if 'error' in result:
            return False
        
        # Check if result has valid response
        if 'result' in result:
            response = result['result']
            # Check for successful batch response
            return response.get('type') == 'succeeded'
        
        return False

    def get_error_message(self, result: Dict[str, Any]) -> str:
        """Extract error message from Anthropic batch response."""
        # Check for top-level error first
        if 'error' in result:
            error = result['error']
            if isinstance(error, dict):
                return error.get('message', f"Error type: {error.get('type', 'unknown')}")
            return str(error)
        
        # Check for result-level error
        if 'result' in result:
            result_obj = result['result']
            if isinstance(result_obj, dict):
                # Check if result type indicates error
                result_type = result_obj.get('type', '')
                if result_type in ['errored', 'canceled', 'expired']:
                    # Try to get error details
                    if 'error' in result_obj:
                        error = result_obj['error']
                        if isinstance(error, dict):
                            error_type = error.get('type', 'unknown')
                            error_message = error.get('message', 'No message provided')
                            return f"Request {result_type}: {error_type} - {error_message}"
                        return f"Request {result_type}: {str(error)}"
                    return f"Request {result_type}"
                elif result_type == 'succeeded':
                    return "No error - request succeeded"
                else:
                    return f"Unexpected result type: {result_type}"
        
        return "Unknown error - unable to parse response"

    def get_response_body_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the response body from a single batch result item."""
        # For Anthropic batch results, the actual message is nested in result.message
        if 'result' in result and 'message' in result['result']:
            return result['result']['message']
        return result.get('result', result)

    # Private helper methods

    def _convert_openai_tools_to_anthropic(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool schema format to Anthropic tool format."""
        anthropic_tools = []

        for tool in tools:
            if tool.get('type') != 'function':
                continue

            function_def = tool.get('function', {})
            anthropic_tool = {
                'name': function_def.get('name'),
                'description': function_def.get('description'),
                'input_schema': function_def.get('parameters', {})
            }

            # Ensure input_schema has required structure
            if 'input_schema' in anthropic_tool:
                schema = anthropic_tool['input_schema']
                if 'type' not in schema:
                    schema['type'] = 'object'
                if 'properties' not in schema:
                    schema['properties'] = {}

            anthropic_tools.append(anthropic_tool)

        return anthropic_tools

    async def _save_custom_id_mappings(self, config_name: str, chunk_index: int):
        """Save custom_id mappings for this chunk."""
        if not self._custom_id_mappings:
            return

        mapping_file = self.file_ops.directories.batch_metadata / f"{config_name}_anthropic_custom_id_mapping_{chunk_index}.json"
        
        try:
            self.file_ops.write_json_file(self._custom_id_mappings, mapping_file)
            _print(f"[bold blue]Saved {len(self._custom_id_mappings)} custom_id mappings to {mapping_file}")
        except Exception as e:
            self.handle_api_error(f"custom_id mapping save", e, {"config": config_name, "chunk": chunk_index})

    def resolve_custom_id(self, hashed_custom_id: str, config_name: str) -> Optional[str]:
        """
        Resolve a hashed custom_id back to its original form.
        
        Args:
            hashed_custom_id: The MD5 hashed custom_id from batch results
            config_name: Configuration name to identify the correct mapping file
            
        Returns:
            Original custom_id if found, None otherwise
        """
        # First check in-memory mappings
        if hashed_custom_id in self._custom_id_mappings:
            return self._custom_id_mappings[hashed_custom_id]
        
        # Try to load mappings from all chunk files for this config
        batch_metadata_dir = self.file_ops.directories.batch_metadata
        
        try:
            # Look for all mapping files for this config
            for mapping_file in batch_metadata_dir.glob(f"{config_name}_anthropic_custom_id_mapping_*.json"):
                try:
                    mappings = self.file_ops.read_json_file(mapping_file)
                    if hashed_custom_id in mappings:
                        return mappings[hashed_custom_id]
                except Exception as e:
                    _print(f"[bold yellow]Warning: Could not read mapping file {mapping_file}: {e}")
                    
        except Exception as e:
            _print(f"[bold yellow]Warning: Error searching for custom_id mappings: {e}")
        
        return None

    async def _update_anthropic_experiment_tracking(self, config_name: str, chunk: List[Dict[str, Any]], 
                                                   batch_id: str, submission_context: Dict[str, Any]):
        """Update experiment tracking with Anthropic-specific logic."""
        # Extract experiment IDs from the requests in this chunk
        experiment_ids = []
        
        for request in chunk:
            hashed_custom_id = request.get('custom_id', '')
            original_custom_id = self._custom_id_mappings.get(hashed_custom_id, '')
            
            if original_custom_id and '|' in original_custom_id:
                # Parse experiment_id from original custom_id
                parts = original_custom_id.split('|')
                if len(parts) >= 3:
                    experiment_id = f"{parts[0]}_{parts[1]}_{parts[2]}"
                    experiment_ids.append(experiment_id)

        if experiment_ids and submission_context:
            # Update tracking through the provided context
            tracking_service = submission_context.get('batch_runtime')
            if tracking_service and hasattr(tracking_service, 'tracking_service'):
                await tracking_service.tracking_service.mark_experiments_submitted(
                    config_name, experiment_ids, [batch_id]
                )