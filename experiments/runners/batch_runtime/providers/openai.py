"""
OpenAI batch provider implementation using enhanced base classes.

Significantly simplified by delegating common operations to base classes and mixins.
"""

import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
from rich import print as _print

from agent.src.typedefs import EngineParams
from experiments.config import ExperimentData

from ..services.file_operations import FileOperationsService
from .base import BaseBatchProvider, BatchProviderConfig, ChunkingStrategy


class OpenAIBatchProvider(BaseBatchProvider):
    """OpenAI-specific batch API implementation using enhanced base classes."""

    def __init__(self, file_ops: FileOperationsService, api_key: Optional[str] = None, 
                 dataset_name: Optional[str] = None, remote: bool = False):
        """
        Initialize OpenAIBatchProvider.

        Args:
            file_ops: File operations service
            api_key: OpenAI API key. If not provided, will use OPENAI_API_KEY environment variable.
            dataset_name: Name of the dataset for unique file naming.
            remote: Whether to use remote GCS URLs instead of base64 image data.
        """
        # Check for openai package
        if openai is None:
            raise ImportError("openai package is required for OpenAI batch processing. Install with: pip install openai")

        # Create configuration
        config = BatchProviderConfig(
            api_key=api_key or os.getenv('OPENAI_API_KEY'),
            dataset_name=dataset_name
        )

        # Initialize base class
        super().__init__(config, file_ops)
        
        # Store remote flag
        self.remote = remote

        if not self.config.api_key:
            _print("[bold yellow]Warning: No OpenAI API key configured. Batch submission will fail.")
            _print("[bold yellow]Set OPENAI_API_KEY environment variable or pass api_key parameter.")

        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.config.api_key) if self.config.api_key else None

    def create_batch_request(self, data: ExperimentData, engine_params: EngineParams,
                           raw_messages: List[Dict[str, Any]], custom_id: str,
                           tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create OpenAI batch request object."""
        batch_request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": engine_params.model,
                "messages": raw_messages,
                "max_tokens": getattr(engine_params, 'max_tokens', 1000),
                "temperature": getattr(engine_params, 'temperature', 0.7),
            }
        }

        if tools:
            batch_request["body"]["tools"] = tools
            batch_request["body"]["tool_choice"] = "auto"

        return batch_request

    def get_request_chunks(self, requests: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """OpenAI uses size-based chunking with 200MB limit."""
        return self.chunk_requests(requests, ChunkingStrategy.BY_SIZE, max_size_mb=200)

    async def _submit_chunk(self, chunk: List[Dict[str, Any]], config_name: str, 
                           chunk_index: int, submission_context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Submit a single chunk to OpenAI API."""
        # Generate unique chunk name
        chunk_name = self.generate_unique_batch_name(config_name, chunk_index)

        try:
            # Upload file first
            file_id = await self._upload_batch_file(chunk, chunk_name)
            if not file_id:
                return None

            # Submit batch
            batch_id = await self._submit_batch_to_api(file_id, chunk_name)
            return batch_id

        except Exception as e:
            self.handle_api_error(f"OpenAI chunk submission", e, 
                                {"chunk_index": chunk_index, "config": config_name})
            return None

    async def monitor_batches(self, batch_ids: List[str]) -> Dict[str, str]:
        """Monitor OpenAI batch processing status."""
        status_map = {}

        for batch_id in batch_ids:
            try:
                batch_info = await self._check_batch_status(batch_id)
                if batch_info:
                    status_map[batch_id] = batch_info.get('status', 'unknown')
                else:
                    status_map[batch_id] = 'error'
            except Exception as e:
                self.handle_api_error(f"OpenAI status check", e, {"batch_id": batch_id})
                status_map[batch_id] = 'error'

        return status_map

    async def download_batch_results(self, batch_id: str, output_path: Path) -> Optional[str]:
        """Download batch results for a completed OpenAI batch."""
        if not self.client:
            _print("[bold red]OpenAI client not configured. Cannot download batch results.")
            return None

        try:
            # Check batch status first
            status_info = await self._check_batch_status(batch_id)
            if not status_info:
                _print(f"[bold red]Failed to get status for batch {batch_id}")
                return None

            # Check if batch has completed successfully
            status = status_info.get('status')
            if status != 'completed':
                _print(f"[bold yellow]Batch {batch_id} status is '{status}', not 'completed'")
                return None

            # Get output file ID
            output_file_id = status_info.get('output_file_id')
            if not output_file_id:
                _print(f"[bold red]No output file ID found for batch {batch_id}")
                return None

            # Download the file
            _print(f"[bold blue]Downloading results for batch {batch_id}...")
            file_response = self.client.files.content(output_file_id)
            
            # Write to output path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(file_response.content)

            _print(f"[bold green]✓ Downloaded results to {output_path}")
            return str(output_path)

        except Exception as e:
            self.handle_api_error(f"OpenAI batch download", e, {"batch_id": batch_id})
            return None

    def parse_tool_calls_from_response(self, response_body: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse tool calls from OpenAI batch response."""
        tool_calls = []

        if 'choices' in response_body and len(response_body['choices']) > 0:
            choice = response_body['choices'][0]
            if 'message' in choice and 'tool_calls' in choice['message']:
                for tool_call in choice['message']['tool_calls']:
                    tool_calls.append({
                        'id': tool_call.get('id'),
                        'type': tool_call.get('type'),
                        'function': tool_call.get('function', {})
                    })

        return tool_calls

    def extract_response_content(self, response_body: Dict[str, Any]) -> str:
        """Extract response content from OpenAI response."""
        return response_body.get('choices', [{}])[0].get('message', {}).get('content', '')

    def is_response_successful(self, result: Dict[str, Any]) -> bool:
        """Check if OpenAI batch response is successful."""
        return result.get('response', {}).get('status_code') == 200

    def get_error_message(self, result: Dict[str, Any]) -> str:
        """Extract error message from OpenAI batch response."""
        return result.get('error', {}).get('message', 'Unknown error')

    def get_response_body_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the response body from a single batch result item."""
        return result.get('response', {}).get('body', {})

    # Private helper methods

    async def _upload_batch_file(self, batch_requests: List[Dict[str, Any]], chunk_name: str) -> Optional[str]:
        """Upload a batch file to OpenAI and return the file ID."""
        if not self.client:
            raise ValueError("OpenAI client not configured. Cannot upload batch file.")

        # Create JSONL content
        jsonl_content = "\n".join(json.dumps(request) for request in batch_requests)

        try:
            _print(f"[bold blue]Uploading OpenAI batch file '{chunk_name}' with {len(batch_requests)} requests...")

            # Use BytesIO to create file-like object
            file_obj = BytesIO(jsonl_content.encode('utf-8'))
            file_obj.name = f"{chunk_name}.jsonl"

            file_response = self.client.files.create(
                file=file_obj,
                purpose="batch"
            )

            file_id = file_response.id
            _print(f"[bold green]✓ OpenAI file '{chunk_name}' uploaded successfully. File ID: {file_id}")
            return file_id

        except Exception as e:
            self.handle_api_error(f"OpenAI file upload", e, {"chunk_name": chunk_name})
            return None

    async def _submit_batch_to_api(self, file_id: str, chunk_name: str) -> Optional[str]:
        """Submit a batch job to OpenAI using an uploaded file."""
        if not self.client:
            raise ValueError("OpenAI client not configured. Cannot submit batch.")

        try:
            _print(f"[bold blue]Submitting batch '{chunk_name}' to OpenAI API...")

            batch = self.client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )

            batch_id = batch.id
            _print(f"[bold green]✓ OpenAI batch '{chunk_name}' submitted successfully. Batch ID: {batch_id}")
            return batch_id

        except Exception as e:
            self.handle_api_error(f"OpenAI batch submission", e, {"chunk_name": chunk_name})
            return None

    async def _check_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Check the status of a submitted batch."""
        if not self.client:
            raise ValueError("OpenAI client not configured. Cannot check batch status.")

        try:
            batch = self.client.batches.retrieve(batch_id)
            # Convert to dict for consistency with other providers
            return {
                "id": batch.id,
                "status": batch.status,
                "input_file_id": batch.input_file_id,
                "output_file_id": batch.output_file_id,
                "request_counts": {
                    "total": batch.request_counts.total if batch.request_counts else 0,
                    "completed": batch.request_counts.completed if batch.request_counts else 0,
                    "failed": batch.request_counts.failed if batch.request_counts else 0
                }
            }

        except Exception as e:
            self.handle_api_error(f"OpenAI batch status check", e, {"batch_id": batch_id})
            return None