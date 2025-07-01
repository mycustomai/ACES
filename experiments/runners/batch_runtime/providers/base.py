"""
Enhanced base classes and mixins for batch providers.

Provides common functionality to reduce code duplication across providers.
"""

import asyncio
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import backoff
from rich import print as _print
from rich.progress import Progress

from agent.src.typedefs import EngineParams
from experiments.config import ExperimentData

from ..services.file_operations import FileOperationsService

# Global lock to prevent multiple Rich Progress displays from conflicting
_progress_lock = threading.Lock()


class ChunkingStrategy(Enum):
    """Strategies for chunking batch requests."""
    BY_COUNT = "by_count"
    BY_SIZE = "by_size"


@dataclass
class BatchMetadata:
    """Metadata for a submitted batch."""
    batch_id: str
    chunk_name: str
    request_count: int
    submitted_at: float
    status: str = "submitted"
    config_name: Optional[str] = None


@dataclass
class BatchProviderConfig:
    """Configuration for batch providers."""
    api_key: Optional[str] = None
    dataset_name: Optional[str] = None
    project_id: Optional[str] = None  # For Google Cloud
    location: Optional[str] = None    # For Google Cloud
    bucket_name: Optional[str] = None # For GCS
    
    def validate(self) -> List[str]:
        """Return list of validation errors."""
        errors = []
        if not self.api_key:
            errors.append("API key is required")
        return errors


class BatchProvider(ABC):
    """Abstract base class for batch API providers."""

    @abstractmethod
    def create_batch_request(self, data: ExperimentData, engine_params: EngineParams,
                           raw_messages: List[Dict[str, Any]], custom_id: str,
                           tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a batch request object for the provider."""
        pass

    @abstractmethod
    async def upload_and_submit_batches(self, batch_requests: List[Dict[str, Any]],
                                      config_name: str, experiment_logs_dir: Path, 
                                      submission_context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Upload and submit batch requests to the provider's API."""
        pass

    @abstractmethod
    async def monitor_batches(self, batch_ids: List[str]) -> Dict[str, str]:
        """Monitor batch processing status."""
        pass

    @abstractmethod
    def parse_tool_calls_from_response(self, response_body: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse tool calls from the provider's batch response format."""
        pass

    @abstractmethod
    def extract_response_content(self, response_body: Dict[str, Any]) -> str:
        """Extract response content from the provider's batch response format."""
        pass

    @abstractmethod
    def is_response_successful(self, result: Dict[str, Any]) -> bool:
        """Check if the provider's batch response is successful."""
        pass

    @abstractmethod
    def get_error_message(self, result: Dict[str, Any]) -> str:
        """Extract error message from failed batch response."""
        pass

    @abstractmethod
    def get_response_body_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the response body from a single batch result item."""
        pass

    @abstractmethod
    async def download_batch_results(self, batch_id: str, output_path: Path) -> Optional[str]:
        """Download batch results for a completed batch."""
        pass


class FileOperationsMixin:
    """Mixin providing standardized file I/O operations."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._file_ops: Optional[FileOperationsService] = None
    
    @property
    def file_ops(self) -> FileOperationsService:
        """Get or create file operations service."""
        if self._file_ops is None:
            raise RuntimeError("FileOperationsService not initialized. Call set_file_ops() first.")
        return self._file_ops
    
    def set_file_ops(self, file_ops: FileOperationsService):
        """Set the file operations service."""
        self._file_ops = file_ops
    
    def write_batch_input_file(self, requests: List[Dict[str, Any]], config_name: str, 
                              chunk_name: str) -> Path:
        """Write batch input requests to JSONL file."""
        file_path = self.file_ops.get_batch_input_file(config_name, chunk_name)
        return self.file_ops.write_jsonl_file(requests, file_path)
    
    def generate_unique_batch_name(self, config_name: str, chunk_index: int = 0) -> str:
        """Generate unique batch name with timestamp."""
        timestamp = int(time.time() * 1000)
        if chunk_index > 0:
            return f"{config_name}_chunk_{chunk_index}_{timestamp}"
        return f"{config_name}_{timestamp}"


class ChunkingMixin:
    """Mixin providing request chunking strategies."""
    
    def chunk_by_count(self, requests: List[Dict[str, Any]], chunk_size: int) -> List[List[Dict[str, Any]]]:
        """
        Chunk requests by count.
        
        Args:
            requests: List of batch requests
            chunk_size: Maximum requests per chunk
            
        Returns:
            List of request chunks
        """
        chunks = []
        for i in range(0, len(requests), chunk_size):
            chunk = requests[i:i + chunk_size]
            chunks.append(chunk)
        return chunks
    
    def chunk_by_size(self, requests: List[Dict[str, Any]], max_size_mb: int) -> List[List[Dict[str, Any]]]:
        """
        Chunk requests by total size.
        
        Args:
            requests: List of batch requests
            max_size_mb: Maximum size per chunk in MB
            
        Returns:
            List of request chunks
        """
        max_size_bytes = max_size_mb * 1024 * 1024
        chunks = []
        current_chunk = []
        current_size = 0
        
        for request in requests:
            request_size = len(json.dumps(request).encode('utf-8'))
            
            if current_size + request_size > max_size_bytes and current_chunk:
                # Start new chunk
                chunks.append(current_chunk)
                current_chunk = [request]
                current_size = request_size
            else:
                current_chunk.append(request)
                current_size += request_size
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def chunk_requests(self, requests: List[Dict[str, Any]], strategy: ChunkingStrategy, 
                      **kwargs) -> List[List[Dict[str, Any]]]:
        """
        Chunk requests using specified strategy.
        
        Args:
            requests: List of batch requests
            strategy: Chunking strategy to use
            **kwargs: Strategy-specific parameters
            
        Returns:
            List of request chunks
        """
        if strategy == ChunkingStrategy.BY_COUNT:
            chunk_size = kwargs.get('chunk_size', 1000)
            return self.chunk_by_count(requests, chunk_size)
        elif strategy == ChunkingStrategy.BY_SIZE:
            max_size_mb = kwargs.get('max_size_mb', 100)
            return self.chunk_by_size(requests, max_size_mb)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")


class MonitoringMixin(ABC):
    """Mixin providing common batch monitoring patterns."""
    @abstractmethod
    async def monitor_batches(self, batch_ids: List[str]) -> Dict[str, str]:
        pass

    async def monitor_batches_concurrent(self, batch_ids: List[str], 
                                       max_concurrent: int = 10) -> Dict[str, str]:
        """
        Monitor batch statuses with concurrent processing for improved performance.
        
        This method provides batching and concurrency optimizations for monitoring
        large numbers of batches efficiently. Providers can override this method
        to implement bulk status checking if their APIs support it.
        
        Args:
            batch_ids: List of batch IDs to monitor
            max_concurrent: Maximum number of concurrent status checks
            
        Returns:
            Dictionary mapping batch_id to status
        """
        if not batch_ids:
            return {}
        
        # If provider has bulk monitoring capability, use it
        if hasattr(self, '_monitor_batches_bulk') and callable(getattr(self, '_monitor_batches_bulk')):
            try:
                return await self._monitor_batches_bulk(batch_ids)
            except Exception as e:
                _print(f"[bold yellow]Bulk monitoring failed, falling back to concurrent: {e}")
        
        # Use concurrent individual status checks with semaphore for rate limiting
        # Increased concurrency for faster batch monitoring
        semaphore = asyncio.Semaphore(min(max_concurrent * 2, 20))
        
        async def check_single_batch(batch_id: str) -> tuple[str, str]:
            """Check status of a single batch with semaphore."""
            async with semaphore:
                try:
                    # Call the provider's individual status check method
                    if hasattr(self, '_get_batch_status'):
                        status = await self._get_batch_status(batch_id)
                        return batch_id, status
                    else:
                        # Fallback to the abstract monitor_batches method for single batch
                        status_map = await self.monitor_batches([batch_id])
                        return batch_id, status_map.get(batch_id, 'error')
                except Exception as e:
                    if hasattr(self, 'handle_api_error'):
                        self.handle_api_error(f"batch status check", e, {"batch_id": batch_id})
                    return batch_id, 'error'
        
        # Execute all status checks concurrently
        tasks = [check_single_batch(batch_id) for batch_id in batch_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        status_map = {}
        for result in results:
            if isinstance(result, Exception):
                _print(f"[bold red]Unexpected error in concurrent monitoring: {result}")
                continue
            
            batch_id, status = result
            status_map[batch_id] = status
        
        return status_map
    
    async def monitor_with_progress(self, batch_ids: List[str], check_interval: int = 10,
                                  max_consecutive_failures: int = 5) -> Dict[str, str]:
        """
        Monitor batches with progress tracking and error handling.
        
        Args:
            batch_ids: List of batch IDs to monitor
            check_interval: Seconds between status checks
            max_consecutive_failures: Maximum consecutive failures before giving up
            
        Returns:
            Final status map for all batches
        """
        consecutive_failures = 0
        
        while True:
            try:
                # Use concurrent monitoring for better performance with large datasets
                status_map = await self.monitor_batches_concurrent(batch_ids)
                
                if not status_map:
                    consecutive_failures += 1
                    _print(f"[bold red]Status check returned empty result")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        _print(f"[bold red]Too many consecutive failures. Stopping monitoring.")
                        break
                    
                    await asyncio.sleep(check_interval)
                    continue
                
                # Reset failure counter on successful check
                consecutive_failures = 0
                
                # Count statuses
                completed = sum(1 for status in status_map.values() 
                              if status in ['completed', 'ended'])
                failed = sum(1 for status in status_map.values() 
                           if status in ['failed', 'error', 'cancelled'])
                in_progress = len(status_map) - completed - failed
                
                _print(f"[bold cyan]Batch status: {completed} completed, {in_progress} in progress, {failed} failed")
                
                # Check if all batches are done
                if in_progress == 0:
                    _print(f"[bold green]All batches completed")
                    return status_map
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                consecutive_failures += 1
                _print(f"[bold red]Error monitoring batches: {e}")
                
                if consecutive_failures >= max_consecutive_failures:
                    _print(f"[bold red]Too many consecutive monitoring failures. Stopping.")
                    break
                
                await asyncio.sleep(check_interval)
        
        # Return last known status or empty dict
        try:
            return await self.monitor_batches_concurrent(batch_ids)
        except Exception:
            return {}


class ErrorHandlingMixin:
    """Mixin providing standardized error handling and logging."""
    
    def handle_api_error(self, operation: str, error: Exception, context: Dict = None):
        """Standardized error logging and handling."""
        context_str = f" (context: {context})" if context else ""
        _print(f"[bold red]Error during {operation}: {error}{context_str}")
    
    def log_batch_failure(self, batch_id: str, error: str):
        """Standardized batch failure logging."""
        _print(f"[bold red]Batch {batch_id} failed: {error}")
    
    def validate_batch_response(self, response: Dict) -> tuple[bool, Optional[str]]:
        """Common response validation logic."""
        if not response:
            return False, "Empty response"
        
        if 'error' in response:
            return False, response['error']
        
        return True, None


class BaseBatchProvider(BatchProvider, FileOperationsMixin, ChunkingMixin, 
                       MonitoringMixin, ErrorHandlingMixin):
    """Enhanced base class with common functionality for all batch providers."""
    
    def __init__(self, config: BatchProviderConfig, file_ops: FileOperationsService):
        super().__init__()
        self.config = config
        self.set_file_ops(file_ops)
        self._submitted_batches: List[BatchMetadata] = []
        
        # Validate configuration
        errors = config.validate()
        if errors:
            _print(f"[bold yellow]Configuration warnings: {', '.join(errors)}")
    
    def track_submitted_batch(self, batch_id: str, config_name: str, chunk_name: str, 
                             request_count: int) -> BatchMetadata:
        """Track a submitted batch."""
        metadata = BatchMetadata(
            batch_id=batch_id,
            chunk_name=chunk_name,
            request_count=request_count,
            submitted_at=time.time(),
            config_name=config_name
        )
        self._submitted_batches.append(metadata)
        return metadata
    
    def get_submitted_batches(self) -> List[BatchMetadata]:
        """Get list of all submitted batches."""
        return self._submitted_batches.copy()
    
    def save_batch_metadata(self, config_name: str, experiment_logs_dir: Path):
        """Save batch metadata to file."""
        metadata_file = self.file_ops.get_batch_metadata_file(config_name)
        
        # Convert to serializable format
        metadata_dict = {
            "provider": self.__class__.__name__,
            "config_name": config_name,
            "batches": [
                {
                    "batch_id": batch.batch_id,
                    "chunk_name": batch.chunk_name,
                    "request_count": batch.request_count,
                    "submitted_at": batch.submitted_at,
                    "status": batch.status
                }
                for batch in self._submitted_batches
            ]
        }
        
        self.file_ops.write_json_file(metadata_dict, metadata_file)
        _print(f"[bold blue]Saved batch metadata to {metadata_file}")
    
    async def upload_and_submit_batches(self, batch_requests: List[Dict[str, Any]],
                                      config_name: str, experiment_logs_dir: Path, 
                                      submission_context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Default implementation that chunks requests and submits them.
        
        Subclasses should override this method with provider-specific logic.
        """
        if not batch_requests:
            return []
        
        # Get chunking strategy from subclass
        chunks = self.get_request_chunks(batch_requests)
        submitted_batch_ids = []
        
        # Try to acquire progress lock non-blocking to avoid conflicts
        progress_acquired = _progress_lock.acquire(blocking=False)
        
        try:
            if progress_acquired:
                # Use progress display if we can acquire the lock
                with Progress() as progress:
                    task = progress.add_task(f"Submitting {config_name} batches", total=len(chunks))
                    
                    for chunk_index, chunk in enumerate(chunks):
                        try:
                            batch_id = await self._submit_chunk_retry(chunk, config_name, chunk_index, submission_context)
                            if batch_id:
                                submitted_batch_ids.append(batch_id)
                                self.track_submitted_batch(batch_id, config_name, f"chunk_{chunk_index}", len(chunk))
                            
                            progress.update(task, advance=1)
                            
                        except Exception as e:
                            self.handle_api_error(f"chunk submission {chunk_index}", e, {"config": config_name})
                            raise
            else:
                # Fallback to simple logging if progress display is busy
                _print(f"[bold blue]Submitting {len(chunks)} batches for {config_name} (progress display busy)...")
                
                for chunk_index, chunk in enumerate(chunks):
                    try:
                        _print(f"[dim]Submitting chunk {chunk_index + 1}/{len(chunks)} for {config_name}...")
                        batch_id = await self._submit_chunk_retry(chunk, config_name, chunk_index, submission_context)
                        if batch_id:
                            submitted_batch_ids.append(batch_id)
                            self.track_submitted_batch(batch_id, config_name, f"chunk_{chunk_index}", len(chunk))
                        
                    except Exception as e:
                        self.handle_api_error(f"chunk submission {chunk_index}", e, {"config": config_name})
                        raise
        finally:
            if progress_acquired:
                _progress_lock.release()
        
        # Save metadata
        self.save_batch_metadata(config_name, experiment_logs_dir)
        
        return submitted_batch_ids
    
    @abstractmethod
    def get_request_chunks(self, requests: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Get request chunks using provider-specific strategy."""
        pass

    @backoff.on_exception(backoff.expo, Exception, max_time=60, max_tries=5)
    async def _submit_chunk_retry(self, chunk: List[Dict[str, Any]], config_name: str,
                                  chunk_index: int, submission_context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Wrapper for _submit_chunk to retry on exceptions."""
        return await self._submit_chunk(chunk, config_name, chunk_index, submission_context)

    @abstractmethod
    async def _submit_chunk(self, chunk: List[Dict[str, Any]], config_name: str, 
                           chunk_index: int, submission_context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Submit a single chunk to the provider API."""
        pass