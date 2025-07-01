"""
Batch runtime submodule for AI agent impact experiments.

This module consolidates all batch processing functionality including:
- Batch providers (OpenAI, Anthropic, Gemini)  
- Core services (file operations, experiment tracking)
- Simplified batch evaluation runtime

## Import

Import BatchEvaluationRuntime from the main runners module:

```python
from experiments.runners import BatchEvaluationRuntime
```

Or import directly from this module:

```python
from experiments.runners.batch_runtime import BatchEvaluationRuntime
```
"""

from ..services.screenshot_validation import ScreenshotValidationService
from ..services.worker_service import ExperimentWorkerService
from .providers.anthropic import AnthropicBatchProvider
from .providers.base import BaseBatchProvider, BatchProvider
from .providers.gemini import GeminiBatchProvider
from .providers.openai import OpenAIBatchProvider
from .runtime import BatchEvaluationRuntime
from .services.batch_operations import BatchOperationsService
from .services.experiment_tracking import ExperimentTrackingService
from .services.file_operations import FileOperationsService

__all__ = [
    'BatchEvaluationRuntime',
    'ExperimentTrackingService',
    'FileOperationsService',
    'BatchOperationsService',
    'ScreenshotValidationService',
    'ExperimentWorkerService',
    'BatchProvider',
    'BaseBatchProvider',
    'OpenAIBatchProvider',
    'AnthropicBatchProvider',
    'GeminiBatchProvider'
]