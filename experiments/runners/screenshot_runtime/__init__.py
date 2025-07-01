"""
Screenshot runtime submodule for AI agent impact experiments.

This module consolidates all screenshot-based experiment functionality including:
- Base classes for screenshot-based experiments
- Local dataset runtime (filesystem screenshots)
- HuggingFace Hub dataset runtime (embedded screenshots)

## Import

Import runtimes from the main runners module:

```python
from experiments.runners import LocalDatasetRuntime, HFHubDatasetRuntime
```

Or import directly from this module:

```python
from experiments.runners.screenshot_runtime import (
    BaseScreenshotRuntime,
    LocalDatasetRuntime,
    HFHubDatasetRuntime
)
```
"""

from .base import BaseScreenshotRuntime
from .hf_hub_dataset import HFHubDatasetRuntime
from .local_dataset import LocalDatasetRuntime

__all__ = ["BaseScreenshotRuntime", "LocalDatasetRuntime", "HFHubDatasetRuntime"]
