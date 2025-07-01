# ACE-v1: Agent E-commerce Environment Codebase

This repository contains the evaluation framework for studying AI agent purchasing behavior in e-commerce environments.
The system uses browser automation where AI agents interact with a mock Amazon-like website to make purchasing decisions.

## Getting Started

### Prerequisites

This project uses `uv` for dependency management and execution.

- Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/):
  
  For macOS and Linux:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  
  For Windows (PowerShell):
  ```powershell
  irm https://astral.sh/uv/install.ps1 | iex
  ```
  
- Install dependencies:
  ```bash
  uv sync --all-packages
  ```

- Create an `.env` file with required API keys. Use `.env.sample` as a template:
  ```bash
  cp .env.sample .env
  # Edit .env with your API keys
  ```

## Running Experiments

### Full ACERS-v1 Evaluation

Run all models on the complete `ACERS-v1`  dataset:
```bash
uv run run.py
```

_NOTE: it is recommended to use the [batch runtime](#batch-runtime) for evaluating the complete dataset due to cost and speed._

### Experiment Subsets

Use the `--subset` argument to run specific portions of `ACERS-v1`:

```bash
# Run only sanity check evaluations
# TODO: `sanity_checks` subset DNE. Do not use.
uv run run.py --subset sanity_checks

# Run only bias experiments  
uv run run.py --subset bias_experiments

# Run price sanity checks
uv run run.py --subset price_sanity_checks

# Run all experiments (default)
uv run run.py --subset all
```

### Model Selection

Use `--include` and `--exclude` to control which models are evaluated:

```bash
# Run only specific models (by config filename without extension)
uv run run.py --include gpt-4o claude-3.5-sonnet

# Exclude specific models
uv run run.py --exclude gemini-2.5-flash-preview

# Combine with subsets
uv run run.py --subset sanity_checks --include gpt-4o
```

These are mutually exclusive. When using `--exclude`, all other model definitions within `config/models` are loaded.

### Runtime Types

ACERS-v1 supports two main runtime modes:

#### Screenshot Runtime (Default)
Uses pre-captured screenshots from the dataset for faster evaluation:
```bash
uv run run.py --runtime-type screenshot
```
- **Pros**: Fast execution for small-scale evaluations.
- **Cons**: Costly, and long turnaround time for large datasets (e.g.: `bias_experiments`).

All providers are evaluated in parallel.

#### Batch Runtime
Processes experiments in batches using provider-specific batch APIs:
```bash
uv run run.py --runtime-type batch
```
- **Pros**: Cost-effective and faster for large-scale evaluations.
- **Cons**: Does not provide immediate results.

### Advanced Options

```bash
# Enable debug mode for detailed logging and failing on exceptions
uv run run.py --debug

# Use remote screenshot URLs (screenshot runtime only)
# TODO: document this further. Update code to upload to GCS
uv run run.py --runtime-type screenshot --remote

# Force resubmission of batches (batch runtime only)
uv run run.py --runtime-type batch --force-submit
```

> TODO: document API key cycling

## Output

All experiment results are stored in the `experiment_logs/` directory, organized by dataset and model configuration. Results include:
- Detailed interaction logs and agent reasoning traces  
- Final purchase decisions

Results are consolidated per-provider and results are globally aggregated in the `aggregated_experiment_data.csv`

## Advanced Usage

### Custom Datasets

**[WIP]** Documentation for custom datasets is under development and will be added in future releases.

### Custom Providers

**[WIP]** Documentation for custom providers is under development and will be added in future releases.
