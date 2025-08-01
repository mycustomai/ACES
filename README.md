# ACES: Agentic e-CommercE Simulator

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

ACES is a sandbox environment for studying how autonomous AI agents behave when shopping in e-commerce settings. It pairs a platform-agnostic Vision-Language Model (VLM) agent with a fully programmable mock marketplace to enable controlled experiments on AI shopping behavior.

## Overview

ACES enables researchers to:
- Test AI agents' basic rationality and instruction-following capabilities
- Measure product selection patterns and market shares under AI-mediated shopping
- Study how agents respond to platform design elements (rankings, badges, promotions)
- Examine strategic dynamics between AI buyers and sellers

The framework consists of:
- **VLM Shopping Agent**: A browser-based agent that can navigate, evaluate products, and make purchases
- **Mock E-commerce Platform**: A controllable environment with randomizable product attributes, positions, and promotional elements

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

Run all models on the complete `ACERS-v1` dataset:
```bash
uv run run.py
```

_NOTE: it is recommended to use the [batch runtime](#batch-runtime) for evaluating the complete dataset due to cost and speed._

### Experiment Subsets

Use the `--subset` argument to run specific portions of `ACERS`:

```bash
# Run only bias experiments  
uv run run.py --subset price_rationality_check

# Run price sanity checks
uv run run.py --subset rating_rationality_check
```

### Model Selection

Use `--include` and `--exclude` to control which models are evaluated:

```bash
# Run only specific models (by config filename without extension)
uv run run.py --include gpt-4o claude-3.5-sonnet

# Exclude specific models
uv run run.py --exclude gemini-2.5-flash

# Combine with subsets
uv run run.py --subset sanity_checks --include gpt-4o
```

### Runtime Types

ACERS-v1 supports two main runtime modes:

#### Screenshot Runtime (Default)
Uses pre-captured screenshots from the dataset for faster evaluation:
```bash
uv run run.py --runtime-type screenshot
```

#### Batch Runtime
Processes experiments in batches using provider-specific batch APIs:
```bash
uv run run.py --runtime-type batch
```

### Advanced Options

```bash
# Enable debug mode for detailed logging
uv run run.py --debug

# Force resubmission of batches (batch runtime only)
uv run run.py --runtime-type batch --force-submit
```

## Output

All experiment results are stored in the `experiment_logs/` directory, organized by dataset and model configuration. Results include:
- Detailed interaction logs and agent reasoning traces  
- Final purchase decisions
- Aggregated results in `aggregated_experiment_data.csv`

## Repository Layout

```
agent/          # VLM wrapper & tool interface
sandbox/        # mock storefront (Flask + HTML/CSS)
experiments/    # datasets, batching & analysis helpers
config/         # model/provider YAMLs
run.py          # experiment entry‑point
```

## Citation

If you use ACES in your research, please cite:

```bibtex
@article{allouah2025aces,
  title={What is your AI Agent buying? Evaluation, Implications and Emerging Questions for Agentic e-Commerce},
  author={Allouah, Amine and Besbes, Omar and Figueroa, Josué D and Kanoria, Yash and Kumar, Akshit},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open a GitHub issue or contact the authors through the paper correspondence information.
