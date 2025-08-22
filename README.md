# ACES: Agentic e-CommercE Simulator

[![arXiv](https://img.shields.io/badge/arXiv-2508.02630-b31b1b.svg)](https://arxiv.org/abs/2508.02630)

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
  
  For AWS Bedrock models, configure the following environment variables:
  - `AWS_REGION`: The AWS region where your Bedrock models are available (e.g., `us-east-1`)
  - `AWS_ACCESS_KEY_ID`: Your AWS access key ID (optional if using default credentials)
  - `AWS_SECRET_ACCESS_KEY`: Your AWS secret access key (optional if using default credentials)
  - `AWS_SESSION_TOKEN`: AWS session token for temporary credentials (optional)
  - `AWS_CREDENTIALS_PROFILE_NAME`: AWS profile name if using AWS SSO or named profiles (optional)
  
  If no AWS credentials are provided in the `.env` file, the default AWS credential chain will be used (e.g., IAM role, AWS CLI configuration, or environment variables).

## Running Experiments

### Datasets

The ACES evaluation datasets are available on Hugging Face:

**ACE-BB Dataset:** https://huggingface.co/datasets/My-Custom-AI/ACE-BB
**ACE-RS Dataset:** https://huggingface.co/datasets/My-Custom-AI/ACE-RS
**ACE-SR Dataset:** https://huggingface.co/datasets/My-Custom-AI/ACE-SR

#### Dataset Configuration

When running experiments, use the `--hf-dataset` argument with one of these shorthand strings:
- `"bb"` - Choice Behavior & Biases
- `"sr"` - Seller's Reaction 
- `"rs"` - Rationality Suite

### Experiment Subsets

Use the `--subset` argument to run specific experiment types. Valid subset names depend on the dataset:

#### For "bb" (Choice Behavior & Biases):
```bash
# Run choice behavior experiments
uv run run.py --hf-dataset bb --subset choice_behavior

# Run market share experiments
uv run run.py --hf-dataset bb --subset market_share
```

#### For "sr" (Seller's Reaction):
```bash
# No subset required - runs title change experiments
uv run run.py --hf-dataset sr
```

#### For "rs" (Rationality Suite):
```bash
# Run absolute and random price experiments
uv run run.py --hf-dataset rs --subset absolute_and_random_price

# Run instruction following experiments
uv run run.py --hf-dataset rs --subset instruction_following

# Run rating experiments
uv run run.py --hf-dataset rs --subset rating

# Run relative price experiments
uv run run.py --hf-dataset rs --subset relative_price
```

### Model Selection

Use `--include` and `--exclude` to control which models are evaluated:

```bash
# Combine dataset, subset, and model selection
uv run run.py --hf-dataset rs --subset rating --include gpt-4o
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
