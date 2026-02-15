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
  
  To use the `analysis` functionality or development-dependencies:
  ```bash
  # installs BOTH dev and analysis dependencies
  uv sync --all-groups --all-packages
  
  # installs only dev
  uv sync --dev --all-packages
  ```
  
  Use the `uv sync --all-groups --all-packages` by default to install _all_ dependencies for various workflows.

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

- **ACE-BB Dataset:** https://huggingface.co/datasets/My-Custom-AI/ACE-BB
- **ACE-RS Dataset:** https://huggingface.co/datasets/My-Custom-AI/ACE-RS
- **ACE-SR Dataset:** https://huggingface.co/datasets/My-Custom-AI/ACE-SR

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

## Analysis

The `analysis` CLI processes experiment output data to generate summary statistics and model coefficients.

To use the `analysis` script, the `analysis` dependency group needs to be installed. Be sure to use the `uv sync --all-groups` or `uv sync --group analysis`
to install the required dependencies.

```
 Usage: analysis [OPTIONS] COMMAND [ARGS]...

 Analysis CLI for experiment data.

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.      │
│ --show-completion             Show completion for the current shell, to copy │
│                               it or customize the installation.              │
│ --help                        Show this message and exit.                    │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ market-share        Analyze market share from a CSV file.                    │
│ choice-model        Generate a choice-model from a CSV file.                 │
│ rationality-suite   Rationality suite sanity checks                          │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### Sub-command to Dataset Mapping

| Analysis Command                | Dataset                       | Subset                              |
|---------------------------------|-------------------------------|-------------------------------------|
| `market-share`                  | bb (Choice Behavior & Biases) | `market_share` or `choice_behavior` |
| `choice-model`                  | bb (Choice Behavior & Biases) | `choice_behavior`                   |
| `rationality-suite price`       | rs (Rationality Suite)        | `relative_price`                    |
| `rationality-suite rating`      | rs (Rationality Suite)        | `rating`                            |
| `rationality-suite instruction` | rs (Rationality Suite)        | `instruction_following`             |
| `rationality-suite ar_price`    | rs (Rationality Suite)        | `absolute_and_random_price`         |

### Usage Examples

```bash
# Analyze market share from bb dataset results
uv run analysis market-share experiment_logs/aggregated_experiment_data.csv

# Generate choice model coefficients
uv run analysis choice-model experiment_logs/aggregated_experiment_data.csv

# Run rationality suite sanity checks
uv run analysis rationality-suite price experiment_logs/aggregated_experiment_data.csv
uv run analysis rationality-suite rating experiment_logs/aggregated_experiment_data.csv
```

The `artifacts/analysis/` directory is the target output location of result CSV.

## Visualization

The `visualization` CLI generates plots from analysis results to visualize model performance, feature impacts, and sanity check failures.

To use the `visualization` script, the `analysis` dependency group needs to be installed (same as Analysis above).

```
 Usage: visualization [OPTIONS] COMMAND [ARGS]...

 Visualization CLI for generating plots from analysis results.

╭─ Commands ───────────────────────────────────────────────────────────────╮
│ position-bias     Generate position bias visualizations                  │
│ heatmap          Generate position probability heatmaps                   │
│ feature-impact   Generate feature impact visualizations                   │
│ sanity-checks    Generate sanity check visualizations                     │
╰──────────────────────────────────────────────────────────────────────────╯
```

### Visualization Types

#### 1. Position Bias (from choice model)
Visualizes how product position in the 2×4 grid affects selection probability.

```bash
# Generate position bias plots (all_providers.png + by_provider.png)
uv run visualization position-bias artifacts/analysis/20260122104301_choice_model.csv
```

**Output:** `artifacts/visualization/position_bias/`

#### 2. Heatmaps (from choice model)
Generates 2×4 position probability heatmaps for each model.

```bash
# Generate heatmaps for all models
uv run visualization heatmap artifacts/analysis/20260122104301_choice_model.csv
```

**Output:** `artifacts/visualization/heatmaps/`

#### 3. Feature Impact (from choice model)
Shows how features (rating, price, tags) impact selection probability.

```bash
# Generate all feature impact plots (rating, price, sponsored tag, overall pick)
# Creates 8 plots: both all_providers and by_provider versions for each feature
uv run visualization feature-impact all artifacts/analysis/20260122104301_choice_model.csv

# Or generate specific feature plots:
uv run visualization feature-impact rating artifacts/analysis/20260122104301_choice_model.csv  # 2 plots
uv run visualization feature-impact price artifacts/analysis/20260122104301_choice_model.csv   # 2 plots
uv run visualization feature-impact tags artifacts/analysis/20260122104301_choice_model.csv    # 4 plots (2 per tag)
```

**Output:** `artifacts/visualization/feature_impact/`

**Note:**
- Each command generates **both** `all_providers` (all models on one plot) and `by_provider` (3 subplots) versions
- The `tags` command generates plots for both "Sponsored Tag" and "Overall Pick" features (4 plots total)

#### 4. Sanity Checks (from rationality suite analysis)
Visualizes sanity check failure rates across models for rating, price, and instruction following experiments.

```bash
# Generate rating sanity check plots
uv run visualization sanity-checks rating artifacts/analysis/20260121184134_rating_sanity_check.csv

# Generate price sanity check plots (combines price and ar_price data)
uv run visualization sanity-checks price \
  artifacts/analysis/20260212101910_price_sanity_check.csv \
  artifacts/analysis/20260213120645_ar_price_sanity_check.csv

# Generate instruction following plots
uv run visualization sanity-checks instruction artifacts/analysis/20260212101915_instruction_sanity_check.csv

# Generate ALL sanity check plots at once
uv run visualization sanity-checks all \
  artifacts/analysis/20260121184134_rating_sanity_check.csv \
  artifacts/analysis/20260212101910_price_sanity_check.csv \
  artifacts/analysis/20260213120645_ar_price_sanity_check.csv \
  artifacts/analysis/20260212101915_instruction_sanity_check.csv
```

**Output:** `artifacts/visualization/sanity_checks/`

#### 5. Price-Equivalent Trade-offs (from choice model)
Shows by what percentage a seller can raise (or must cut) price to keep utility constant when adding (or losing) a feature.

```bash
# Generate price-equivalent trade-off plots for all features
# Creates 10 plots: 4 individual features + 1 combined (each with all_providers and by_provider versions)
uv run visualization price-tradeoffs artifacts/analysis/20260122104301_choice_model.csv
```

**Output:** `artifacts/visualization/price_equivalent_tradeoffs/`

**Features analyzed:**
- **Overall Pick tag**: Price increase possible when adding this tag
- **Rating +0.1**: Price increase possible when rating improves by 0.1
- **Double Reviews**: Price increase possible when review count doubles
- **Sponsored Tag**: Price cut needed to offset negative perception

**Note:** Positive percentages indicate the seller can raise prices; negative percentages indicate the seller must cut prices.

### Visualization Output Structure

All visualization plots are organized in subdirectories under `artifacts/visualization/`:

```
artifacts/visualization/
├── position_bias/                  # Position bias across models
├── heatmaps/                      # 2×4 position probability heatmaps
├── feature_impact/                # Feature impact on selection probability
├── sanity_checks/                 # Rationality suite sanity check failure rates
└── price_equivalent_tradeoffs/    # Price-equivalent trade-offs for features
```

Each visualization type generates plots grouped by provider (Anthropic, Google, OpenAI) showing model performance evolution by release date.

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
  journal={arXiv preprint arXiv:2508.02630},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open a GitHub issue or contact the authors through the paper correspondence information.
