import asyncio
import os
import argparse
from typing import Optional

from dotenv import load_dotenv

from experiments.engine_loader import load_all_model_engine_params
from experiments.runners import (
    BatchOrchestratorRuntime,
    HeadlessRuntime,
    ScreenshotRuntime,
    SimpleEvaluationRuntime,
)

load_dotenv()

# Configuration for parallel execution
MAX_CONCURRENT_MODELS = int(os.getenv("MAX_CONCURRENT_MODELS", "20"))
EXPERIMENT_LABEL_FILTER = None
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
RUNTIME_TYPE = os.getenv("RUNTIME_TYPE", "screenshot").lower()

DATASET_MAPPING = {
    "rs": "My-Custom-AI/ACE-RS",
    "bb": "My-Custom-AI/ACE-BB",
    "sr": "My-Custom-AI/ACE-SR",
}

DEFAULT_HF_DATASET = DATASET_MAPPING["rs"]

DATASET_SUBSET_REQUIREMENTS = {
    "My-Custom-AI/ACE-BB": ["choice_behavior", "market_share"],
    "My-Custom-AI/ACE-SR": None,
    "My-Custom-AI/ACE-RS": ["absolute_and_random_price", "instruction_following", "rating", "relative_price"],
}

def validate_dataset_subset(dataset_name: str, subset: Optional[str]) -> None:
    """
    Validate that the dataset and subset combination is correct.

    Only validates known ACES datasets. For unknown datasets, no validation is performed.

    Args:
        dataset_name: Full HuggingFace dataset name (e.g., "My-Custom-AI/ACE-BB")
        subset: Subset name or None

    Raises:
        ValueError: If the dataset/subset combination is invalid
    """
    if dataset_name not in DATASET_SUBSET_REQUIREMENTS:
        return

    valid_subsets = DATASET_SUBSET_REQUIREMENTS[dataset_name]

    if valid_subsets is None:
        if subset is not None:
            raise ValueError(
                f"Dataset {dataset_name} does not support subsets. "
                f"Please omit the --subset argument."
            )
    else:
        if subset is None:
            raise ValueError(
                f"Dataset {dataset_name} requires a subset. "
                f"Valid subsets: {', '.join(valid_subsets)}"
            )
        if subset not in valid_subsets:
            raise ValueError(
                f"Invalid subset '{subset}' for dataset {dataset_name}. "
                f"Valid subsets: {', '.join(valid_subsets)}"
            )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AI Agent Impact Evaluation')
    parser.add_argument('--runtime-type', choices=['simple', 'screenshot', 'batch', 'headless'],
                        default=RUNTIME_TYPE, help='Runtime type to use')
    parser.add_argument('--local-dataset', type=str,
                        help='Local dataset path')
    parser.add_argument('--hf-dataset', type=str, default=DEFAULT_HF_DATASET,
                        help=f"Hugging Face dataset name to use (e.g., \"{DEFAULT_HF_DATASET}\")")
    parser.add_argument('--subset', type=str,
                        help="Subset of ACERS dataset to use (e.g., 'sanity_checks', 'bias_experiments'). View dataset card for more details.",
                        default=None)
    parser.add_argument('--debug', action='store_true', default=DEBUG_MODE,
                        help='Enable debug mode')
    parser.add_argument('--remote', action='store_true', default=False,
                        help='Use remote GCS URLs instead of local screenshot bytes (screenshot and batch runtimes)')
    parser.add_argument('--include', nargs='+', help='Include only specified model configurations (by filename without extension)')
    parser.add_argument('--exclude', nargs='+', help='Exclude specified model config (by filename without extension)')
    parser.add_argument('--force-submit', action='store_true', default=False,
                        help='Force resubmission of batches, overriding submitted experiments tracking (batch runtime only)')
    parser.add_argument('--monitor-only', action='store_true', default=False,
                        help='Monitor existing batches without submitting new ones (batch runtime only)')
    parser.add_argument('--resubmit-failures', action='store_true', default=False,
                        help='Resubmit failed experiments (keeps failure records)')
    parser.add_argument('--experiment-count-limit', type=int, default=None,
                        help='Maximum number of experiments to run (for debugging)')
    return parser.parse_args()


async def main():
    args = parse_args()

    # Load all model configurations from YAML files
    model_configs = load_all_model_engine_params("config/models", include=args.include, exclude=args.exclude)

    hf_dataset = None
    if args.hf_dataset:
        if args.hf_dataset in DATASET_MAPPING:
            hf_dataset = DATASET_MAPPING[args.hf_dataset]
        else:
            hf_dataset = args.hf_dataset

    # Validate dataset/subset combination for HuggingFace datasets
    if hf_dataset and not args.local_dataset:
        validate_dataset_subset(hf_dataset, args.subset)

    # Create and run the evaluation runtime
    # noinspection PyUnreachableCode
    match args.runtime_type:
        case "screenshot":
            if args.local_dataset:
                print("Using ScreenshotRuntime (local dataset with screenshots)")
            else:
                print(
                    f"Using ScreenshotRuntime (HuggingFace dataset: {args.hf_dataset})"
                )

            runtime = ScreenshotRuntime(
                engine_params_list=model_configs,
                remote=args.remote,
                max_concurrent_per_engine=MAX_CONCURRENT_MODELS,
                experiment_count_limit=args.experiment_count_limit,
                experiment_label_filter=EXPERIMENT_LABEL_FILTER,
                debug_mode=args.debug,
                local_dataset_path=args.local_dataset,
                hf_dataset_name=None if args.local_dataset else hf_dataset,
                hf_subset=args.subset,
            )
            await runtime.run()
        case "batch":
            print("Using BatchEvaluationRuntime (for Batch processing APIs)")
            runtime = BatchOrchestratorRuntime(
                engine_params_list=model_configs,
                remote=args.remote,
                local_dataset_path=args.local_dataset,
                hf_dataset_name=None if args.local_dataset else hf_dataset,
                hf_subset=args.subset,
                experiment_count_limit=args.experiment_count_limit,
                debug_mode=args.debug,
                force_submit=args.force_submit,
                monitor_only=args.monitor_only,
                resubmit_failures=args.resubmit_failures,
            )
            runtime.run()
        case "headless":
            if not args.local_dataset:
                raise ValueError("HeadlessRuntime requires --local-dataset argument")
            print("Using HeadlessRuntime (headless product selection)")
            runtime = HeadlessRuntime(
                local_dataset_path=args.local_dataset,
                engine_params_list=model_configs,
                experiment_count_limit=args.experiment_count_limit,
                experiment_label_filter=EXPERIMENT_LABEL_FILTER,
                debug_mode=args.debug,
            )
            await runtime.run()
        case _:
            print("Using SimpleEvaluationRuntime (browser-based)")
            runtime = SimpleEvaluationRuntime(
                local_dataset_path=args.local_dataset,
                engine_params_list=model_configs,
                max_concurrent_models=MAX_CONCURRENT_MODELS,
                experiment_count_limit=args.experiment_count_limit,
                experiment_label_filter=EXPERIMENT_LABEL_FILTER,
                debug_mode=args.debug
            )
            await runtime.run()


if __name__ == "__main__":
    asyncio.run(main())
