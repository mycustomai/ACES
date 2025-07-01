import asyncio
import os
import argparse

from dotenv import load_dotenv

from experiments.engine_loader import load_all_model_engine_params
from experiments.runners import (
    BatchEvaluationRuntime,
    HFHubDatasetRuntime,
    LocalDatasetRuntime,
    SimpleEvaluationRuntime,
)

load_dotenv()

# Configuration for parallel execution
MAX_CONCURRENT_MODELS = int(os.getenv("MAX_CONCURRENT_MODELS", "20"))
EXPERIMENT_COUNT_LIMIT = None if os.getenv("EXPERIMENT_COUNT_LIMIT") is None else int(os.getenv("EXPERIMENT_COUNT_LIMIT"))
EXPERIMENT_LABEL_FILTER = None
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
RUNTIME_TYPE = os.getenv("RUNTIME_TYPE", "screenshot").lower()
DEFAULT_HF_DATASET = "My-Custom-AI/ACERS-v1"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AI Agent Impact Evaluation')
    parser.add_argument('--runtime-type', choices=['simple', 'screenshot', 'batch'], 
                       default=RUNTIME_TYPE, help='Runtime type to use')
    parser.add_argument('--local-dataset', type=str,
                       help='Local dataset path')
    parser.add_argument('--hf-dataset', type=str, default=DEFAULT_HF_DATASET,
                        help=f"Hugging Face dataset name to use (e.g., \"{DEFAULT_HF_DATASET}\")")
    parser.add_argument('--subset', type=str,
                        help="Subset of ACERS dataset to use (e.g., 'sanity_checks', 'bias_experiments', 'all'). View dataset card for more details.",
                        default="all")
    parser.add_argument('--debug', action='store_true', default=DEBUG_MODE,
                       help='Enable debug mode')
    parser.add_argument('--remote', action='store_true', default=False,
                       help='Use remote GCS URLs instead of local screenshot bytes (screenshot runtime only)')
    parser.add_argument('--include', nargs='+', help='Include only specified model configurations (by filename without extension)')
    parser.add_argument('--exclude', nargs='+', help='Exclude specified model config (by filename without extension)')
    parser.add_argument('--force-submit', action='store_true', default=False,
                       help='Force resubmission of batches, overriding submitted experiments tracking (batch runtime only)')
    return parser.parse_args()


async def main():
    args = parse_args()
    
    # Load all model configurations from YAML files
    model_configs = load_all_model_engine_params("config/models", include=args.include, exclude=args.exclude)
    
    # Create and run the evaluation runtime
    # noinspection PyUnreachableCode
    match args.runtime_type:
        case "screenshot":
            if args.local_dataset:
                print("Using LocalDatasetRuntime (local dataset with screenshots)")
                runtime = LocalDatasetRuntime(
                    local_dataset_path=args.local_dataset,
                    engine_params_list=model_configs,
                    max_concurrent_per_engine=MAX_CONCURRENT_MODELS,
                    experiment_count_limit=EXPERIMENT_COUNT_LIMIT,
                    experiment_label_filter=EXPERIMENT_LABEL_FILTER,
                    debug_mode=args.debug,
                    remote=args.remote
                )
            else:
                print(f"Using HFHubDatasetRuntime (HuggingFace dataset: {args.hf_dataset})")
                runtime = HFHubDatasetRuntime(
                    hf_dataset_name=args.hf_dataset,
                    subset=args.subset,
                    engine_params_list=model_configs,
                    max_concurrent_per_engine=MAX_CONCURRENT_MODELS,
                    experiment_count_limit=EXPERIMENT_COUNT_LIMIT,
                    experiment_label_filter=EXPERIMENT_LABEL_FILTER,
                    debug_mode=args.debug
                )
            await runtime.run()
        case "batch":
            print("Using BatchEvaluationRuntime (for Batch processing APIs)")
            runtime = BatchEvaluationRuntime(
                local_dataset_path=args.local_dataset,
                engine_params_list=model_configs,
                experiment_count_limit=EXPERIMENT_COUNT_LIMIT,
                debug_mode=args.debug,
                force_submit=args.force_submit,
            )
            await runtime.run()
        case _:
            print("Using SimpleEvaluationRuntime (browser-based)")
            runtime = SimpleEvaluationRuntime(
                local_dataset_path=args.dataset,
                engine_params_list=model_configs,
                max_concurrent_models=MAX_CONCURRENT_MODELS,
                experiment_count_limit=EXPERIMENT_COUNT_LIMIT,
                experiment_label_filter=EXPERIMENT_LABEL_FILTER,
                debug_mode=args.debug
            )
            await runtime.run()


if __name__ == "__main__":
    asyncio.run(main())
