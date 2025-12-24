from datetime import datetime
from pathlib import Path

import pandas as pd
import typer

from experiments.analysis.common import (
    load_query_shortnames,
    load_model_display_names,
    get_rationality_suite_experiment_names,
)
from experiments.analysis.market_share import analyze_market_share
from experiments.analysis.rationality import calculate_sanity_check

app = typer.Typer()
rationality_app = typer.Typer()
app.add_typer(rationality_app, name="rationality-suite", help="Rationality suite sanity checks")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Analysis CLI for experiment data."""
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())
        raise typer.Exit(0)


@app.command("market-share")
def market_share(csv_file: Path) -> None:
    """Analyze market share from a CSV file."""
    target_path = Path("artifacts/analysis")
    target_path.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    filename = f"{now:%Y%m%d%H%M%S}_market_share.csv"
    output_filepath = target_path / filename

    short_titles = load_query_shortnames()

    analyze_market_share(csv_file, short_titles=short_titles, output_filepath=output_filepath)


@rationality_app.command("price")
def rationality_price(csv_file: Path) -> None:
    """Run price sanity checks."""
    experiment_names = get_rationality_suite_experiment_names()
    model_display_names = load_model_display_names()

    df = pd.read_csv(csv_file)
    results = calculate_sanity_check(df, "price", experiment_names, model_display_names)

    target_path = Path("artifacts/analysis")
    target_path.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.now():%Y%m%d%H%M%S}_price_sanity_check.csv"
    output_filepath = target_path / filename
    results.to_csv(output_filepath)
    print(f"Saved to {output_filepath}")


@rationality_app.command("rating")
def rationality_rating(csv_file: Path) -> None:
    """Run rating sanity checks."""
    experiment_names = get_rationality_suite_experiment_names()
    model_display_names = load_model_display_names()

    df = pd.read_csv(csv_file)
    results = calculate_sanity_check(df, "rating", experiment_names, model_display_names)

    target_path = Path("artifacts/analysis")
    target_path.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.now():%Y%m%d%H%M%S}_rating_sanity_check.csv"
    output_filepath = target_path / filename
    results.to_csv(output_filepath)
    print(f"Saved to {output_filepath}")


@rationality_app.command("instruction")
def rationality_instruction(csv_file: Path) -> None:
    """Run instruction sanity checks."""
    experiment_names = get_rationality_suite_experiment_names()
    model_display_names = load_model_display_names()

    df = pd.read_csv(csv_file)
    results = calculate_sanity_check(df, "instruction", experiment_names, model_display_names)

    target_path = Path("artifacts/analysis")
    target_path.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.now():%Y%m%d%H%M%S}_instruction_sanity_check.csv"
    output_filepath = target_path / filename
    results.to_csv(output_filepath)
    print(f"Saved to {output_filepath}")


if __name__ == "__main__":
    app()
