from datetime import datetime
from pathlib import Path

import pandas as pd
import typer
from experiments.analysis.choice_model import generate_choice_model_results

from experiments.analysis.common import (
    load_query_shortnames,
    load_model_display_names,
    get_rationality_suite_experiment_names,
    SanityCheckMode,
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


@app.command("choice-model")
def choice_model(csv_file: Path) -> None:
    """Generate a choice-model from a CSV file."""
    target_path = Path("artifacts/analysis")
    target_path.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    filename = f"{now:%Y%m%d%H%M%S}_choice_model.csv"
    output_filepath = target_path / filename

    #short_titles = load_query_shortnames()

    generate_choice_model_results(csv_file, output_filepath=output_filepath)


def _run_sanity_check(csv_file: Path, mode: SanityCheckMode) -> None:
    experiment_names = get_rationality_suite_experiment_names()
    model_display_names = load_model_display_names()

    df = pd.read_csv(csv_file)
    results = calculate_sanity_check(df, mode, experiment_names, model_display_names)

    target_path = Path("artifacts/analysis")
    target_path.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.now():%Y%m%d%H%M%S}_{mode}_sanity_check.csv"
    output_filepath = target_path / filename
    results.to_csv(output_filepath)
    print(f"Saved to {output_filepath}")


def _make_sanity_check_command(mode: SanityCheckMode):
    def command(csv_file: Path) -> None:
        _run_sanity_check(csv_file, mode)
    command.__doc__ = f"Run {mode} sanity checks."
    return command


for _mode in SanityCheckMode:
    rationality_app.command(_mode.value)(_make_sanity_check_command(_mode))


if __name__ == "__main__":
    app()
