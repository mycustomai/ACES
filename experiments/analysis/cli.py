from datetime import datetime
from pathlib import Path

import typer
from experiments.analysis.common import load_query_shortnames

from experiments.analysis.market_share import analyze_market_share

app = typer.Typer()


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


if __name__ == "__main__":
    app()
