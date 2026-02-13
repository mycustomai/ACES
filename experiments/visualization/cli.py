from pathlib import Path

import typer

app = typer.Typer()
sanity_checks_app = typer.Typer()
feature_impact_app = typer.Typer()
app.add_typer(sanity_checks_app, name="sanity-checks", help="Generate sanity check visualizations")
app.add_typer(feature_impact_app, name="feature-impact", help="Generate feature impact visualizations")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Visualization CLI for generating plots from analysis results."""
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())
        raise typer.Exit(0)


@app.command("position-bias")
def position_bias(csv_file: Path) -> None:
    """Generate position bias visualizations from choice model CSV."""
    if not csv_file.exists():
        typer.echo(f"Error: File not found: {csv_file}", err=True)
        raise typer.Exit(1)

    from experiments.visualization.choice_model_position_bias import plot_all_providers, plot_by_provider

    output_dir = Path("artifacts/visualization/position_bias")
    plot_all_providers(csv_file, output_dir / "all_providers.png")
    plot_by_provider(csv_file, output_dir / "by_provider.png")
    typer.echo(f"Position bias plots generated in {output_dir}/")


@app.command("heatmap")
def heatmap(csv_file: Path) -> None:
    """Generate position probability heatmaps from choice model CSV."""
    if not csv_file.exists():
        typer.echo(f"Error: File not found: {csv_file}", err=True)
        raise typer.Exit(1)

    from experiments.visualization.heatmaps import generate_all_heatmaps

    output_dir = Path("artifacts/visualization/heatmaps")
    generate_all_heatmaps(csv_file, output_dir)
    typer.echo(f"Heatmap plots generated in {output_dir}/")


@sanity_checks_app.callback(invoke_without_command=True)
def sanity_checks_callback(ctx: typer.Context) -> None:
    """Generate sanity check visualizations."""
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())
        raise typer.Exit(0)


@sanity_checks_app.command("rating")
def sanity_checks_rating(csv_file: Path) -> None:
    """Generate rating sanity check visualizations."""
    if not csv_file.exists():
        typer.echo(f"Error: File not found: {csv_file}", err=True)
        raise typer.Exit(1)

    from experiments.visualization.sanity_checks import generate_rating_plots

    output_dir = Path("artifacts/visualization/sanity_checks")
    generate_rating_plots(csv_file, output_dir)
    typer.echo(f"Rating sanity check plots generated in {output_dir}/")


@sanity_checks_app.command("price")
def sanity_checks_price(
    price_csv: Path,
    ar_price_csv: Path,
) -> None:
    """Generate price sanity check visualizations (combines price and ar_price data)."""
    if not price_csv.exists():
        typer.echo(f"Error: File not found: {price_csv}", err=True)
        raise typer.Exit(1)
    if not ar_price_csv.exists():
        typer.echo(f"Error: File not found: {ar_price_csv}", err=True)
        raise typer.Exit(1)

    from experiments.visualization.sanity_checks import generate_price_plots

    output_dir = Path("artifacts/visualization/sanity_checks")
    generate_price_plots(price_csv, ar_price_csv, output_dir)
    typer.echo(f"Price sanity check plots generated in {output_dir}/")


@sanity_checks_app.command("instruction")
def sanity_checks_instruction(csv_file: Path) -> None:
    """Generate instruction following sanity check visualizations."""
    if not csv_file.exists():
        typer.echo(f"Error: File not found: {csv_file}", err=True)
        raise typer.Exit(1)

    from experiments.visualization.sanity_checks import generate_instruction_plots

    output_dir = Path("artifacts/visualization/sanity_checks")
    generate_instruction_plots(csv_file, output_dir)
    typer.echo(f"Instruction sanity check plots generated in {output_dir}/")


@sanity_checks_app.command("all")
def sanity_checks_all(
    rating_csv: Path,
    price_csv: Path,
    ar_price_csv: Path,
    instruction_csv: Path,
) -> None:
    """Generate all sanity check visualizations."""
    if not rating_csv.exists():
        typer.echo(f"Error: File not found: {rating_csv}", err=True)
        raise typer.Exit(1)
    if not price_csv.exists():
        typer.echo(f"Error: File not found: {price_csv}", err=True)
        raise typer.Exit(1)
    if not ar_price_csv.exists():
        typer.echo(f"Error: File not found: {ar_price_csv}", err=True)
        raise typer.Exit(1)
    if not instruction_csv.exists():
        typer.echo(f"Error: File not found: {instruction_csv}", err=True)
        raise typer.Exit(1)

    from experiments.visualization.sanity_checks import generate_all_sanity_check_plots

    output_dir = Path("artifacts/visualization/sanity_checks")
    typer.echo("Generating all sanity check plots...")
    generate_all_sanity_check_plots(rating_csv, price_csv, ar_price_csv, instruction_csv, output_dir)
    typer.echo(f"\nAll sanity check plots generated in {output_dir}/")


@feature_impact_app.callback(invoke_without_command=True)
def feature_impact_callback(ctx: typer.Context) -> None:
    """Generate feature impact visualizations."""
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())
        raise typer.Exit(0)


@feature_impact_app.command("all")
def feature_impact_all(csv_file: Path) -> None:
    """Generate all feature impact visualizations."""
    if not csv_file.exists():
        typer.echo(f"Error: File not found: {csv_file}", err=True)
        raise typer.Exit(1)

    from experiments.visualization.feature_impact import generate_all_feature_impact_plots

    output_dir = Path("artifacts/visualization/feature_impact")
    generate_all_feature_impact_plots(csv_file, output_dir)
    typer.echo(f"Feature impact plots generated in {output_dir}/")


@feature_impact_app.command("rating")
def feature_impact_rating(csv_file: Path) -> None:
    """Generate rating feature impact visualizations (all_providers + by_provider)."""
    if not csv_file.exists():
        typer.echo(f"Error: File not found: {csv_file}", err=True)
        raise typer.Exit(1)

    from experiments.visualization.feature_impact import plot_rating_impact_by_provider

    output_dir = Path("artifacts/visualization/feature_impact")
    plot_rating_impact_by_provider(csv_file, output_dir)
    typer.echo(f"Rating impact plots generated in {output_dir}/ (all_providers + by_provider)")


@feature_impact_app.command("price")
def feature_impact_price(csv_file: Path) -> None:
    """Generate price feature impact visualizations (all_providers + by_provider)."""
    if not csv_file.exists():
        typer.echo(f"Error: File not found: {csv_file}", err=True)
        raise typer.Exit(1)

    from experiments.visualization.feature_impact import plot_price_impact_by_provider

    output_dir = Path("artifacts/visualization/feature_impact")
    plot_price_impact_by_provider(csv_file, output_dir)
    typer.echo(f"Price impact plots generated in {output_dir}/ (all_providers + by_provider)")


@feature_impact_app.command("tags")
def feature_impact_tags(csv_file: Path) -> None:
    """Generate tags feature impact visualizations (all_providers + by_provider for both tags)."""
    if not csv_file.exists():
        typer.echo(f"Error: File not found: {csv_file}", err=True)
        raise typer.Exit(1)

    from experiments.visualization.feature_impact import plot_tags_impact_by_provider

    output_dir = Path("artifacts/visualization/feature_impact")
    plot_tags_impact_by_provider(csv_file, output_dir)
    typer.echo(f"Tags impact plots generated in {output_dir}/ (4 plots: all_providers + by_provider for sponsored tag and overall pick)")


if __name__ == "__main__":
    app()
