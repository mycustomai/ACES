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

    from experiments.visualization.sanity_checks import (
        load_sanity_check_data,
        plot_single_task_by_provider,
        plot_category_combined,
    )

    output_dir = Path("artifacts/visualization/sanity_checks")
    _, tests, data = load_sanity_check_data(csv_file)

    # Individual plots
    tasks = [
        (0, "rating_plus_0.1", "Rating +0.1 sanity check failure rates by provider", "+0.1 Rating"),
        (1, "rating_random_low_variance", "Rating random (low variance) failure rates by provider", "Low Variance"),
        (2, "rating_random_high_variance", "Rating random (high variance) failure rates by provider", "High Variance"),
    ]

    for task_idx, filename, title, task_label in tasks:
        plot_single_task_by_provider(
            data, task_idx, output_dir / f"{filename}.png",
            title=title, task_label=task_label
        )

    # Combined plot
    plot_category_combined(
        data,
        ["+0.1 Rating", "Low Variance", "High Variance"],
        output_dir / "rating_combined.png",
        title="Rating sanity checks combined (by release date)",
    )

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

    from experiments.visualization.sanity_checks import (
        load_combined_price_data,
        plot_single_task_by_provider,
        plot_category_combined,
    )

    output_dir = Path("artifacts/visualization/sanity_checks")
    _, tests, data = load_combined_price_data(price_csv, ar_price_csv)

    # Individual plots
    tasks = [
        (0, "price_random_low_variance", "Price random (low variance) failure rates by provider", "Random (Low Var)"),
        (1, "price_random_high_variance", "Price random (high variance) failure rates by provider", "Random (High Var)"),
        (2, "price_1_percent", "Price 1% reduction failure rates by provider", "1% Reduction"),
        (3, "price_5_percent", "Price 5% reduction failure rates by provider", "5% Reduction"),
        (4, "price_10_percent", "Price 10% reduction failure rates by provider", "10% Reduction"),
    ]

    for task_idx, filename, title, task_label in tasks:
        plot_single_task_by_provider(
            data, task_idx, output_dir / f"{filename}.png",
            title=title, task_label=task_label
        )

    # Combined plot
    plot_category_combined(
        data,
        ["Random (Low Var)", "Random (High Var)", "1% Reduction", "5% Reduction", "10% Reduction"],
        output_dir / "price_combined.png",
        title="Price sanity checks combined (by release date)",
    )

    typer.echo(f"Price sanity check plots generated in {output_dir}/")


@sanity_checks_app.command("instruction")
def sanity_checks_instruction(csv_file: Path) -> None:
    """Generate instruction following sanity check visualizations."""
    if not csv_file.exists():
        typer.echo(f"Error: File not found: {csv_file}", err=True)
        raise typer.Exit(1)

    from experiments.visualization.sanity_checks import (
        load_sanity_check_data,
        plot_single_task_by_provider,
        plot_category_combined,
    )

    output_dir = Path("artifacts/visualization/sanity_checks")
    _, tests, data = load_sanity_check_data(csv_file)

    # Individual plots
    tasks = [
        (0, "instruction_budget", "Instruction following (budget) failure rates by provider", "Budget"),
        (1, "instruction_color", "Instruction following (color) failure rates by provider", "Color"),
        (2, "instruction_brand", "Instruction following (brand) failure rates by provider", "Brand"),
    ]

    for task_idx, filename, title, task_label in tasks:
        plot_single_task_by_provider(
            data, task_idx, output_dir / f"{filename}.png",
            title=title, task_label=task_label
        )

    # Combined plot
    plot_category_combined(
        data,
        ["Budget", "Color", "Brand"],
        output_dir / "instruction_combined.png",
        title="Instruction following combined (by release date)",
    )

    typer.echo(f"Instruction sanity check plots generated in {output_dir}/")


@sanity_checks_app.command("all")
def sanity_checks_all(
    rating_csv: Path,
    price_csv: Path,
    ar_price_csv: Path,
    instruction_csv: Path,
) -> None:
    """Generate all sanity check visualizations."""
    from typer import Context

    ctx = Context(sanity_checks_app)

    typer.echo("Generating rating plots...")
    ctx.invoke(sanity_checks_rating, csv_file=rating_csv)

    typer.echo("\nGenerating price plots...")
    ctx.invoke(sanity_checks_price, price_csv=price_csv, ar_price_csv=ar_price_csv)

    typer.echo("\nGenerating instruction plots...")
    ctx.invoke(sanity_checks_instruction, csv_file=instruction_csv)

    typer.echo("\nAll sanity check plots generated!")


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
    """Generate rating feature impact visualizations only."""
    if not csv_file.exists():
        typer.echo(f"Error: File not found: {csv_file}", err=True)
        raise typer.Exit(1)

    from experiments.visualization.feature_impact import plot_rating_impact_by_provider

    output_dir = Path("artifacts/visualization/feature_impact")
    plot_rating_impact_by_provider(csv_file, output_dir)
    typer.echo(f"Rating impact plot generated in {output_dir}/")


@feature_impact_app.command("price")
def feature_impact_price(csv_file: Path) -> None:
    """Generate price feature impact visualizations only."""
    if not csv_file.exists():
        typer.echo(f"Error: File not found: {csv_file}", err=True)
        raise typer.Exit(1)

    from experiments.visualization.feature_impact import plot_price_impact_by_provider

    output_dir = Path("artifacts/visualization/feature_impact")
    plot_price_impact_by_provider(csv_file, output_dir)
    typer.echo(f"Price impact plot generated in {output_dir}/")


@feature_impact_app.command("tags")
def feature_impact_tags(csv_file: Path) -> None:
    """Generate tags feature impact visualizations only."""
    if not csv_file.exists():
        typer.echo(f"Error: File not found: {csv_file}", err=True)
        raise typer.Exit(1)

    from experiments.visualization.feature_impact import plot_tags_impact_by_provider

    output_dir = Path("artifacts/visualization/feature_impact")
    plot_tags_impact_by_provider(csv_file, output_dir)
    typer.echo(f"Tags impact plot generated in {output_dir}/")


if __name__ == "__main__":
    app()
