import typer
from rich.console import Console
from rich.table import Table

from cryptocast import __version__
from cryptocast.config import get_settings
from cryptocast.features.pipeline import FeaturePipeline
from cryptocast.ingest.binance import BinanceIngester
from cryptocast.ingest.gnews import GNewsIngester

app = typer.Typer(
    name="cryptocast",
    help="Cryptocast: Automated Crypto Forecasting Platform CLI",
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Print package version and active configuration."""
    settings = get_settings()
    console.print(f"[bold green]Cryptocast[/bold green] version: {__version__}")
    console.print(f"Storage backend: [cyan]{settings.storage_backend}[/cyan]")
    console.print(f"Data directory: [cyan]{settings.local_storage_dir}[/cyan]")


@app.command()
def info() -> None:
    """Show configured assets, horizons, and settings."""
    settings = get_settings()
    assets_cfg = settings.get_assets_config()

    table = Table(title="Configured Crypto Assets")
    table.add_column("Symbol", style="cyan", no_wrap=True)
    table.add_column("Base", style="magenta")
    table.add_column("Quote", style="green")
    table.add_column("Keywords", style="yellow")
    table.add_column("Benchmark", style="bold red")

    for symbol, cfg in assets_cfg.assets.items():
        table.add_row(
            symbol,
            cfg.base_asset,
            cfg.quote_asset,
            ", ".join(cfg.news_keywords),
            "Yes" if cfg.is_benchmark else "No",
        )
    console.print(table)
    console.print(f"Interval: [bold]{assets_cfg.interval}[/bold]")
    console.print(
        f"Horizons: [bold]{assets_cfg.horizons}[/bold] bars (Primary: {assets_cfg.primary_horizon})"
    )


@app.command()
def ingest(
    symbol: str | None = typer.Option(None, help="Specific symbol to ingest, e.g. BTCUSDT"),
    include_news: bool = typer.Option(True, help="Whether to also ingest news"),
    lookback_bars: int = typer.Option(48, help="Number of recent bars to fetch"),
) -> None:
    """Run data ingestion for klines and news."""
    settings = get_settings()
    assets_cfg = settings.get_assets_config()
    symbols_to_ingest = [symbol] if symbol else list(assets_cfg.assets.keys())

    console.print(
        f"[bold blue]Running ingestion...[/bold blue] (symbols={symbols_to_ingest}, news={include_news})"
    )

    binance_ingester = BinanceIngester()
    for s in symbols_to_ingest:
        try:
            df = binance_ingester.ingest_recent(
                symbol=s,
                interval=assets_cfg.interval,
                lookback_bars=lookback_bars,
                validate=True,
            )
            console.print(f"  [green]✓[/green] Ingested {len(df)} klines for {s}")
        except Exception as e:
            console.print(f"  [red]✗[/red] Failed to ingest {s}: {e}")

    if include_news:
        news_ingester = GNewsIngester()
        for s in symbols_to_ingest:
            asset_info = assets_cfg.assets.get(s)
            if asset_info and asset_info.news_keywords:
                news_df = news_ingester.ingest_asset_news(
                    asset=asset_info.base_asset,
                    keywords=asset_info.news_keywords,
                )
                console.print(
                    f"  [green]✓[/green] Ingested {len(news_df)} news articles for {asset_info.base_asset}"
                )


@app.command()
def features(
    symbol: str | None = typer.Option(None, help="Asset symbol, e.g. BTCUSDT"),
    horizon: int = typer.Option(4, help="Prediction horizon in bars (default: 4)"),
) -> None:
    """Compute and update feature store (Gold layer)."""
    settings = get_settings()
    assets_cfg = settings.get_assets_config()
    symbols = [symbol] if symbol else list(assets_cfg.assets.keys())

    pipeline = FeaturePipeline()
    for s in symbols:
        console.print(
            f"[bold blue]Building Gold features for {s} (horizon={horizon})...[/bold blue]"
        )
        try:
            df = pipeline.build_and_save_features(symbol=s, horizon=horizon)
            if df.empty:
                console.print(
                    f"  [yellow]![/yellow] No silver data found for {s}. Run 'cryptocast ingest' first."
                )
            else:
                data_hash = df.attrs.get("data_hash", "unknown")
                console.print(
                    f"  [green]✓[/green] Saved {len(df)} feature rows ({len(df.columns)} columns, hash={data_hash})"
                )
        except Exception as e:
            console.print(f"  [red]✗[/red] Failed to build features for {s}: {e}")


@app.command()
def train(
    symbol: str = typer.Option("BTCUSDT", help="Asset symbol"),
    model_type: str = typer.Option("xgboost", help="Model family: xgboost | random_forest"),
    tune: bool = typer.Option(False, help="Run Optuna hyperparameter tuning"),
) -> None:
    """Train direction classifier and return regressor models."""
    console.print(f"[bold blue]Training {model_type} for {symbol} (tuning={tune})...[/bold blue]")
    # Will be connected to training pipeline in M4/M6
    console.print("[green]Model training stub executed.[/green]")


@app.command()
def evaluate(
    challenger: str = typer.Option("latest", help="Challenger model version"),
    champion: str = typer.Option("champion", help="Champion model version"),
) -> None:
    """Evaluate challenger model against champion and baselines."""
    console.print(f"[bold blue]Evaluating {challenger} vs {champion}...[/bold blue]")
    # Will be connected to promotion gate in M6
    console.print("[green]Evaluation stub executed.[/green]")


@app.command()
def predict(
    symbol: str | None = typer.Option(None, help="Asset symbol"),
) -> None:
    """Generate latest direction probability and expected return predictions."""
    console.print(f"[bold blue]Generating predictions for {symbol or 'all assets'}...[/bold blue]")
    # Will be connected to prediction pipeline in M5
    console.print("[green]Prediction stub executed.[/green]")


@app.command()
def backfill(
    start_date: str = typer.Option(..., help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(..., help="End date (YYYY-MM-DD)"),
    symbols: str | None = typer.Option(None, help="Comma-separated symbols"),
) -> None:
    """Backfill historical market data and compute features."""
    console.print(
        f"[bold blue]Backfilling from {start_date} to {end_date} for {symbols or 'all'}...[/bold blue]"
    )
    # Will be connected to backfill flow in M2
    console.print("[green]Backfill task stub completed.[/green]")


@app.command()
def monitor() -> None:
    """Run data quality, drift, and model performance checks."""
    console.print(
        "[bold blue]Running monitor checks (freshness, drift, DQ, live accuracy)...[/bold blue]"
    )
    # Will be connected to monitor flow in M7
    console.print("[green]Monitor stub completed.[/green]")


if __name__ == "__main__":
    app()
