from unittest.mock import patch

import pandas as pd
from typer.testing import CliRunner

from cryptocast import __version__
from cryptocast.cli import app

runner = CliRunner()


def test_cli_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout
    assert "Cryptocast" in result.stdout


def test_cli_info():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "BTCUSDT" in result.stdout
    assert "ETHUSDT" in result.stdout
    assert "SOLUSDT" in result.stdout


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.stdout
    assert "features" in result.stdout
    assert "train" in result.stdout
    assert "predict" in result.stdout


@patch("cryptocast.ingest.binance.BinanceIngester.ingest_recent", return_value=pd.DataFrame())
@patch("cryptocast.ingest.gnews.GNewsIngester.ingest_asset_news", return_value=pd.DataFrame())
def test_cli_commands(mock_news, mock_binance):
    for cmd in ["ingest", "features", "predict", "monitor"]:
        res = runner.invoke(app, [cmd])
        assert res.exit_code == 0
