.PHONY: bootstrap lint format test smoke run-hourly run-retrain clean

bootstrap:
	uv sync --extra dev

lint:
	uv run ruff check src tests
	uv run mypy src

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

test:
	uv run pytest tests

smoke:
	uv run python scripts/m0_binance_spike.py
	uv run python -m cryptocast.cli --help

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build *.egg-info
