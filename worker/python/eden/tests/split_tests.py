from pathlib import Path

# Split tests into individual files by re-exporting from the combined suite if present.
base = Path(__file__).resolve().parent
combined = base / "test_suite.py"
if combined.exists():
    content = combined.read_text()
    # Simple splits
    (base / "test_data_loader.py").write_text(
        "\n".join(
            [
                line
                for line in content.splitlines()
                if "test_sample_csv_load_and_columns" in line
                or "test_yfinance_fetch_mock_fallback" in line
                or "import" in line
                or "Path" in line
            ]
        )
    )
    (base / "test_indicators.py").write_text(
        "\n".join(
            [
                line
                for line in content.splitlines()
                if "test_indicators_on_synthetic" in line or "import" in line
            ]
        )
    )
    (base / "test_backtest_engine.py").write_text(
        "\n".join(
            [
                line
                for line in content.splitlines()
                if "test_backtest_engine_basic_flow" in line
                or "import" in line
                or "BacktestEngine" in line
            ]
        )
    )
    (base / "test_strategy_signals.py").write_text(
        "\n".join(
            [
                line
                for line in content.splitlines()
                if "test_strategy_signals_generation" in line or "import" in line
            ]
        )
    )
    (base / "test_integration_backtest.py").write_text(
        "\n".join(
            [
                line
                for line in content.splitlines()
                if "test_integration_backtest_end_to_end" in line or "import" in line
            ]
        )
    )
