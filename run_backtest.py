"""AlphaForge — Main backtest entry point.

Usage:
    python run_backtest.py --config strategies/examples/momentum_example.yaml --days 90
    python run_backtest.py --config strategies/examples/momentum_example.yaml --walk-forward --days 120
"""

import argparse
from src.strategy import load_strategy_config, compute_composite_signal, generate_positions
from src.backtest import run_backtest, walk_forward, BacktestConfig
from src.data_fetcher import fetch_ohlcv


def main():
    """Run backtest from CLI."""
    parser = argparse.ArgumentParser(description="AlphaForge Backtest")
    parser.add_argument("--config", required=True, help="Strategy YAML config path")
    parser.add_argument("--days", type=int, default=90, help="Backtest lookback days")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation")
    parser.add_argument("--symbol", default="BTC-USDT-SWAP", help="Instrument")
    args = parser.parse_args()

    strategy = load_strategy_config(args.config)
    print(f"Strategy: {strategy.get('name', 'unnamed')}")
    print(f"Symbol: {args.symbol}")
    print(f"Days: {args.days}")

    # Fetch data
    df = fetch_ohlcv(args.symbol, days=args.days)
    print(f"Data: {len(df)} bars loaded")

    # TODO: compute your factors here
    # factors_df = your_factor_function(df)
    # positions = generate_positions(compute_composite_signal(factors_df, strategy["factors"]))
    # result = run_backtest(df, positions)

    print("\nTo complete this pipeline:")
    print("1. Implement your factors in src/factor_pipeline/factors/")
    print("2. Register them with @register_factor decorator")
    print("3. Configure factor list in your strategy YAML")
    print("4. Uncomment the backtest logic above")


if __name__ == "__main__":
    main()
