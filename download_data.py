#!/usr/bin/env python3
"""Script to download OHLCV data for multiple symbols and timeframes from MT5."""

import argparse
import MetaTrader5 as mt5
from forex_lstm.data import download_multi_symbol_timeframe, initialize_mt5

# Define symbols and timeframes
DEFAULT_SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF']
TIMEFRAMES = {
    'M15': mt5.TIMEFRAME_M15,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
}


def main():
    parser = argparse.ArgumentParser(
        description="Download MT5 data for multiple symbols and timeframes")
    parser.add_argument("--symbols", nargs='+', default=DEFAULT_SYMBOLS,
                        help="List of symbols to download (default: major USD pairs)")
    parser.add_argument("--timeframes", nargs='+', choices=TIMEFRAMES.keys(), default=['M15', 'H1', 'H4'],
                        help="Timeframes to download (default: M15, H1, H4)")
    parser.add_argument("--bars", type=int, default=10000,
                        help="Number of bars to download per symbol/timeframe (default: 10000)")
    parser.add_argument("--data-dir", type=str, default="DATA",
                        help="Directory to save data (default: DATA)")

    args = parser.parse_args()

    # Filter timeframes dict to only selected ones
    selected_timeframes = {tf: TIMEFRAMES[tf] for tf in args.timeframes}

    print(f"Initializing MT5...")
    initialize_mt5()

    print(f"Downloading data for symbols: {args.symbols}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Bars per download: {args.bars}")
    print(f"Save directory: {args.data_dir}")

    download_multi_symbol_timeframe(
        args.symbols, selected_timeframes, args.bars, args.data_dir)

    print("\nData download complete!")
    print(f"Check the {args.data_dir} folder for CSV files.")


if __name__ == "__main__":
    main()
