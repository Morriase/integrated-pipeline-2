#!/usr/bin/env python3
"""Script to consolidate multi-symbol, multi-timeframe MT5 data into a single dataset for modeling."""

import os
import pandas as pd
import numpy as np
from pathlib import Path


def load_all_data(data_dir: str = "DATA") -> pd.DataFrame:
    """Load all CSV files from data directory and consolidate."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")

    all_data = []

    # Find all CSV files
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"Found {len(csv_files)} CSV files")

    for csv_file in csv_files:
        # Parse filename: SYMBOL_TIMEFRAME_BARSbars.csv
        filename = csv_file.stem  # Remove .csv
        parts = filename.split('_')
        if len(parts) >= 3:
            symbol = parts[0]
            timeframe = parts[1]
            bars = parts[2] if len(parts) > 2 else 'unknown'
        else:
            symbol = 'unknown'
            timeframe = 'unknown'
            bars = 'unknown'

        print(
            f"Loading {csv_file.name} (symbol: {symbol}, timeframe: {timeframe})")

        try:
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            df['bars'] = bars
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue

    if not all_data:
        raise ValueError("No data could be loaded")

    # Concatenate all data
    consolidated = pd.concat(all_data, ignore_index=False)

    # Sort by timestamp
    consolidated = consolidated.sort_index()

    return consolidated


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the consolidated data."""
    print(f"Initial data shape: {df.shape}")

    # Remove duplicates
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape}")

    # Handle missing values
    df = df.dropna()
    print(f"After dropping NaN: {df.shape}")

    # Ensure OHLCV columns are numeric
    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in ohlcv_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows with NaN after conversion
    df = df.dropna()
    print(f"After numeric conversion: {df.shape}")

    # Add derived features
    df['returns'] = df.groupby(['symbol', 'timeframe'])['Close'].pct_change()
    df['log_returns'] = np.log(
        df['Close'] / df.groupby(['symbol', 'timeframe'])['Close'].shift(1))

    # Add time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    return df


def create_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """Create wide format with features for each symbol/timeframe combination."""
    # Pivot to wide format
    # This will create columns like: EURUSD_H1_Open, EURUSD_H1_Close, etc.

    # First, create unique column names
    df['col_prefix'] = df['symbol'] + '_' + df['timeframe'] + '_'

    # List of base columns to pivot
    base_cols = ['Open', 'High', 'Low', 'Close',
                 'Volume', 'returns', 'log_returns']

    wide_dfs = []

    for col in base_cols:
        if col in df.columns:
            pivot_df = df.pivot_table(
                values=col,
                index=df.index,
                columns='col_prefix',
                aggfunc='first'
            )
            pivot_df.columns = [
                f"{prefix}{col}" for prefix in pivot_df.columns]
            wide_dfs.append(pivot_df)

    # Merge all pivoted dataframes
    wide_df = pd.concat(wide_dfs, axis=1)

    # Forward fill missing values (for alignment)
    wide_df = wide_df.fillna(method='ffill').dropna()

    return wide_df


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Consolidate MT5 data into single dataset")
    parser.add_argument("--data-dir", type=str, default="DATA",
                        help="Directory containing CSV files")
    parser.add_argument("--output", type=str, default="consolidated_dataset.csv",
                        help="Output CSV file path")
    parser.add_argument("--wide-format", action="store_true",
                        help="Create wide format with features for each symbol/timeframe")

    args = parser.parse_args()

    print("Loading all data...")
    df = load_all_data(args.data_dir)

    print("Cleaning data...")
    df = clean_data(df)

    if args.wide_format:
        print("Creating wide format...")
        df = create_wide_format(df)
        print(f"Wide format shape: {df.shape}")
        print(f"Columns: {list(df.columns[:10])}...")  # Show first 10 columns

    print(f"Saving consolidated dataset to {args.output}...")
    df.to_csv(args.output)
    print(f"Dataset saved! Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Print summary
    if 'symbol' in df.columns:
        symbol_counts = df.groupby('symbol').size()
        print(f"Symbols: {dict(symbol_counts)}")

    if 'timeframe' in df.columns:
        timeframe_counts = df.groupby('timeframe').size()
        print(f"Timeframes: {dict(timeframe_counts)}")


if __name__ == "__main__":
    main()
