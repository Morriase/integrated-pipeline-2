#!/usr/bin/env python3
"""
Complete SMC Feature Engineering Script for LSTM Training

This script processes consolidated forex data and generates comprehensive SMC features
for training LSTM models. It computes Order Blocks, Fair Value Gaps, Break of Structure,
and other SMC indicators, then generates trading labels.

Output: A complete CSV dataset with all features and labels for LSTM training.
"""

from forex_lstm import data as forex_data
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Ensure we're in the correct working directory for Kaggle
if os.path.exists('/kaggle/working'):
    # If running on Kaggle, change to the working directory
    os.chdir('/kaggle/working')

# Import SMC functions from the main data module


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR)."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def detect_swings(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Detect swing highs and lows."""
    highs = df['High']
    lows = df['Low']

    swing_high = (highs == highs.rolling(window=window, center=True).max())
    swing_low = (lows == lows.rolling(window=window, center=True).min())

    df = df.copy()
    df['swing_high'] = swing_high
    df['swing_low'] = swing_low
    return df


def add_candlestick_colors(df: pd.DataFrame) -> pd.DataFrame:
    """Add candlestick color column (1 for bullish, -1 for bearish)."""
    df = df.copy()
    df['color'] = np.where(df['Close'] > df['Open'], 1, -1)
    return df


def identify_order_blocks(df: pd.DataFrame, atr: pd.Series, displacement_threshold: float = 1.0) -> pd.DataFrame:
    """Identify Order Blocks with displacement validation."""
    df = df.copy()
    df['color'] = np.where(df['Close'] > df['Open'],
                           1, -1)  # 1 bullish, -1 bearish

    obs = []

    for i in range(2, len(df) - 5):  # Leave room for confirmation
        # Bullish OB: last bearish candle before bullish move
        if df.iloc[i]['color'] == -1:  # Bearish candle
            # Check if next candles are bullish and have displacement
            future_high = df.iloc[i+1:i+6]['High'].max()
            future_low = df.iloc[i+1:i+6]['Low'].min()
            displacement = (future_high - df.iloc[i]['Low']) / atr.iloc[i]
            # Mostly bullish
            if displacement > displacement_threshold and df.iloc[i+1:i+4]['color'].sum() > 0:
                obs.append({
                    'index': i,
                    'type': 'bullish',
                    'high': df.iloc[i]['High'],
                    'low': df.iloc[i]['Low'],
                    'open': df.iloc[i]['Open'],
                    'close': df.iloc[i]['Close'],
                    'displacement_atr': displacement
                })

        # Bearish OB: last bullish candle before bearish move
        elif df.iloc[i]['color'] == 1:  # Bullish candle
            future_high = df.iloc[i+1:i+6]['High'].max()
            future_low = df.iloc[i+1:i+6]['Low'].min()
            displacement = (df.iloc[i]['High'] - future_low) / atr.iloc[i]
            # Mostly bearish
            if displacement > displacement_threshold and df.iloc[i+1:i+4]['color'].sum() < 0:
                obs.append({
                    'index': i,
                    'type': 'bearish',
                    'high': df.iloc[i]['High'],
                    'low': df.iloc[i]['Low'],
                    'open': df.iloc[i]['Open'],
                    'close': df.iloc[i]['Close'],
                    'displacement_atr': displacement
                })

    return pd.DataFrame(obs)


def identify_fvgs(df: pd.DataFrame, atr: pd.Series) -> pd.DataFrame:
    """Identify Fair Value Gaps."""
    fvgs = []

    for i in range(1, len(df) - 1):
        c1, c2, c3 = df.iloc[i-1], df.iloc[i], df.iloc[i+1]

        # Bullish FVG
        if c1['High'] < c3['Low']:
            top = c1['High']
            bottom = c3['Low']
            depth_atr = (bottom - top) / atr.iloc[i]
            fvgs.append({
                'index': i,
                'type': 'bullish',
                'top': top,
                'bottom': bottom,
                'depth_atr': depth_atr
            })

        # Bearish FVG
        elif c1['Low'] > c3['High']:
            top = c3['High']
            bottom = c1['Low']
            depth_atr = (bottom - top) / atr.iloc[i]
            fvgs.append({
                'index': i,
                'type': 'bearish',
                'top': top,
                'bottom': bottom,
                'depth_atr': depth_atr
            })

    return pd.DataFrame(fvgs)


def detect_bos_choch(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Break of Structure (BOS) and Change of Character (ChoCH)."""
    df = df.copy()
    df['bos'] = 0  # 0: no BOS, 1: bullish BOS, -1: bearish BOS
    df['choch'] = 0  # 0: no ChoCH, 1: bullish ChoCH, -1: bearish ChoCH

    # Detect swing points
    swing_df = detect_swings(df)

    swing_highs = swing_df[swing_df['swing_high']].index
    swing_lows = swing_df[swing_df['swing_low']].index

    # Bullish BOS: price breaks above previous swing high
    for i in range(1, len(swing_highs)):
        prev_high_idx = swing_highs[i-1]
        curr_high_idx = swing_highs[i]
        prev_high = df.loc[prev_high_idx, 'High']

        # Check if price breaks above previous swing high
        break_idx = df.loc[curr_high_idx:, 'High'].gt(prev_high).idxmax()
        if break_idx > curr_high_idx:
            df.loc[break_idx, 'bos'] = 1

    # Bearish BOS: price breaks below previous swing low
    for i in range(1, len(swing_lows)):
        prev_low_idx = swing_lows[i-1]
        curr_low_idx = swing_lows[i]
        prev_low = df.loc[prev_low_idx, 'Low']

        # Check if price breaks below previous swing low
        break_idx = df.loc[curr_low_idx:, 'Low'].lt(prev_low).idxmax()
        if break_idx > curr_low_idx:
            df.loc[break_idx, 'bos'] = -1

    # ChoCH: change in market structure (higher highs/lower lows vs lower highs/higher lows)
    for i in range(10, len(df)):
        # Bullish ChoCH: higher high and higher low after lower high and lower low
        recent_hh = df.iloc[i-10:i]['High'].max()
        recent_hl = df.iloc[i-10:i]['Low'].max()
        prev_hh = df.iloc[i-20:i-10]['High'].max()
        prev_hl = df.iloc[i-20:i-10]['Low'].max()

        if recent_hh > prev_hh and recent_hl > prev_hl:
            df.loc[i, 'choch'] = 1

        # Bearish ChoCH: lower high and lower low after higher high and higher low
        elif recent_hh < prev_hh and recent_hl < prev_hl:
            df.loc[i, 'choch'] = -1

    return df


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute additional technical indicators."""
    df = df.copy()

    # ATR
    df['atr'] = compute_atr(df)

    # RSI
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['rsi'] = compute_rsi(df['Close'])

    # Moving averages
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    df['ema_20'] = df['Close'].ewm(span=20).mean()

    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['bb_upper'] = sma_20 + (std_20 * 2)
    df['bb_lower'] = sma_20 - (std_20 * 2)
    df['bb_middle'] = sma_20

    # Volume indicators (if volume exists)
    if 'Volume' in df.columns:
        df['volume_sma'] = df['Volume'].rolling(window=20).mean()

    return df


def add_smc_features_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMC features as columns to the dataframe."""
    df = df.copy()

    # Add candlestick colors first (required for order block detection)
    df = add_candlestick_colors(df)

    # Compute ATR first
    atr = compute_atr(df)

    # Identify SMC structures
    obs = identify_order_blocks(df, atr)
    fvgs = identify_fvgs(df, atr)

    # Initialize SMC feature columns
    df['ob_bullish'] = 0
    df['ob_bearish'] = 0
    df['ob_high'] = np.nan
    df['ob_low'] = np.nan
    df['ob_displacement_atr'] = np.nan

    df['fvg_bullish'] = 0
    df['fvg_bearish'] = 0
    df['fvg_top'] = np.nan
    df['fvg_bottom'] = np.nan
    df['fvg_depth_atr'] = np.nan

    # Add Order Block features
    for _, ob in obs.iterrows():
        idx = int(ob['index'])
        if idx < len(df):
            if ob['type'] == 'bullish':
                df.loc[df.index[idx], 'ob_bullish'] = 1
            else:
                df.loc[df.index[idx], 'ob_bearish'] = 1

            df.loc[df.index[idx], 'ob_high'] = ob['high']
            df.loc[df.index[idx], 'ob_low'] = ob['low']
            df.loc[df.index[idx], 'ob_displacement_atr'] = ob['displacement_atr']

    # Add FVG features
    for _, fvg in fvgs.iterrows():
        idx = int(fvg['index'])
        if idx < len(df):
            if fvg['type'] == 'bullish':
                df.loc[df.index[idx], 'fvg_bullish'] = 1
            else:
                df.loc[df.index[idx], 'fvg_bearish'] = 1

            df.loc[df.index[idx], 'fvg_top'] = fvg['top']
            df.loc[df.index[idx], 'fvg_bottom'] = fvg['bottom']
            df.loc[df.index[idx], 'fvg_depth_atr'] = fvg['depth_atr']

    # Add BOS/ChoCH features
    df = detect_bos_choch(df)

    # Forward fill SMC features to make them available for all timesteps
    smc_cols = ['ob_high', 'ob_low', 'ob_displacement_atr',
                'fvg_top', 'fvg_bottom', 'fvg_depth_atr']
    df[smc_cols] = df[smc_cols].fillna(method='ffill')

    # Fill remaining NaN with 0 for displacement/depth, -1 for prices (indicating no structure)
    df['ob_displacement_atr'] = df['ob_displacement_atr'].fillna(0)
    df['fvg_depth_atr'] = df['fvg_depth_atr'].fillna(0)
    df[['ob_high', 'ob_low', 'fvg_top', 'fvg_bottom']] = df[[
        'ob_high', 'ob_low', 'fvg_top', 'fvg_bottom']].fillna(-1)

    return df


def generate_smc_labels(df: pd.DataFrame, seq_len: int = 60) -> Tuple[pd.Series, pd.Series]:
    """Generate enhanced SMC labels with mitigation and quality scoring."""
    # Use the enhanced SMC labeling from the main data module
    labels, quality_scores = forex_data.generate_enhanced_smc_labels(
        df, seq_len)

    # Convert back to pandas Series with proper indexing
    labels_series = pd.Series(labels, index=df.index[seq_len:], name='label')
    quality_series = pd.Series(
        quality_scores, index=df.index[seq_len:], name='quality_score')

    return labels_series, quality_series


def process_symbol_data(df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    """Process data for a single symbol/timeframe combination."""
    print(f"Processing {symbol} {timeframe}...")

    # Filter data for this symbol/timeframe
    symbol_data = df[(df['symbol'] == symbol) & (
        df['timeframe'] == timeframe)].copy()

    if len(symbol_data) < 100:  # Minimum data requirement
        print(
            f"Skipping {symbol} {timeframe}: insufficient data ({len(symbol_data)} rows)")
        return pd.DataFrame()

    # Sort by time
    symbol_data = symbol_data.sort_index()

    # Add technical indicators
    symbol_data = compute_technical_indicators(symbol_data)

    # Add SMC features
    symbol_data = add_smc_features_to_dataframe(symbol_data)

    # Generate labels with quality scores
    labels, quality_scores = generate_smc_labels(symbol_data)

    # Add to dataframe
    symbol_data['label'] = labels
    symbol_data['quality_score'] = quality_scores

    # Add symbol/timeframe identifiers
    symbol_data['symbol'] = symbol
    symbol_data['timeframe'] = timeframe

    print(f"Completed {symbol} {timeframe}: {len(symbol_data)} rows, "
          f"{symbol_data['label'].value_counts().to_dict()}")

    return symbol_data


def create_lstm_training_dataset(input_file: str = "consolidated_dataset.csv",
                                 output_file: str = "smc_lstm_training_dataset.csv") -> None:
    """Create complete LSTM training dataset with SMC features."""

    print("Loading consolidated dataset...")
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)

    print(f"Loaded {len(df)} rows from {input_file}")
    print(f"Symbols: {df['symbol'].unique()}")
    print(f"Timeframes: {df['timeframe'].unique()}")

    # Process each symbol/timeframe combination
    processed_data = []

    for symbol in df['symbol'].unique():
        for timeframe in df['timeframe'].unique():
            symbol_df = process_symbol_data(df, symbol, timeframe)
            if not symbol_df.empty:
                processed_data.append(symbol_df)

    if not processed_data:
        raise ValueError("No data could be processed")

    # Combine all processed data
    final_df = pd.concat(processed_data, ignore_index=False)

    # Sort by time
    final_df = final_df.sort_index()

    # Final cleanup
    final_df = final_df.dropna()  # Remove any remaining NaN values

    # Feature columns for LSTM (exclude target and identifiers)
    feature_cols = [
        # OHLCV
        'Open', 'High', 'Low', 'Close', 'Volume',

        # Returns
        'returns', 'log_returns',

        # Time features
        'hour', 'day_of_week', 'month',

        # Technical indicators
        'atr', 'rsi', 'sma_20', 'sma_50', 'ema_20',
        'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_lower', 'bb_middle',

        # SMC features
        'ob_bullish', 'ob_bearish', 'ob_high', 'ob_low', 'ob_displacement_atr',
        'fvg_bullish', 'fvg_bearish', 'fvg_top', 'fvg_bottom', 'fvg_depth_atr',
        'bos', 'choch'
    ]

    # Ensure all feature columns exist
    existing_cols = [col for col in feature_cols if col in final_df.columns]
    final_df = final_df[existing_cols + ['symbol', 'timeframe', 'label']]

    print(f"Saving dataset to {output_file}...")
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_df.to_csv(output_file)

    print("\nDataset Summary:")
    print(f"Total rows: {len(final_df)}")
    print(f"Features: {len(existing_cols)}")
    print(f"Symbols: {final_df['symbol'].unique()}")
    print(f"Timeframes: {final_df['timeframe'].unique()}")
    print(f"Label distribution: {final_df['label'].value_counts().to_dict()}")
    print(f"Date range: {final_df.index.min()} to {final_df.index.max()}")

    # Feature importance summary
    print("\nFeature completeness:")
    for col in existing_cols:
        completeness = (final_df[col].notna().sum() / len(final_df)) * 100
        print(f"{col}: {completeness:.1f}%")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create SMC-based LSTM training dataset")
    parser.add_argument("--input", type=str, default="/kaggle/working/training_data/consolidated_dataset.csv",
                        help="Input consolidated CSV file")
    parser.add_argument("--output", type=str, default="/kaggle/working/training_data/smc_lstm_training_dataset.csv",
                        help="Output training dataset CSV file")

    args = parser.parse_args()

    create_lstm_training_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
