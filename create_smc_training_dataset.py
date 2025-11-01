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


def detect_bos_choch(df: pd.DataFrame, swing_window: int = 5, choch_lookback: int = 20) -> pd.DataFrame:
    """Detect Break of Structure (BOS) and Change of Character (ChoCH) with confirmation flags."""
    frame = df.copy()

    swing_df = detect_swings(frame, window=swing_window)
    frame['prev_swing_high'] = np.where(
        swing_df['swing_high'], frame['High'], np.nan)
    frame['prev_swing_low'] = np.where(
        swing_df['swing_low'], frame['Low'], np.nan)
    frame['prev_swing_high'] = frame['prev_swing_high'].ffill()
    frame['prev_swing_low'] = frame['prev_swing_low'].ffill()

    frame['bos_bull_wick'] = ((frame['High'] > frame['prev_swing_high'])
                              & frame['prev_swing_high'].notna()).astype(int)
    frame['bos_bull_close'] = ((frame['Close'] > frame['prev_swing_high'])
                               & frame['prev_swing_high'].notna()).astype(int)
    frame['bos_bear_wick'] = ((frame['Low'] < frame['prev_swing_low'])
                              & frame['prev_swing_low'].notna()).astype(int)
    frame['bos_bear_close'] = (
        (frame['Close'] < frame['prev_swing_low']) & frame['prev_swing_low'].notna()).astype(int)

    atr_series = forex_data.compute_atr(frame)
    atr_safe = atr_series.replace(0, np.nan)
    frame['bos_momentum_atr'] = 0.0
    bull_mask = frame['bos_bull_close'] == 1
    bear_mask = frame['bos_bear_close'] == 1
    frame.loc[bull_mask, 'bos_momentum_atr'] = (
        (frame.loc[bull_mask, 'Close'] - frame.loc[bull_mask, 'prev_swing_high']) / atr_safe.loc[bull_mask]).fillna(0.0)
    frame.loc[bear_mask, 'bos_momentum_atr'] = (
        (frame.loc[bear_mask, 'prev_swing_low'] - frame.loc[bear_mask, 'Close']) / atr_safe.loc[bear_mask]).fillna(0.0)

    frame['bos_commitment_flag'] = ((frame['bos_bull_close'] == 1) | (
        frame['bos_bear_close'] == 1)).astype(int)
    frame['bos'] = frame['bos_bull_close'] - frame['bos_bear_close']

    structure_direction = frame['bos'].replace(
        0, np.nan).ffill().fillna(0).astype(int)
    frame['trend_state'] = structure_direction

    prev_direction = structure_direction.shift(1).fillna(0).astype(int)
    frame['choch_bull_close'] = ((structure_direction == 1) & (
        prev_direction == -1) & (frame['bos_bull_close'] == 1)).astype(int)
    frame['choch_bear_close'] = ((structure_direction == -1) & (
        prev_direction == 1) & (frame['bos_bear_close'] == 1)).astype(int)

    wick_direction = (frame['bos_bull_wick'] - frame['bos_bear_wick']
                      ).replace(0, np.nan).ffill().fillna(0).astype(int)
    prev_wick_direction = wick_direction.shift(1).fillna(0)
    frame['choch_bull_wick'] = ((frame['bos_bull_wick'] == 1) & (
        prev_wick_direction == -1)).astype(int)
    frame['choch_bear_wick'] = ((frame['bos_bear_wick'] == 1) & (
        prev_wick_direction == 1)).astype(int)

    frame['choch'] = frame['choch_bull_close'] - frame['choch_bear_close']

    if choch_lookback > 0:
        frame['recent_bull_break'] = frame['bos_bull_close'].rolling(
            window=choch_lookback, min_periods=1).max().fillna(0)
        frame['recent_bear_break'] = frame['bos_bear_close'].rolling(
            window=choch_lookback, min_periods=1).max().fillna(0)
    else:
        frame['recent_bull_break'] = 0
        frame['recent_bear_break'] = 0

    return frame


def add_smc_features_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMC features as columns to the dataframe."""
    df = df.copy()

    # Add candlestick colors first (required for order block detection)
    df = add_candlestick_colors(df)

    # Compute ATR first
    atr = forex_data.compute_atr(df)

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

    # Volatility-safe denominators
    atr_series = atr.replace(0, np.nan)

    # Order block size and displacement normalisation
    df['ob_size_atr'] = 0.0
    valid_ob = (df['ob_high'] > 0) & (df['ob_low'] > 0) & atr_series.notna()
    df.loc[valid_ob, 'ob_size_atr'] = ((df.loc[valid_ob, 'ob_high'] - df.loc[valid_ob, 'ob_low']) /
                                       atr_series.loc[valid_ob]).replace([np.inf, -np.inf], 0).fillna(0)

    df['ob_displacement_zscore'] = forex_data.calculate_zscore(
        df['ob_displacement_atr'].fillna(0))

    # Fair value gap depth normalisation
    df['fvg_depth_zscore'] = forex_data.calculate_zscore(
        df['fvg_depth_atr'].fillna(0))

    # Entry distances in ATR terms
    ob_entry_price = np.where(df['ob_bullish'] == 1, df['ob_low'],
                              np.where(df['ob_bearish'] == 1, df['ob_high'], np.nan))
    ob_entry_price = pd.Series(ob_entry_price, index=df.index).ffill()
    ob_entry_price[ob_entry_price < 0] = np.nan
    df['ob_entry_price'] = ob_entry_price
    df['distance_to_ob_entry_atr'] = (
        (df['Close'] - ob_entry_price).abs() / atr_series)
    df['distance_to_ob_entry_atr'] = df['distance_to_ob_entry_atr'].replace(
        [np.inf, -np.inf], np.nan).fillna(0.0)
    df['ob_entry_price'] = df['ob_entry_price'].fillna(-1)

    fvg_entry_price = np.where(df['fvg_bullish'] == 1, df['fvg_top'],
                               np.where(df['fvg_bearish'] == 1, df['fvg_bottom'], np.nan))
    fvg_entry_price = pd.Series(fvg_entry_price, index=df.index).ffill()
    fvg_entry_price[fvg_entry_price < 0] = np.nan
    df['fvg_entry_price'] = fvg_entry_price
    df['distance_to_fvg_entry_atr'] = (
        (df['Close'] - fvg_entry_price).abs() / atr_series)
    df['distance_to_fvg_entry_atr'] = df['distance_to_fvg_entry_atr'].replace(
        [np.inf, -np.inf], np.nan).fillna(0.0)
    df['fvg_entry_price'] = df['fvg_entry_price'].fillna(-1)

    # Risk per trade using structural boundaries
    df['risk_per_trade_atr'] = 0.0
    bullish_mask = (df['ob_bullish'] == 1) & atr_series.notna()
    bearish_mask = (df['ob_bearish'] == 1) & atr_series.notna()
    df.loc[bullish_mask, 'risk_per_trade_atr'] = ((df.loc[bullish_mask, 'ob_entry_price'] - df.loc[bullish_mask, 'ob_low']) /
                                                  atr_series.loc[bullish_mask]).abs().replace([np.inf, -np.inf], 0).fillna(0)
    df.loc[bearish_mask, 'risk_per_trade_atr'] = ((df.loc[bearish_mask, 'ob_high'] - df.loc[bearish_mask, 'ob_entry_price']) /
                                                  atr_series.loc[bearish_mask]).abs().replace([np.inf, -np.inf], 0).fillna(0)

    # Displacement momentum to align with institutional footprint spec
    df['displacement_mag_zscore'] = forex_data.calculate_zscore(
        df['bos_momentum_atr'].fillna(0))

    return df


def generate_smc_labels(df: pd.DataFrame, seq_len: int = 60) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Generate enhanced SMC labels, directions, quality scores, and triple barrier outcomes."""
    (labels,
     quality_scores,
     tbm_outcomes,
     directions,
     trade_returns) = forex_data.generate_enhanced_smc_labels(
        df,
        seq_len=seq_len,
        return_meta=True)

    index = df.index[seq_len:]
    label_series = pd.Series(labels, index=index, name='raw_label')
    quality_series = pd.Series(
        quality_scores, index=index, name='quality_score')
    outcome_series = pd.Series(tbm_outcomes, index=index, name='tbm_outcome')
    direction_series = pd.Series(
        directions, index=index, name='signal_direction')
    return_series = pd.Series(trade_returns, index=index, name='trade_return')

    return label_series, quality_series, outcome_series, direction_series, return_series


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

    # Ensure base statistical features exist
    if 'returns' not in symbol_data.columns:
        symbol_data['returns'] = symbol_data['Close'].pct_change()
    if 'log_returns' not in symbol_data.columns:
        symbol_data['log_returns'] = np.log(
            symbol_data['Close'] / symbol_data['Close'].shift(1))
    if 'hour' not in symbol_data.columns:
        symbol_data['hour'] = symbol_data.index.hour
    if 'day_of_week' not in symbol_data.columns:
        symbol_data['day_of_week'] = symbol_data.index.dayofweek
    if 'month' not in symbol_data.columns:
        symbol_data['month'] = symbol_data.index.month

    # Add technical indicators
    symbol_data = forex_data.compute_technical_indicators(symbol_data)

    # Add SMC features
    symbol_data = add_smc_features_to_dataframe(symbol_data)

    # Generate labels with quality scores
    (raw_labels,
     quality_scores,
     tbm_outcomes,
     signal_directions,
     trade_returns) = generate_smc_labels(symbol_data)

    valid_index = raw_labels.index
    symbol_data = symbol_data.loc[valid_index].copy()

    symbol_data['raw_signal_label'] = raw_labels.reindex(
        valid_index).fillna(0).astype(int)
    symbol_data['quality_score'] = quality_scores.reindex(
        valid_index).fillna(0.0)
    symbol_data['signal_direction'] = signal_directions.reindex(
        valid_index).fillna(0).astype(int)
    symbol_data['tbm_outcome'] = tbm_outcomes.reindex(
        valid_index).fillna(0).astype(int)
    symbol_data['trade_return'] = trade_returns.reindex(
        valid_index).fillna(0.0)

    # Per specification the TBM outcome is the target label (-1, 0, 1)
    symbol_data['label'] = symbol_data['tbm_outcome']

    # Add symbol/timeframe identifiers
    symbol_data['symbol'] = symbol
    symbol_data['timeframe'] = timeframe

    # Replace remaining NaNs (if any structural columns were empty)
    numeric_cols = symbol_data.select_dtypes(include=[np.number]).columns
    symbol_data[numeric_cols] = symbol_data[numeric_cols].fillna(0.0)

    int_columns = [
        'signal_direction',
        'tbm_outcome',
        'label',
        'raw_signal_label',
        'volatility_state',
        'trend_state',
        'bos_bull_wick',
        'bos_bull_close',
        'bos_bear_wick',
        'bos_bear_close',
        'bos_commitment_flag',
        'choch_bull_close',
        'choch_bear_close',
        'choch_bull_wick',
        'choch_bear_wick',
        'recent_bull_break',
        'recent_bear_break'
    ]
    for col in int_columns:
        if col in symbol_data.columns:
            symbol_data[col] = symbol_data[col].astype(int)

    print(f"Completed {symbol} {timeframe}: {len(symbol_data)} rows, "
          f"TBM distribution {symbol_data['label'].value_counts().to_dict()}")

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

    # Feature columns for LSTM (exclude target and identifiers)
    feature_cols = [
        # OHLCV
        'Open', 'High', 'Low', 'Close', 'Volume',

        # Returns
        'returns', 'log_returns',

        # Time features
        'hour', 'day_of_week', 'month',

        # Technical indicators
        'atr', 'rsi', 'sma_20', 'sma_50', 'ema_20', 'ema_50',
        'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_lower', 'bb_middle', 'volume_sma',
        'trend_bias_indicator', 'volatility_state',

        # SMC features
        'ob_bullish', 'ob_bearish', 'ob_high', 'ob_low', 'ob_displacement_atr',
        'ob_size_atr', 'ob_displacement_zscore', 'ob_entry_price', 'distance_to_ob_entry_atr',
        'risk_per_trade_atr',
        'fvg_bullish', 'fvg_bearish', 'fvg_top', 'fvg_bottom', 'fvg_depth_atr',
        'fvg_depth_zscore', 'fvg_entry_price', 'distance_to_fvg_entry_atr',
        'bos_bull_wick', 'bos_bull_close', 'bos_bear_wick', 'bos_bear_close',
        'bos_commitment_flag', 'bos_momentum_atr', 'displacement_mag_zscore',
        'choch_bull_close', 'choch_bear_close', 'choch_bull_wick', 'choch_bear_wick',
        'trend_state', 'recent_bull_break', 'recent_bear_break',

        # Signal metadata features
        'signal_direction', 'quality_score', 'trade_return', 'raw_signal_label'
    ]

    # Ensure all feature columns exist
    existing_cols = [col for col in feature_cols if col in final_df.columns]
    final_df = final_df[existing_cols + ['symbol', 'timeframe', 'label']]

    print(f"Saving dataset to {output_file}...")
    # Create output directory if it exists in path
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
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
