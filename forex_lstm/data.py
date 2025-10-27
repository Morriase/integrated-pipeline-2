import os
from typing import Tuple

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from sklearn.preprocessing import MinMaxScaler


def initialize_mt5():
    """Initialize MetaTrader 5 connection."""
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")
    print("MT5 initialized successfully")


def download_ticker(symbol: str, timeframe=mt5.TIMEFRAME_H1, bars: int = 10000, save_path: str = None) -> pd.DataFrame:
    """Download OHLCV data for a symbol using MT5. Default is 10,000 H1 bars.

    For forex, common symbols are 'EURUSD', 'GBPUSD', etc.
    """
    if not mt5.initialize():
        initialize_mt5()

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        raise RuntimeError(
            f"No data downloaded for {symbol}: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('time')
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume'
    })

    # Save to CSV if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)
        print(f"Data saved to {save_path}")

    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def download_multi_timeframe(symbol: str, bars: int = 10000) -> pd.DataFrame:
    """Download OHLCV data for multiple timeframes: M15, H1, H4."""
    timeframes = [
        (mt5.TIMEFRAME_M15, 'M15'),
        (mt5.TIMEFRAME_H1, 'H1'),
        (mt5.TIMEFRAME_H4, 'H4')
    ]

    dfs = {}
    for tf, name in timeframes:
        df = download_ticker(symbol, tf, bars)
        df = df.add_suffix(f'_{name}')
        dfs[name] = df

    # Merge on time index, forward fill to align
    combined = pd.concat(dfs.values(), axis=1, join='outer').fillna(
        method='ffill').dropna()
    return combined


def download_multi_symbol_timeframe(symbols: list, timeframes: dict, bars: int = 10000, data_dir: str = "DATA"):
    """Download OHLCV data for multiple symbols and timeframes."""
    import os
    os.makedirs(data_dir, exist_ok=True)

    for symbol in symbols:
        for tf_name, tf_code in timeframes.items():
            save_path = f"{data_dir}/{symbol}_{tf_name}_{bars}bars.csv"
            print(f"Downloading {symbol} {tf_name}...")
            try:
                df = download_ticker(symbol, tf_code, bars, save_path)
                print(f"Saved {len(df)} bars to {save_path}")
            except Exception as e:
                print(f"Failed to download {symbol} {tf_name}: {e}")

    print("Batch download completed.")


def load_csv(path: str, date_col: str = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    return df


def prepare_ohlc_series(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare OHLCV data, ensuring datetime index and dropping NaNs."""
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain {required_cols}")
    series = df[required_cols].copy()
    series = series.dropna()
    return series


def scale_series(series: pd.DataFrame, scaler: MinMaxScaler = None) -> Tuple[np.ndarray, MinMaxScaler]:
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaled = scaler.fit_transform(series.values)
    return scaled, scaler


def save_scaler(scaler: MinMaxScaler, path: str):
    import joblib

    joblib.dump(scaler, path)


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


def generate_smc_labels(df: pd.DataFrame, seq_len: int) -> np.ndarray:
    """Generate labels based on SMC strategy: buy/sell/hold."""
    atr = compute_atr(df)
    obs = identify_order_blocks(df, atr)
    fvgs = identify_fvgs(df, atr)

    labels = np.zeros(len(df), dtype=int)  # 0 hold, 1 buy, 2 sell

    # For each OB or FVG, find retracements and apply simple labeling
    for _, ob in obs.iterrows():
        idx = ob['index']
        if ob['type'] == 'bullish':
            # Look for retracement to OB low, then buy if price breaks high
            for i in range(idx + 1, len(df)):
                if df.iloc[i]['Low'] <= ob['low'] and df.iloc[i]['Close'] > ob['high']:
                    labels[i] = 1  # buy
                    break
        elif ob['type'] == 'bearish':
            for i in range(idx + 1, len(df)):
                if df.iloc[i]['High'] >= ob['high'] and df.iloc[i]['Close'] < ob['low']:
                    labels[i] = 2  # sell
                    break

    # Similar for FVGs
    for _, fvg in fvgs.iterrows():
        idx = fvg['index']
        if fvg['type'] == 'bullish':
            for i in range(idx + 1, len(df)):
                if df.iloc[i]['Low'] <= fvg['bottom'] and df.iloc[i]['Close'] > fvg['top']:
                    labels[i] = 1
                    break
        elif fvg['type'] == 'bearish':
            for i in range(idx + 1, len(df)):
                if df.iloc[i]['High'] >= fvg['top'] and df.iloc[i]['Close'] < fvg['bottom']:
                    labels[i] = 2
                    break

    # For sequences, the label is the action at the end of the sequence
    seq_labels = []
    for i in range(seq_len, len(labels)):
        seq_labels.append(labels[i])

    return np.array(seq_labels)


def load_scaler(path: str) -> MinMaxScaler:
    import joblib

    return joblib.load(path)


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


def generate_smc_labels(df: pd.DataFrame, seq_len: int) -> np.ndarray:
    """Generate labels based on SMC strategy: buy/sell/hold."""
    atr = compute_atr(df)
    obs = identify_order_blocks(df, atr)
    fvgs = identify_fvgs(df, atr)

    labels = np.zeros(len(df), dtype=int)  # 0 hold, 1 buy, 2 sell

    # For each OB or FVG, find retracements and apply simple labeling
    for _, ob in obs.iterrows():
        idx = ob['index']
        if ob['type'] == 'bullish':
            # Look for retracement to OB low, then buy if price breaks high
            for i in range(idx + 1, len(df)):
                if df.iloc[i]['Low'] <= ob['low'] and df.iloc[i]['Close'] > ob['high']:
                    labels[i] = 1  # buy
                    break
        elif ob['type'] == 'bearish':
            for i in range(idx + 1, len(df)):
                if df.iloc[i]['High'] >= ob['high'] and df.iloc[i]['Close'] < ob['low']:
                    labels[i] = 2  # sell
                    break

    # Similar for FVGs
    for _, fvg in fvgs.iterrows():
        idx = fvg['index']
        if fvg['type'] == 'bullish':
            for i in range(idx + 1, len(df)):
                if df.iloc[i]['Low'] <= fvg['bottom'] and df.iloc[i]['Close'] > fvg['top']:
                    labels[i] = 1
                    break
        elif fvg['type'] == 'bearish':
            for i in range(idx + 1, len(df)):
                if df.iloc[i]['High'] >= fvg['top'] and df.iloc[i]['Close'] < fvg['bottom']:
                    labels[i] = 2
                    break

    # For sequences, the label is the action at the end of the sequence
    seq_labels = []
    for i in range(seq_len, len(labels)):
        seq_labels.append(labels[i])

    return np.array(seq_labels)
