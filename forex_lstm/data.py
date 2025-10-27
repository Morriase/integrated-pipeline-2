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


def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect candlestick patterns for mitigation confirmation."""
    patterns = pd.DataFrame(index=df.index)

    # Pin Bar detection
    body_size = (df['Close'] - df['Open']).abs()
    upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
    lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']
    total_range = df['High'] - df['Low']

    # Bullish Pin Bar: long lower wick, small body, short upper wick
    patterns['bullish_pin'] = (
        (lower_wick > body_size * 2) &
        (lower_wick > upper_wick * 2) &
        (body_size < total_range * 0.3)
    )

    # Bearish Pin Bar: long upper wick, small body, short lower wick
    patterns['bearish_pin'] = (
        (upper_wick > body_size * 2) &
        (upper_wick > lower_wick * 2) &
        (body_size < total_range * 0.3)
    )

    # Engulfing patterns
    prev_body = body_size.shift(1)
    prev_color = np.where(df['Close'].shift(1) > df['Open'].shift(1), 1, -1)
    curr_color = np.where(df['Close'] > df['Open'], 1, -1)

    # Bullish Engulfing
    patterns['bullish_engulfing'] = (
        (prev_color == -1) & (curr_color == 1) &
        (df['Close'] > df['Open'].shift(1)) &
        (df['Open'] < df['Close'].shift(1)) &
        (body_size > prev_body)
    )

    # Bearish Engulfing
    patterns['bearish_engulfing'] = (
        (prev_color == 1) & (curr_color == -1) &
        (df['Close'] < df['Open'].shift(1)) &
        (df['Open'] > df['Close'].shift(1)) &
        (body_size > prev_body)
    )

    # Inside Bar (contraction)
    patterns['inside_bar'] = (
        (df['High'] <= df['High'].shift(1)) &
        (df['Low'] >= df['Low'].shift(1))
    )

    # Outside Bar (expansion)
    patterns['outside_bar'] = (
        (df['High'] >= df['High'].shift(1)) &
        (df['Low'] <= df['Low'].shift(1))
    )

    return patterns


def detect_mitigation(df: pd.DataFrame, ob: pd.Series, patterns: pd.DataFrame = None, lookback: int = 10) -> dict:
    """Detect if order block has been mitigated with pattern confirmation."""
    ob_high, ob_low = ob['high'], ob['low']
    ob_index = ob['index']
    ob_type = ob['type']

    mitigation_info = {
        'mitigated': False,
        'entry_candle': None,
        'rejection_candle': None,
        'pattern_confirm': False,
        'quality_score': 0.0
    }

    # Look for price entering the OB and then rejection
    start_idx = max(0, ob_index + 1)
    end_idx = min(len(df), ob_index + lookback + 1)

    if ob_type == 'bullish':
        # For bullish OB: price should drop into OB low, then reject upward
        for i in range(start_idx, end_idx):
            if df.iloc[i]['Low'] <= ob_low:
                mitigation_info['entry_candle'] = i
                # Look for rejection (price closes above OB high)
                for j in range(i, end_idx):
                    if df.iloc[j]['Close'] > ob_high:
                        mitigation_info['rejection_candle'] = j
                        mitigation_info['mitigated'] = True

                        # Check for pattern confirmation
                        if patterns is not None and j < len(patterns):
                            rejection_patterns = patterns.iloc[j]
                            if rejection_patterns['bullish_pin'] or rejection_patterns['bullish_engulfing']:
                                mitigation_info['pattern_confirm'] = True

                        break
                break

    elif ob_type == 'bearish':
        # For bearish OB: price should rally into OB high, then reject downward
        for i in range(start_idx, end_idx):
            if df.iloc[i]['High'] >= ob_high:
                mitigation_info['entry_candle'] = i
                # Look for rejection (price closes below OB low)
                for j in range(i, end_idx):
                    if df.iloc[j]['Close'] < ob_low:
                        mitigation_info['rejection_candle'] = j
                        mitigation_info['mitigated'] = True

                        # Check for pattern confirmation
                        if patterns is not None and j < len(patterns):
                            rejection_patterns = patterns.iloc[j]
                            if rejection_patterns['bearish_pin'] or rejection_patterns['bearish_engulfing']:
                                mitigation_info['pattern_confirm'] = True

                        break
                break

    return mitigation_info


def fuzzy_ob_quality_score(ob: pd.Series, df: pd.DataFrame, patterns: pd.DataFrame = None) -> float:
    """Calculate order block quality score using fuzzy logic and multiple factors."""
    try:
        # Base factors
        displacement_score = min(ob.get('displacement_atr', 1.0) / 3.0, 1.0)
        volume_score = min(df.iloc[ob['index']]['Volume'] / 100000, 1.0)

        # Pattern proximity score
        pattern_bonus = 0.0
        if patterns is not None:
            ob_idx = ob['index']
            lookback_window = patterns.iloc[max(
                0, ob_idx-5):min(len(patterns), ob_idx+5)]
            if ob['type'] == 'bullish':
                pattern_bonus = lookback_window[[
                    'bullish_pin', 'bullish_engulfing']].any().any() * 0.2
            else:
                pattern_bonus = lookback_window[[
                    'bearish_pin', 'bearish_engulfing']].any().any() * 0.2

        # Volatility context score
        atr = compute_atr(df)
        volatility_score = min(
            atr.iloc[ob['index']] / atr.rolling(20).mean().iloc[ob['index']], 2.0) / 2.0

        # Recency score (more recent = higher score)
        current_idx = len(df) - 1
        recency_score = max(0, 1 - (current_idx - ob['index']) / 100)

        # Combine scores with weights
        final_score = (
            displacement_score * 0.3 +  # Displacement importance
            volume_score * 0.2 +        # Volume confirmation
            pattern_bonus +             # Pattern bonus
            volatility_score * 0.2 +    # Market context
            recency_score * 0.1         # Recency preference
        )

        return min(final_score, 1.0)

    except Exception as e:
        # Fallback to simple scoring if anything fails
        return (displacement_score + volume_score) / 2.0


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
    """Generate labels based on SMC strategy with mitigation and quality scoring."""
    atr = compute_atr(df)
    obs = identify_order_blocks(df, atr)
    fvgs = identify_fvgs(df, atr)
    patterns = detect_candlestick_patterns(df)

    labels = np.zeros(len(df), dtype=int)  # 0 hold, 1 buy, 2 sell
    quality_scores = np.zeros(len(df))     # Quality scores for each signal

    # Process Order Blocks with mitigation and quality scoring
    for _, ob in obs.iterrows():
        # Calculate quality score
        quality_score = fuzzy_ob_quality_score(ob, df, patterns)

        # Only consider high-quality OBs (above threshold)
        if quality_score < 0.5:
            continue

        # Check for mitigation
        mitigation = detect_mitigation(df, ob, patterns)

        if mitigation['mitigated']:
            rejection_idx = mitigation['rejection_candle']

            # Additional confirmation with pattern
            if mitigation['pattern_confirm']:
                quality_score *= 1.2  # Boost score for pattern confirmation

            if ob['type'] == 'bullish':
                # Bullish signal: OB mitigated, price rejected upward
                if rejection_idx < len(labels):
                    labels[rejection_idx] = 1  # buy
                    quality_scores[rejection_idx] = quality_score

            elif ob['type'] == 'bearish':
                # Bearish signal: OB mitigated, price rejected downward
                if rejection_idx < len(labels):
                    labels[rejection_idx] = 2  # sell
                    quality_scores[rejection_idx] = quality_score

    # Process FVGs with similar logic (less strict quality requirements)
    for _, fvg in fvgs.iterrows():
        # FVGs need less quality threshold but still check mitigation
        quality_score = 0.6  # Base score for FVGs

        # Simple mitigation check for FVGs
        fvg_idx = fvg['index']
        if fvg['type'] == 'bullish':
            # Look for price to fill FVG and reject
            for i in range(fvg_idx + 1, min(len(df), fvg_idx + 15)):
                if df.iloc[i]['Low'] <= fvg['bottom']:
                    # Price entered FVG
                    for j in range(i, min(len(df), i + 10)):
                        if df.iloc[j]['Close'] > fvg['top']:
                            # Price rejected upward
                            labels[j] = 1
                            quality_scores[j] = quality_score
                            break
                    break

        elif fvg['type'] == 'bearish':
            for i in range(fvg_idx + 1, min(len(df), fvg_idx + 15)):
                if df.iloc[i]['High'] >= fvg['top']:
                    for j in range(i, min(len(df), i + 10)):
                        if df.iloc[j]['Close'] < fvg['bottom']:
                            labels[j] = 2
                            quality_scores[j] = quality_score
                            break
                    break

    # For sequences, the label is the action at the end of the sequence
    seq_labels = []
    seq_qualities = []
    for i in range(seq_len, len(labels)):
        seq_labels.append(labels[i])
        seq_qualities.append(quality_scores[i])

    # Return both labels and quality scores
    return np.array(seq_labels), np.array(seq_qualities)


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
