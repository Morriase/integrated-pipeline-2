import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Optional MT5 import - only needed for live data download
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("MetaTrader5 not available - live data download disabled")

from sklearn.preprocessing import MinMaxScaler

from .utils import create_sequences


def initialize_mt5():
    """Initialize MetaTrader 5 connection."""
    if not MT5_AVAILABLE:
        raise RuntimeError("MetaTrader5 not available on this system")
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")
    print("MT5 initialized successfully")


def download_ticker(symbol: str, timeframe=None, bars: int = 10000, save_path: str = None) -> pd.DataFrame:
    """Download OHLCV data for a symbol using MT5. Default is 10,000 H1 bars.

    For forex, common symbols are 'EURUSD', 'GBPUSD', etc.
    """
    if not MT5_AVAILABLE:
        raise RuntimeError("MetaTrader5 not available on this system - cannot download live data")

    # Set default timeframe if MT5 is available
    if timeframe is None:
        timeframe = mt5.TIMEFRAME_H1
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
    if not MT5_AVAILABLE:
        raise RuntimeError("MetaTrader5 not available on this system - cannot download live data")

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

    # Hammer patterns (long lower wick, small body)
    patterns['hammer'] = (
        (lower_wick > body_size * 2) &
        (lower_wick > upper_wick) &
        (body_size < total_range * 0.3)
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

        # Enhanced pattern quality analysis (replaces simple pattern proximity)
        pattern_quality_score = 0.0
        if patterns is not None and ob['index'] < len(patterns):
            pattern_quality_score = calculate_pattern_quality_score(
                patterns.iloc[ob['index']], df.iloc[ob['index']], ob['type'])

        # Volatility context score
        atr = compute_atr(df)
        volatility_score = min(
            atr.iloc[ob['index']] / atr.rolling(20).mean().iloc[ob['index']], 2.0) / 2.0

        # Recency score (more recent = higher score)
        current_idx = len(df) - 1
        recency_score = max(0, 1 - (current_idx - ob['index']) / 100)

        # Combine scores with updated weights
        final_score = (
            displacement_score * 0.25 +   # Displacement importance (reduced)
            volume_score * 0.15 +         # Volume confirmation (reduced)
            # Enhanced pattern quality (increased)
            pattern_quality_score * 0.35 +
            volatility_score * 0.15 +     # Market context
            recency_score * 0.10          # Recency preference
        )

        return min(final_score, 1.0)

    except Exception as e:
        # Fallback to simple scoring if anything fails
        return (displacement_score + volume_score) / 2.0


def calculate_pattern_quality_score(pattern_row: pd.Series, candle_row: pd.Series, ob_type: str) -> float:
    """Calculate sophisticated pattern quality score using fuzzy logic."""
    quality_score = 0.0

    # Pin Bar Quality Assessment
    if ob_type == 'bullish' and pattern_row.get('bullish_pin', False):
        quality_score = max(
            quality_score, calculate_pin_quality(candle_row, 'bullish'))
    elif ob_type == 'bearish' and pattern_row.get('bearish_pin', False):
        quality_score = max(
            quality_score, calculate_pin_quality(candle_row, 'bearish'))

    # Engulfing Pattern Quality
    if pattern_row.get('bullish_engulfing', False) or pattern_row.get('bearish_engulfing', False):
        quality_score = max(
            quality_score, calculate_engulfing_quality(candle_row, ob_type))

    # Hammer Pattern Quality (if we add hammer detection)
    if pattern_row.get('hammer', False):
        quality_score = max(
            quality_score, calculate_hammer_quality(candle_row))

    return quality_score


def calculate_pin_quality(candle: pd.Series, direction: str) -> float:
    """Calculate pin bar quality using fuzzy logic (0-1 scale)."""
    try:
        open_price = candle['Open']
        high_price = candle['High']
        low_price = candle['Low']
        close_price = candle['Close']

        total_range = high_price - low_price
        if total_range == 0:
            return 0.0

        body_size = abs(close_price - open_price)
        body_ratio = body_size / total_range

        if direction == 'bullish':
            # Bullish pin: long lower wick, small body, short upper wick
            lower_wick = open_price - \
                low_price if close_price > open_price else close_price - low_price
            upper_wick = high_price - max(open_price, close_price)

            lower_wick_ratio = lower_wick / total_range
            upper_wick_ratio = upper_wick / total_range

            # Fuzzy scoring: lower wick should be > 60%, upper wick < 20%, body < 30%
            lower_wick_score = min(lower_wick_ratio / 0.6, 1.0)  # Ideal: 60%+
            upper_wick_score = max(
                0, 1 - (upper_wick_ratio / 0.2))  # Ideal: <20%
            body_score = max(0, 1 - (body_ratio / 0.3))  # Ideal: <30%

        else:  # bearish
            # Bearish pin: long upper wick, small body, short lower wick
            upper_wick = high_price - max(open_price, close_price)
            lower_wick = min(open_price, close_price) - low_price

            upper_wick_ratio = upper_wick / total_range
            lower_wick_ratio = lower_wick / total_range

            # Fuzzy scoring: upper wick should be > 60%, lower wick < 20%, body < 30%
            upper_wick_score = min(upper_wick_ratio / 0.6, 1.0)  # Ideal: 60%+
            lower_wick_score = max(
                0, 1 - (lower_wick_ratio / 0.2))  # Ideal: <20%
            body_score = max(0, 1 - (body_ratio / 0.3))  # Ideal: <30%

        # Combine scores with weights
        final_score = (lower_wick_score * 0.4 +
                       upper_wick_score * 0.3 + body_score * 0.3)

        return min(final_score, 1.0)

    except:
        return 0.0


def calculate_engulfing_quality(candle: pd.Series, ob_type: str) -> float:
    """Calculate engulfing pattern quality using fuzzy logic."""
    try:
        body_size = abs(candle['Close'] - candle['Open'])
        total_range = candle['High'] - candle['Low']

        if total_range == 0:
            return 0.0

        body_ratio = body_size / total_range

        # Engulfing patterns should have large bodies
        body_score = min(body_ratio / 0.7, 1.0)  # Ideal: 70%+ body

        # Small wicks are better for engulfing
        upper_wick = candle['High'] - max(candle['Open'], candle['Close'])
        lower_wick = min(candle['Open'], candle['Close']) - candle['Low']
        total_wicks = upper_wick + lower_wick
        wick_ratio = total_wicks / total_range
        wick_score = max(0, 1 - wick_ratio)  # Prefer small wicks

        # Direction alignment bonus
        direction_score = 1.0  # Engulfing patterns are inherently directional

        final_score = (body_score * 0.5 + wick_score *
                       0.3 + direction_score * 0.2)
        return min(final_score, 1.0)

    except:
        return 0.0


def calculate_hammer_quality(candle: pd.Series) -> float:
    """Calculate hammer pattern quality (works for both bullish/bearish hammers)."""
    try:
        total_range = candle['High'] - candle['Low']
        if total_range == 0:
            return 0.0

        body_size = abs(candle['Close'] - candle['Open'])
        body_ratio = body_size / total_range

        # Hammer should have small body and long lower wick
        lower_wick = min(candle['Open'], candle['Close']) - candle['Low']
        lower_wick_ratio = lower_wick / total_range

        # Fuzzy scoring
        body_score = max(0, 1 - (body_ratio / 0.3))  # Prefer small body (<30%)
        # Prefer long lower wick (60%+)
        wick_score = min(lower_wick_ratio / 0.6, 1.0)

        final_score = (body_score * 0.4 + wick_score * 0.6)
        return min(final_score, 1.0)

    except:
        return 0.0


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


def generate_enhanced_smc_labels(df: pd.DataFrame, seq_len: int = 60,
                                 use_trend_filter: bool = True,
                                 use_triple_barrier: bool = True,
                                 min_adx: float = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Generate enhanced SMC labels with trend alignment and Triple Barrier Method.

    Args:
        df: OHLCV DataFrame
        seq_len: Sequence length for LSTM
        use_trend_filter: Whether to filter signals based on trend alignment
        use_triple_barrier: Whether to use Triple Barrier Method for labeling
        min_adx: Minimum ADX threshold for trend strength

    Returns:
        Tuple of (labels, quality_scores) arrays
    """
    atr = compute_atr(df)
    obs = identify_order_blocks(df, atr)
    fvgs = identify_fvgs(df, atr)
    patterns = detect_candlestick_patterns(df)

    # Calculate trend indicators if needed
    if use_trend_filter:
        trend = detect_trend_direction(df)
        adx = calculate_adx(df)

    labels = np.zeros(len(df), dtype=int)  # 0 hold, 1 buy, 2 sell
    quality_scores = np.zeros(len(df))     # Quality scores for each signal

    # Process Order Blocks with enhanced logic
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

            # Trend filter: only trade if trend is aligned and strong
            if use_trend_filter:
                signal_trend = 1 if ob['type'] == 'bullish' else -1
                current_trend = trend.iloc[rejection_idx]
                current_adx = adx.iloc[rejection_idx]

                # Check trend alignment
                if current_trend != signal_trend or current_adx < min_adx:
                    continue  # Skip signal if trend not aligned

                quality_score *= 1.1  # Boost score for trend alignment

            # Apply Triple Barrier Method if enabled
            if use_triple_barrier:
                direction = 1 if ob['type'] == 'bullish' else -1
                barrier_result = apply_triple_barrier_method(
                    df, rejection_idx, direction)

                # Only keep signal if it resulted in a win
                if barrier_result['outcome'] != 1:
                    continue

                # Adjust quality score based on return
                quality_score *= (1 + barrier_result['return'])

            # Set the signal
            if ob['type'] == 'bullish':
                if rejection_idx < len(labels):
                    labels[rejection_idx] = 1  # buy
                    quality_scores[rejection_idx] = quality_score
            elif ob['type'] == 'bearish':
                if rejection_idx < len(labels):
                    labels[rejection_idx] = 2  # sell
                    quality_scores[rejection_idx] = quality_score

    # Process FVGs with similar enhanced logic (less strict requirements)
    for _, fvg in fvgs.iterrows():
        quality_score = 0.6  # Base score for FVGs
        fvg_idx = fvg['index']

        # Trend filter for FVGs
        if use_trend_filter:
            signal_trend = 1 if fvg['type'] == 'bullish' else -1
            current_trend = trend.iloc[fvg_idx]
            current_adx = adx.iloc[fvg_idx]

            if current_trend != signal_trend or current_adx < min_adx:
                continue

        if fvg['type'] == 'bullish':
            # Look for price to fill FVG and reject
            for i in range(fvg_idx + 1, min(len(df), fvg_idx + 15)):
                if df.iloc[i]['Low'] <= fvg['bottom']:
                    # Price entered FVG
                    for j in range(i, min(len(df), i + 10)):
                        if df.iloc[j]['Close'] > fvg['top']:
                            # Price rejected upward - check trend and barriers
                            if use_trend_filter:
                                current_trend = trend.iloc[j]
                                current_adx = adx.iloc[j]
                                if current_trend != 1 or current_adx < min_adx:
                                    break

                            if use_triple_barrier:
                                barrier_result = apply_triple_barrier_method(
                                    df, j, 1)
                                if barrier_result['outcome'] != 1:
                                    break

                            labels[j] = 1
                            quality_scores[j] = quality_score
                            break
                    break

        elif fvg['type'] == 'bearish':
            for i in range(fvg_idx + 1, min(len(df), fvg_idx + 15)):
                if df.iloc[i]['High'] >= fvg['top']:
                    for j in range(i, min(len(df), i + 10)):
                        if df.iloc[j]['Close'] < fvg['bottom']:
                            # Price rejected downward - check trend and barriers
                            if use_trend_filter:
                                current_trend = trend.iloc[j]
                                current_adx = adx.iloc[j]
                                if current_trend != -1 or current_adx < min_adx:
                                    break

                            if use_triple_barrier:
                                barrier_result = apply_triple_barrier_method(
                                    df, j, -1)
                                if barrier_result['outcome'] != 1:
                                    break

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


# ============================================================================
# TREND ANALYSIS AND MULTI-TIMEFRAME FUNCTIONS
# ============================================================================

def detect_trend_direction(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Detect trend direction using EMA crossover and slope analysis.

    Returns:
        Series with values: 1 (uptrend), -1 (downtrend), 0 (sideways)
    """
    # Calculate EMAs
    ema_short = df['Close'].ewm(span=period//2).mean()
    ema_long = df['Close'].ewm(span=period).mean()

    # Calculate slope using linear regression on recent closes
    def calculate_slope(series, window):
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(0)
            else:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(window)
                slope = np.polyfit(x, y, 1)[0]
                slopes.append(slope)
        return pd.Series(slopes, index=series.index)

    slope_short = calculate_slope(df['Close'], period//2)
    slope_long = calculate_slope(df['Close'], period)

    # Trend determination
    trend = pd.Series(0, index=df.index, name='trend')

    # Uptrend: short EMA > long EMA and positive slope
    uptrend_mask = (ema_short > ema_long) & (
        slope_short > 0) & (slope_long > 0)
    trend[uptrend_mask] = 1

    # Downtrend: short EMA < long EMA and negative slope
    downtrend_mask = (ema_short < ema_long) & (
        slope_short < 0) & (slope_long < 0)
    trend[downtrend_mask] = -1

    return trend


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index (ADX) for trend strength."""
    # True Range
    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ], axis=1).max(axis=1)

    # Directional Movement
    dm_plus = pd.Series(0, index=df.index)
    dm_minus = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        move_up = df.iloc[i]['High'] - df.iloc[i-1]['High']
        move_down = df.iloc[i-1]['Low'] - df.iloc[i]['Low']

        dm_plus.iloc[i] = move_up if move_up > move_down and move_up > 0 else 0
        dm_minus.iloc[i] = move_down if move_down > move_up and move_down > 0 else 0

    # Smoothed averages
    atr = tr.rolling(window=period).mean()
    di_plus = 100 * (dm_plus.rolling(window=period).mean() / atr)
    di_minus = 100 * (dm_minus.rolling(window=period).mean() / atr)

    # ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=period).mean()

    return adx.fillna(0)


def check_trend_alignment(current_trend: int, higher_trend: int, adx_current: float,
                          adx_higher: float, min_adx: float = 20) -> bool:
    """Check if trends are aligned between timeframes.

    Args:
        current_trend: Trend direction of current timeframe (-1, 0, 1)
        higher_trend: Trend direction of higher timeframe (-1, 0, 1)
        adx_current: ADX value of current timeframe
        adx_higher: ADX value of higher timeframe
        min_adx: Minimum ADX threshold for trend strength

    Returns:
        True if trends are aligned and strong enough
    """
    # Both timeframes must have the same trend direction
    if current_trend != higher_trend:
        return False

    # Both timeframes must have sufficient trend strength
    if adx_current < min_adx or adx_higher < min_adx:
        return False

    # Must be a clear trend (not sideways)
    if current_trend == 0:
        return False

    return True


def get_multi_timeframe_trend_alignment(df_multi: pd.DataFrame, current_tf: str = 'H1',
                                        higher_tf: str = 'H4', min_adx: float = 20) -> pd.Series:
    """Get trend alignment mask for multi-timeframe data.

    Args:
        df_multi: DataFrame with multi-timeframe data (columns suffixed with _M15, _H1, _H4, _D1)
        current_tf: Current timeframe suffix (e.g., 'H1')
        higher_tf: Higher timeframe suffix (e.g., 'H4')
        min_adx: Minimum ADX for trend strength

    Returns:
        Series of boolean values indicating trend alignment
    """
    # Extract data for each timeframe
    cols_current = [
        col for col in df_multi.columns if col.endswith(f'_{current_tf}')]
    cols_higher = [
        col for col in df_multi.columns if col.endswith(f'_{higher_tf}')]

    if not cols_current or not cols_higher:
        return pd.Series(False, index=df_multi.index)

    # Create dataframes for each timeframe
    df_current = df_multi[cols_current].copy()
    df_higher = df_multi[cols_higher].copy()

    # Remove timeframe suffix from column names
    df_current.columns = [col.replace(
        f'_{current_tf}', '') for col in df_current.columns]
    df_higher.columns = [col.replace(f'_{higher_tf}', '')
                         for col in df_higher.columns]

    # Resample higher timeframe to match current timeframe index
    df_higher_resampled = df_higher.reindex(
        df_multi.index).fillna(method='ffill')

    # Calculate trends and ADX for both timeframes
    trend_current = detect_trend_direction(df_current)
    trend_higher = detect_trend_direction(df_higher_resampled)
    adx_current = calculate_adx(df_current)
    adx_higher = calculate_adx(df_higher_resampled)

    # Check alignment for each timestamp
    alignment = pd.Series(False, index=df_multi.index)
    for i in range(len(df_multi)):
        alignment.iloc[i] = check_trend_alignment(
            trend_current.iloc[i],
            trend_higher.iloc[i],
            adx_current.iloc[i],
            adx_higher.iloc[i],
            min_adx
        )

    return alignment


# ============================================================================
# TRIPLE BARRIER METHOD
# ============================================================================

def apply_triple_barrier_method(df: pd.DataFrame, entry_idx: int, direction: int,
                                profit_factor: float = 2.0, stop_factor: float = 1.0,
                                max_bars: int = 50) -> dict:
    """Apply Triple Barrier Method to determine trade outcome.

    Args:
        df: OHLCV DataFrame
        entry_idx: Entry candle index
        direction: 1 for long, -1 for short
        profit_factor: Profit target multiplier (relative to stop distance)
        stop_factor: Stop loss multiplier (relative to ATR)
        max_bars: Maximum bars to hold the trade

    Returns:
        Dict with outcome, exit_idx, and return
    """
    if entry_idx >= len(df) - 1:
        return {'outcome': 0, 'exit_idx': entry_idx, 'return': 0.0}

    entry_price = df.iloc[entry_idx]['Close']
    atr = compute_atr(df).iloc[entry_idx]

    # Calculate stop distance based on ATR
    stop_distance = atr * stop_factor

    # Set profit target and stop loss
    if direction == 1:  # Long
        profit_target = entry_price + (stop_distance * profit_factor)
        stop_loss = entry_price - stop_distance
    else:  # Short
        profit_target = entry_price - (stop_distance * profit_factor)
        stop_loss = entry_price + stop_distance

    # Look forward from entry
    for i in range(entry_idx + 1, min(len(df), entry_idx + max_bars + 1)):
        high = df.iloc[i]['High']
        low = df.iloc[i]['Low']
        close = df.iloc[i]['Close']

        # Check profit target
        if direction == 1 and high >= profit_target:
            return {'outcome': 1, 'exit_idx': i, 'return': (profit_target - entry_price) / entry_price}
        elif direction == -1 and low <= profit_target:
            return {'outcome': 1, 'exit_idx': i, 'return': (entry_price - profit_target) / entry_price}

        # Check stop loss
        if direction == 1 and low <= stop_loss:
            return {'outcome': -1, 'exit_idx': i, 'return': (stop_loss - entry_price) / entry_price}
        elif direction == -1 and high >= stop_loss:
            return {'outcome': -1, 'exit_idx': i, 'return': (entry_price - stop_loss) / entry_price}

    # Time exit (max bars reached)
    exit_price = df.iloc[min(len(df) - 1, entry_idx + max_bars)]['Close']
    trade_return = (exit_price - entry_price) / \
        entry_price if direction == 1 else (
            entry_price - exit_price) / entry_price
    outcome = 0 if abs(trade_return) < 0.001 else (
        1 if trade_return > 0 else -1)  # Small threshold for neutral

    return {
        'outcome': outcome,
        'exit_idx': min(len(df) - 1, entry_idx + max_bars),
        'return': trade_return
    }


# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================

def walk_forward_split(data: pd.DataFrame, n_splits: int = 5, train_size: float = 0.7,
                       validation_size: float = 0.2, step_size: float = 0.1):
    """Generate walk-forward validation splits for time series.

    Args:
        data: Time series DataFrame
        n_splits: Number of splits
        train_size: Proportion of data for training (0-1)
        validation_size: Proportion of data for validation (0-1)
        step_size: Step size for sliding window (0-1)

    Yields:
        Tuple of (train_indices, validation_indices, test_indices)
    """
    n_samples = len(data)
    train_samples = int(n_samples * train_size)
    validation_samples = int(n_samples * validation_size)
    step_samples = int(n_samples * step_size)

    for i in range(n_splits):
        train_start = i * step_samples
        train_end = train_start + train_samples

        if train_end >= n_samples:
            break

        val_start = train_end
        val_end = val_start + validation_samples

        test_start = val_end
        test_end = min(test_start + validation_samples, n_samples)

        if test_end > n_samples:
            break

        train_indices = range(train_start, train_end)
        val_indices = range(val_start, val_end)
        test_indices = range(test_start, test_end)

        yield train_indices, val_indices, test_indices


def evaluate_walk_forward(model_class, data: pd.DataFrame, labels: pd.Series,
                          n_splits: int = 5, seq_len: int = 60, **model_kwargs):
    """Evaluate model using walk-forward validation.

    Args:
        model_class: PyTorch model class
        data: Feature DataFrame
        labels: Target labels
        n_splits: Number of walk-forward splits
        seq_len: Sequence length for LSTM
        **model_kwargs: Model initialization parameters

    Returns:
        Dict with performance metrics across all splits
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    all_predictions = []
    all_true_labels = []
    split_metrics = []

    for split_idx, (train_idx, val_idx, test_idx) in enumerate(
            walk_forward_split(data, n_splits=n_splits)):

        print(f"Processing split {split_idx + 1}/{n_splits}")

        # Prepare data for this split
        train_data = data.iloc[train_idx]
        train_labels = labels.iloc[train_idx]
        val_data = data.iloc[val_idx]
        val_labels = labels.iloc[val_idx]
        test_data = data.iloc[test_idx]
        test_labels = labels.iloc[test_idx]

        # Create sequences
        X_train, y_train = create_sequences(
            train_data.values, train_labels.values, seq_len)
        X_val, y_val = create_sequences(
            val_data.values, val_labels.values, seq_len)
        X_test, y_test = create_sequences(
            test_data.values, test_labels.values, seq_len)

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        # Convert to tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        X_train = torch.FloatTensor(X_train).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        X_val = torch.FloatTensor(X_val).to(device)
        y_val = torch.LongTensor(y_val).to(device)
        X_test = torch.FloatTensor(X_test).to(device)
        y_test = torch.LongTensor(y_test).to(device)

        # Initialize model
        model = model_class(**model_kwargs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Train model
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(100):  # Max epochs
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(),
                           f'best_model_split_{split_idx}.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        # Load best model and evaluate on test set
        model.load_state_dict(torch.load(f'best_model_split_{split_idx}.pth'))
        model.eval()

        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs, 1)

        # Store predictions
        all_predictions.extend(predicted.cpu().numpy())
        all_true_labels.extend(y_test.cpu().numpy())

        # Calculate metrics for this split
        accuracy = accuracy_score(
            y_test.cpu().numpy(), predicted.cpu().numpy())
        precision = precision_score(y_test.cpu().numpy(
        ), predicted.cpu().numpy(), average='weighted', zero_division=0)
        recall = recall_score(y_test.cpu().numpy(), predicted.cpu(
        ).numpy(), average='weighted', zero_division=0)
        f1 = f1_score(y_test.cpu().numpy(), predicted.cpu().numpy(),
                      average='weighted', zero_division=0)

        split_metrics.append({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

        print(f"Split {split_idx + 1} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    # Overall metrics
    overall_metrics = {
        'mean_accuracy': np.mean([m['accuracy'] for m in split_metrics]),
        'std_accuracy': np.std([m['accuracy'] for m in split_metrics]),
        'mean_f1': np.mean([m['f1'] for m in split_metrics]),
        'std_f1': np.std([m['f1'] for m in split_metrics]),
        'split_metrics': split_metrics
    }

    return overall_metrics
