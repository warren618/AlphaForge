"""Example factor implementations — replace with your own alpha."""

import pandas as pd
import numpy as np
from src.factor_pipeline.registry import register_factor


@register_factor("simple_momentum")
def simple_momentum(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """N-bar momentum (rate of change).

    Args:
        df: OHLCV DataFrame.
        window: Lookback window.

    Returns:
        Momentum signal series.
    """
    return df["close"].pct_change(window)


@register_factor("rsi")
def rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Relative Strength Index.

    Args:
        df: OHLCV DataFrame.
        window: RSI lookback period.

    Returns:
        RSI values (0-100), centered around 50.
    """
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


@register_factor("volume_momentum")
def volume_momentum(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volume relative to rolling average.

    Args:
        df: OHLCV DataFrame.
        window: Rolling average window.

    Returns:
        Volume ratio (>1 = above average).
    """
    return df["volume"] / df["volume"].rolling(window).mean()
