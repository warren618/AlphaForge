"""Strategy engine — converts factor signals into position decisions.

The strategy reads factor scores from a config-driven factor list,
combines them via weighted z-score aggregation, and produces
directional signals (+1 long / -1 short / 0 flat).

No factor names are hardcoded. All factor definitions come from
the strategy YAML config file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def load_strategy_config(path: str) -> dict:
    """Load strategy YAML config.

    Args:
        path: Path to strategy YAML file.

    Returns:
        Strategy configuration dict.
    """
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_composite_signal(
    df: pd.DataFrame,
    factors: list[dict],
    weights: Optional[dict[str, float]] = None,
) -> pd.Series:
    """Compute composite signal from multiple factors via weighted z-score.

    Args:
        df: DataFrame with factor columns.
        factors: List of factor configs, each with 'name' and 'column' keys.
        weights: Optional factor name -> weight mapping. Equal weights if None.

    Returns:
        Series of composite signal values.
    """
    scores = []
    factor_weights = []

    for fconf in factors:
        col = fconf.get("column", fconf["name"])
        if col not in df.columns:
            continue

        series = df[col].astype(float)
        # Z-score normalization (rolling 60-bar window)
        rolling_mean = series.rolling(60, min_periods=20).mean()
        rolling_std = series.rolling(60, min_periods=20).std().replace(0, np.nan)
        z = (series - rolling_mean) / rolling_std
        z = z.clip(-3, 3).fillna(0)

        w = 1.0
        if weights and fconf["name"] in weights:
            w = weights[fconf["name"]]

        scores.append(z * w)
        factor_weights.append(w)

    if not scores:
        return pd.Series(0.0, index=df.index)

    total_weight = sum(factor_weights)
    composite = sum(scores) / total_weight
    return composite


def generate_positions(
    composite_signal: pd.Series,
    long_threshold: float = 0.5,
    short_threshold: float = -0.5,
) -> pd.Series:
    """Convert composite signal to position direction.

    Args:
        composite_signal: Weighted z-score signal.
        long_threshold: Signal above this → long (+1).
        short_threshold: Signal below this → short (-1).

    Returns:
        Series of position directions: +1, -1, or 0.
    """
    positions = pd.Series(0, index=composite_signal.index, dtype=int)
    positions[composite_signal > long_threshold] = 1
    positions[composite_signal < short_threshold] = -1
    return positions
