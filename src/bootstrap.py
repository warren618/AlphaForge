"""Bootstrap 置信区间计算。

对日频收益序列做 Bootstrap 采样，估计 Sharpe Ratio 的置信区间。
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd


def bootstrap_sharpe(
    returns: Union[pd.Series, np.ndarray],
    n_bootstrap: int = 5000,
    annualization: float = 365.0,
    seed: int = 42,
) -> dict:
    """Bootstrap 估计 Sharpe Ratio 的置信区间。

    Args:
        returns: 日频收益序列。
        n_bootstrap: Bootstrap 采样次数。
        annualization: 年化因子（crypto = 365）。
        seed: 随机种子。

    Returns:
        包含 p2.5, p50, p97.5, mean, qualified 的字典。
        qualified=True 表示 5th percentile > 0。

    Example:
        >>> result = bootstrap_sharpe(daily_returns)
        >>> print(f"Sharpe 95% CI: [{result['p2.5']:.2f}, {result['p97.5']:.2f}]")
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 10:
        return {
            "p2.5": 0.0,
            "p5": 0.0,
            "p50": 0.0,
            "p97.5": 0.0,
            "mean": 0.0,
            "qualified": False,
            "n_samples": len(returns),
        }

    rng = np.random.default_rng(seed)
    n = len(returns)
    sharpes = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(returns, size=n, replace=True)
        mu = sample.mean()
        sigma = sample.std(ddof=1)
        if sigma > 0:
            sharpes[i] = mu / sigma * np.sqrt(annualization)
        else:
            sharpes[i] = 0.0

    p2_5, p5, p50, p97_5 = np.percentile(sharpes, [2.5, 5, 50, 97.5])

    return {
        "p2.5": round(float(p2_5), 4),
        "p5": round(float(p5), 4),
        "p50": round(float(p50), 4),
        "p97.5": round(float(p97_5), 4),
        "mean": round(float(sharpes.mean()), 4),
        "qualified": bool(p5 > 0),
        "n_samples": n,
    }
