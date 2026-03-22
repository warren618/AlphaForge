"""多因子组合搜索引擎。

支持三种搜索策略：
- greedy_forward: 按 IC IR 排序 → 逐个加入(相关性<0.7互斥) → 直到边际增益 < 阈值
- regularized_optimize: 因子 z-score 矩阵 X + fwd_return y → LassoCV/RidgeCV → coef 即权重
- cluster_select: 相关性矩阵 → 层次聚类 → 每 cluster 选 IC IR 最高代表 → greedy/lasso
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.cluster.hierarchy import fcluster, linkage

from src.logging_config import get_logger

logger = get_logger("combo_search")

STRATEGIES_DIR = Path(__file__).parent.parent.parent / "strategies" / "auto"


@dataclass
class StrategyCandidate:
    """组合搜索产出的策略候选。

    Attributes:
        name: 策略名称。
        factors: 选中的因子列表。
        weights: 因子权重字典。
        method: 搜索方法。
        ic_ir_composite: 组合 IC IR 估计。
        diversity_score: 因子多样性分数。
        flipped: 被反转的因子集合（IC 为负 → 乘以 -1）。
    """

    name: str
    factors: list[str]
    weights: dict[str, float]
    method: str
    ic_ir_composite: float = 0.0
    diversity_score: float = 0.0
    flipped: set = field(default_factory=set)
    metadata: dict = field(default_factory=dict)


def auto_flip_factors(
    df: pd.DataFrame,
    factor_cols: list[str],
    horizon: int = 12,
    min_abs_ic_ir: float = 0.3,
) -> tuple[pd.DataFrame, set[str]]:
    """检测负 IC 因子并翻转，使所有因子 IC 方向一致（正）。

    Args:
        df: 含因子列和 close 列的 DataFrame。
        factor_cols: 候选因子列名列表。
        horizon: 前瞻周期。
        min_abs_ic_ir: 翻转的最低 |IC IR| 门槛。

    Returns:
        (modified_df, flipped_cols) — 翻转后的 DataFrame 和被翻转的列名集合。
    """
    df = df.copy()
    fwd_ret = df["close"].pct_change(horizon).shift(-horizon)
    flipped: set[str] = set()

    for col in factor_cols:
        if col not in df.columns:
            continue
        fv = df[col].astype(float)
        valid = fv.notna() & fwd_ret.notna()
        if valid.sum() < 100:
            continue
        ic, _ = sp_stats.spearmanr(fv[valid], fwd_ret[valid])
        if np.isnan(ic):
            continue
        # 粗估 IC IR：全局 IC / 滚动 IC 的 std 近似
        # 这里只需要判断方向，用全局 IC 即可
        if ic < 0:
            df[col] = -df[col]
            flipped.add(col)
            logger.info(f"Flipped {col}: raw IC={ic:.4f} → {-ic:.4f}")

    return df, flipped


def _calc_rolling_ic(factor_values: pd.Series, fwd_ret: pd.Series, window: int = 252) -> pd.Series:
    """计算滚动 Rank IC 序列（向量化版本）。

    大数据集上使用降采样 + 向量化 rank 相关计算，
    避免纯 Python 循环导致的内存/速度问题。

    Args:
        factor_values: 因子值序列。
        fwd_ret: 前瞻收益序列。
        window: 滚动窗口大小。

    Returns:
        IC 时序 Series。
    """
    valid = factor_values.notna() & fwd_ret.notna()
    fv = factor_values[valid].values.astype(np.float64)
    fr = fwd_ret[valid].values.astype(np.float64)
    idx = factor_values[valid].index

    n = len(fv)
    if n < window:
        return pd.Series(dtype=float)

    # 对超大数据集做均匀降采样 (每 step 个点取一个 IC)
    max_ic_points = 2000
    total_points = n - window
    step = max(1, total_points // max_ic_points)

    ic_list = []
    ic_idx = []
    for i in range(window, n, step):
        chunk_f = fv[i - window:i]
        chunk_r = fr[i - window:i]
        # 快速 rank 相关 (Pearson of ranks)
        rf = chunk_f.argsort().argsort().astype(np.float64)
        rr = chunk_r.argsort().argsort().astype(np.float64)
        rf -= rf.mean()
        rr -= rr.mean()
        denom = np.sqrt((rf ** 2).sum() * (rr ** 2).sum())
        ic = (rf * rr).sum() / denom if denom > 0 else 0.0
        ic_list.append(ic)
        ic_idx.append(idx[i])

    return pd.Series(ic_list, index=ic_idx)


def _calc_factor_stats(
    df: pd.DataFrame,
    factor_cols: list[str],
    horizon: int = 12,
) -> dict[str, dict]:
    """计算每个因子的 IC / IC IR。

    Args:
        df: 含因子列和 close 列的 DataFrame。
        factor_cols: 因子列名列表。
        horizon: 前瞻周期。

    Returns:
        {col: {"ic_mean": ..., "ic_ir": ..., "ic_series": ...}} 字典。
    """
    fwd_ret = df["close"].pct_change(horizon).shift(-horizon)
    stats = {}

    for col in factor_cols:
        if col not in df.columns:
            continue
        fv = df[col].astype(float)
        # 跳过常数列（无信息量，spearmanr 会报错）
        if fv.nunique() <= 1:
            stats[col] = {"ic_mean": 0.0, "ic_ir": 0.0, "ic_series": pd.Series(dtype=float)}
            continue
        ic_series = _calc_rolling_ic(fv, fwd_ret)

        if len(ic_series) < 10:
            stats[col] = {"ic_mean": 0.0, "ic_ir": 0.0, "ic_series": ic_series}
            continue

        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0

        stats[col] = {
            "ic_mean": float(ic_mean),
            "ic_ir": float(ic_ir),
            "ic_series": ic_series,
        }

    return stats


def greedy_forward(
    df: pd.DataFrame,
    factor_cols: list[str],
    horizon: int = 12,
    max_corr: float = 0.7,
    min_ic_ir: float = 0.3,
    max_factors: int = 15,
) -> StrategyCandidate:
    """贪心前向选择。

    按 IC IR 排序因子，逐个加入（与已选因子相关性 < max_corr 才可加入），
    直到无更多因子或达到上限。

    Args:
        df: 含因子列和 close 列的 DataFrame。
        factor_cols: 候选因子列名列表。
        horizon: 前瞻周期。
        max_corr: 最大允许相关性。
        min_ic_ir: 最低 IC IR 门槛。
        max_factors: 最多选择因子数。

    Returns:
        StrategyCandidate 实例。
    """
    factor_stats = _calc_factor_stats(df, factor_cols, horizon)

    # 按 IC IR 降序排列
    ranked = sorted(
        [(col, s["ic_ir"]) for col, s in factor_stats.items() if s["ic_ir"] >= min_ic_ir],
        key=lambda x: -x[1],
    )

    if not ranked:
        logger.warning("No factors meet IC IR threshold")
        return StrategyCandidate(
            name="greedy_empty", factors=[], weights={}, method="greedy"
        )

    selected: list[str] = []
    selected_weights: dict[str, float] = {}

    for col, ic_ir in ranked:
        if len(selected) >= max_factors:
            break

        # 检查与已选因子的相关性
        too_correlated = False
        for existing in selected:
            valid = df[col].notna() & df[existing].notna()
            if valid.sum() < 30:
                continue
            corr = df[col][valid].corr(df[existing][valid])
            if abs(corr) >= max_corr:
                too_correlated = True
                logger.debug(f"Skipping {col}: corr={corr:.3f} with {existing}")
                break

        if not too_correlated:
            selected.append(col)
            selected_weights[col] = round(ic_ir, 4)

    # 归一化权重
    total = sum(selected_weights.values())
    if total > 0:
        selected_weights = {k: round(v / total, 4) for k, v in selected_weights.items()}

    return StrategyCandidate(
        name=f"greedy_{len(selected)}f",
        factors=selected,
        weights=selected_weights,
        method="greedy",
        ic_ir_composite=sum(factor_stats[f]["ic_ir"] for f in selected) / max(len(selected), 1),
        metadata={"horizon": horizon, "max_corr": max_corr},
    )


def regularized_optimize(
    df: pd.DataFrame,
    factor_cols: list[str],
    horizon: int = 12,
    method: str = "lasso",
) -> StrategyCandidate:
    """正则化回归优化。

    使用 LassoCV 或 RidgeCV 拟合因子 z-score → 前瞻收益，
    回归系数即为因子权重。

    Args:
        df: 含因子列和 close 列的 DataFrame。
        factor_cols: 候选因子列名列表。
        horizon: 前瞻周期。
        method: "lasso" 或 "ridge"。

    Returns:
        StrategyCandidate 实例。
    """
    from sklearn.linear_model import LassoCV, RidgeCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    fwd_ret = df["close"].pct_change(horizon).shift(-horizon)

    # 构建特征矩阵
    available_cols = [c for c in factor_cols if c in df.columns]
    X = df[available_cols].copy()
    y = fwd_ret.copy()

    # 去掉缺失值
    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]
    y = y[valid]

    if len(X) < 100:
        logger.warning(f"Too few samples ({len(X)}) for regularized optimization")
        return StrategyCandidate(
            name=f"{method}_empty", factors=[], weights={}, method=method
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # TimeSeriesSplit 替代 K-Fold，防止时间序列数据泄露
    tscv = TimeSeriesSplit(n_splits=5)
    if method == "lasso":
        model = LassoCV(cv=tscv, max_iter=10000, random_state=42)
    else:
        model = RidgeCV(cv=tscv)

    model.fit(X_scaled, y)
    coefs = model.coef_

    # 提取非零系数因子
    selected = []
    weights = {}
    for col, coef in zip(available_cols, coefs):
        if abs(coef) > 1e-6:
            selected.append(col)
            weights[col] = round(float(abs(coef)), 6)

    # 归一化
    total = sum(weights.values())
    if total > 0:
        weights = {k: round(v / total, 4) for k, v in weights.items()}

    return StrategyCandidate(
        name=f"{method}_{len(selected)}f",
        factors=selected,
        weights=weights,
        method=method,
        metadata={
            "horizon": horizon,
            "alpha": float(model.alpha_) if hasattr(model, "alpha_") else None,
            "n_samples": len(X),
        },
    )


def cluster_select(
    df: pd.DataFrame,
    factor_cols: list[str],
    horizon: int = 12,
    n_clusters: int = 6,
    use_lasso: bool = False,
) -> StrategyCandidate:
    """层次聚类选择。

    计算因子间相关性矩阵 → 层次聚类 → 每个 cluster 选 IC IR 最高代表 → greedy 或 lasso。

    Args:
        df: 含因子列和 close 列的 DataFrame。
        factor_cols: 候选因子列名列表。
        horizon: 前瞻周期。
        n_clusters: 聚类数。
        use_lasso: 聚类后用 lasso 而非 greedy 优化权重。

    Returns:
        StrategyCandidate 实例。
    """
    available_cols = [c for c in factor_cols if c in df.columns]
    if len(available_cols) < 3:
        return greedy_forward(df, available_cols, horizon)

    # 相关性矩阵
    corr_matrix = df[available_cols].corr().fillna(0)
    dist_matrix = 1 - corr_matrix.abs()

    # 层次聚类
    condensed = dist_matrix.values[np.triu_indices(len(dist_matrix), k=1)]
    Z = linkage(condensed, method="ward")
    clusters = fcluster(Z, t=n_clusters, criterion="maxclust")

    # 每个 cluster 的因子
    cluster_map: dict[int, list[str]] = {}
    for col, cl in zip(available_cols, clusters):
        cluster_map.setdefault(int(cl), []).append(col)

    # 计算各因子 IC IR
    factor_stats = _calc_factor_stats(df, available_cols, horizon)

    # 每个 cluster 选 IC IR 最高的代表
    representatives = []
    for cl_id, cols in sorted(cluster_map.items()):
        best_col = max(cols, key=lambda c: factor_stats.get(c, {}).get("ic_ir", 0))
        ic_ir = factor_stats.get(best_col, {}).get("ic_ir", 0)
        if ic_ir > 0:
            representatives.append(best_col)
            logger.info(
                f"Cluster {cl_id}: selected {best_col} (IC IR={ic_ir:.3f}) "
                f"from {cols}"
            )

    if use_lasso and len(representatives) >= 3:
        return regularized_optimize(df, representatives, horizon, method="lasso")

    return greedy_forward(df, representatives, horizon, max_corr=0.9, min_ic_ir=0.0)


def _ensure_diversity(candidates: list[StrategyCandidate], max_overlap: float = 0.5) -> list[StrategyCandidate]:
    """过滤候选策略，确保 top-N 间因子重叠 < max_overlap。

    Args:
        candidates: 候选策略列表（已按质量排序）。
        max_overlap: 最大允许因子重叠率。

    Returns:
        过滤后的候选列表。
    """
    if len(candidates) <= 1:
        return candidates

    selected = [candidates[0]]
    for cand in candidates[1:]:
        cand_set = set(cand.factors)
        too_similar = False
        for existing in selected:
            existing_set = set(existing.factors)
            if not cand_set or not existing_set:
                continue
            overlap = len(cand_set & existing_set) / min(len(cand_set), len(existing_set))
            if overlap >= max_overlap:
                too_similar = True
                break
        if not too_similar:
            selected.append(cand)

    return selected


def _build_scoring_rules(candidate: StrategyCandidate) -> list[dict]:
    """为候选策略生成 scoring_rules，处理反转因子的方向。

    Args:
        candidate: 策略候选。

    Returns:
        scoring_rules 列表，每个 rule 含 name/column/long_when/short_when。
    """
    rules = []
    for col in candidate.factors:
        is_flipped = col in candidate.flipped
        # 默认：正值利多
        long_when = "> {z_threshold}"
        short_when = "< -{z_threshold}"
        if is_flipped:
            # 翻转：原始负值利多 → 实际上 low value = long
            long_when = "< -{z_threshold}"
            short_when = "> {z_threshold}"
        rules.append({
            "name": col,
            "column": col,
            "long_when": long_when,
            "short_when": short_when,
            "default_threshold": 1.5,
            "optional": True,
            "flipped": is_flipped,
        })
    return rules


def save_candidate_yaml(candidate: StrategyCandidate, output_dir: Path | None = None) -> Path:
    """将候选策略保存为 YAML profile。

    Args:
        candidate: 策略候选。
        output_dir: 输出目录，默认 strategies/auto/。

    Returns:
        输出文件路径。
    """
    import yaml

    if output_dir is None:
        output_dir = STRATEGIES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    scoring_rules = _build_scoring_rules(candidate)

    profile = {
        "name": candidate.name,
        "description": f"Auto-generated by {candidate.method} combo search (flip_negative)",
        "version": "0.1.0",
        "instruments": ["BTC", "ETH"],
        "scoring_mode": "continuous",
        "ic_weights": candidate.weights,
        "factors_selected": candidate.factors,
        "flipped_factors": sorted(candidate.flipped),
        "scoring_rules_generated": scoring_rules,
        "metadata": {
            "method": candidate.method,
            "ic_ir_composite": candidate.ic_ir_composite,
            "diversity_score": candidate.diversity_score,
            **candidate.metadata,
        },
        "risk": {
            "max_leverage": 10,
            "max_loss_per_trade": 0.02,
            "max_positions_per_symbol": 1,
            "max_position_pct": 0.95,
        },
        "strategy": {
            "signal": {
                "min_score": 3,
                "cooldown_bars": 12,
                "entry_percentile": 0.95,
            },
        },
    }

    path = output_dir / f"{candidate.name}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(profile, f, allow_unicode=True, sort_keys=False)

    logger.info(f"Saved candidate strategy to {path}")
    return path


def run_combo_search(
    df: pd.DataFrame,
    factor_cols: list[str],
    method: str = "greedy",
    top_n: int = 10,
    horizon: int = 12,
    days: int = 90,
    flip_negative: bool = False,
) -> list[StrategyCandidate]:
    """运行组合搜索主流程。

    Args:
        df: 含所有因子和 close 列的 DataFrame。
        factor_cols: 候选因子列名列表。
        method: 搜索方法 ("greedy"|"lasso"|"ridge"|"cluster")。
        top_n: 输出 top-N 策略。
        horizon: 前瞻周期。
        days: 数据天数（仅记录用）。
        flip_negative: 自动翻转负 IC 因子。

    Returns:
        top-N StrategyCandidate 列表。
    """
    flipped: set[str] = set()
    if flip_negative:
        logger.info("Auto-flipping negative IC factors...")
        df, flipped = auto_flip_factors(df, factor_cols, horizon)
        logger.info(f"Flipped {len(flipped)} factors: {sorted(flipped)}")

    logger.info(f"Running combo search: method={method}, factors={len(factor_cols)}, horizon={horizon}")

    candidates: list[StrategyCandidate] = []

    if method == "greedy":
        for max_corr in [0.5, 0.6, 0.7, 0.8]:
            for min_ir in [0.2, 0.3, 0.5]:
                cand = greedy_forward(
                    df, factor_cols, horizon,
                    max_corr=max_corr, min_ic_ir=min_ir,
                )
                if cand.factors:
                    cand.name = f"greedy_c{max_corr}_ir{min_ir}_{len(cand.factors)}f"
                    cand.flipped = flipped & set(cand.factors)
                    candidates.append(cand)

    elif method in ("lasso", "ridge"):
        cand = regularized_optimize(df, factor_cols, horizon, method=method)
        if cand.factors:
            cand.flipped = flipped & set(cand.factors)
            candidates.append(cand)

    elif method == "cluster":
        for n_cl in [4, 6, 8]:
            for use_lasso in [False, True]:
                cand = cluster_select(df, factor_cols, horizon, n_clusters=n_cl, use_lasso=use_lasso)
                if cand.factors:
                    suffix = "lasso" if use_lasso else "greedy"
                    cand.name = f"cluster_{n_cl}c_{suffix}_{len(cand.factors)}f"
                    cand.flipped = flipped & set(cand.factors)
                    candidates.append(cand)

    # 去重 + 多样性过滤
    candidates.sort(key=lambda c: -c.ic_ir_composite)
    candidates = _ensure_diversity(candidates, max_overlap=0.5)

    # 截取 top-N
    result = candidates[:top_n]

    # 保存每个候选的 YAML
    for cand in result:
        save_candidate_yaml(cand)

    logger.info(f"Combo search complete: {len(result)} candidates saved")
    return result
