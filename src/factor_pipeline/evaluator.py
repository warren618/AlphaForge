"""单因子评测器。

专业级因子评测：滚动窗口 Rank IC、IC IR、IC 衰减 half-life、
Newey-West t-stat、Benjamini-Hochberg FDR、条件收益 t-test。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .registry import FactorRegistry
from src.logging_config import get_logger

logger = get_logger("evaluator")


class FactorEvaluator:
    """单因子评测器。

    计算滚动 Rank IC (Spearman)、IC IR、IC 衰减曲线 + half-life、
    Newey-West t-stat、触发率、换手率、条件收益 + t-test、因子间相关性。

    Attributes:
        registry: 因子注册表。
        horizons: 评测前瞻周期列表（单位: bars）。

    Example:
        >>> evaluator = FactorEvaluator(registry)
        >>> report = evaluator.evaluate("basis_dislocation", df, cfg, extra_data)
    """

    def __init__(self, registry: FactorRegistry, horizons: list[int] | None = None):
        """初始化评测器。

        Args:
            registry: 已完成自动发现的因子注册表。
            horizons: 前瞻周期列表，默认 [6, 12, 24, 48] bars。
        """
        self.registry = registry
        self.horizons = horizons or [6, 12, 24, 48]

    def evaluate(
        self,
        factor_name: str,
        df: pd.DataFrame,
        cfg: dict,
        extra_data: dict | None = None,
        z_threshold: float | None = None,
        ic_window: int = 252,
    ) -> dict:
        """对单因子运行全套评测。

        当 z_threshold=None（默认）时自动检测合适阈值：
        - 离散因子（unique < 10）: threshold = 0.5
        - 连续因子 std < 1.0: 绝对值 75th percentile
        - 连续因子 std >= 1.0: 经典 z_threshold = 1.5

        Args:
            factor_name: 因子名称。
            df: 含 OHLCV 的主 DataFrame（需有 close 列）。
            cfg: 因子参数配置。
            extra_data: 额外数据源。
            z_threshold: 触发阈值。None 时自动检测。
            ic_window: 滚动 IC 窗口大小（默认 252 bars）。

        Returns:
            评测报告字典，含 ts_rank_ic、IC IR、t_stat、nw_pvalue、half_life 等指标。
        """
        factor = self.registry.get(factor_name)
        df = factor.compute(df.copy(), cfg, extra_data)
        col = factor.meta.score_column

        if col not in df.columns:
            return {"name": factor_name, "error": f"Column '{col}' not found after compute"}

        report: dict = {"name": factor_name, "score_column": col}
        factor_values = df[col].astype(float)

        # 自适应阈值检测
        if z_threshold is None:
            z_threshold = _detect_adaptive_threshold(factor_values)
        report["adaptive_threshold"] = round(z_threshold, 4)

        # 0.1.1: 滚动窗口 Rank IC + IC series per horizon
        ic_series_by_horizon: dict[int, pd.Series] = {}

        for h in self.horizons:
            fwd_ret = df["close"].pct_change(h).shift(-h)
            valid = factor_values.notna() & fwd_ret.notna()
            n_valid = valid.sum()

            if n_valid < 30:
                report[f"ts_rank_ic_{h}"] = None
                report[f"ic_ir_{h}"] = None
                report[f"t_stat_{h}"] = None
                report[f"nw_pvalue_{h}"] = None
                continue

            # 滚动窗口 Spearman IC
            window = min(ic_window, n_valid // 2)
            if window < 30:
                window = 30

            fv = factor_values[valid].reset_index(drop=True)
            fr = fwd_ret[valid].reset_index(drop=True)

            ic_list = []
            for i in range(window, len(fv)):
                chunk_f = fv.iloc[i - window:i]
                chunk_r = fr.iloc[i - window:i]
                ic, _ = sp_stats.spearmanr(chunk_f, chunk_r)
                ic_list.append(ic)

            ic_series = pd.Series(ic_list)
            ic_series_by_horizon[h] = ic_series

            if len(ic_series) < 5:
                report[f"ts_rank_ic_{h}"] = None
                report[f"ic_ir_{h}"] = None
                report[f"t_stat_{h}"] = None
                report[f"nw_pvalue_{h}"] = None
                continue

            # 0.1.1: 时序均值 Rank IC
            ts_rank_ic = float(ic_series.mean())
            report[f"ts_rank_ic_{h}"] = round(ts_rank_ic, 6)

            # 0.1.2: IC IR = mean(ic_series) / std(ic_series)
            ic_std = float(ic_series.std())
            ic_ir = ts_rank_ic / ic_std if ic_std > 0 else 0.0
            report[f"ic_ir_{h}"] = round(ic_ir, 4)

            # 0.1.3: t-stat + Newey-West p-value
            t_stat, nw_pvalue = _newey_west_tstat(ic_series.values)
            report[f"t_stat_{h}"] = round(t_stat, 4) if t_stat is not None else None
            report[f"nw_pvalue_{h}"] = round(nw_pvalue, 6) if nw_pvalue is not None else None

        # 向后兼容：保留 rank_ic_X 字段（等于 ts_rank_ic）
        for h in self.horizons:
            report[f"rank_ic_{h}"] = report.get(f"ts_rank_ic_{h}")

        # 0.1.5: IC 衰减曲线 + half-life
        ic_decay = []
        for h in range(1, 49):
            fwd = df["close"].pct_change(h).shift(-h)
            valid = factor_values.notna() & fwd.notna()
            if valid.sum() < 30:
                ic_decay.append(None)
                continue
            ic, _ = sp_stats.spearmanr(factor_values[valid], fwd[valid])
            ic_decay.append(round(float(ic), 6))
        report["ic_decay"] = ic_decay

        # 拟合 IC(h) = IC_0 * exp(-h/tau)，half_life = tau * ln(2)
        report["half_life"] = _fit_ic_halflife(ic_decay)

        # 触发率
        report["trigger_rate_1.0"] = round(float((factor_values.abs() > 1.0).mean()), 4)
        report["trigger_rate_1.5"] = round(float((factor_values.abs() > z_threshold).mean()), 4)
        report["trigger_rate_2.0"] = round(float((factor_values.abs() > 2.0).mean()), 4)

        # Turnover
        signal = (factor_values > z_threshold).astype(int) - (factor_values < -z_threshold).astype(int)
        report["turnover"] = round(float((signal.diff().abs() > 0).mean()), 4)

        # 0.1.6: 条件收益 + t-test
        fwd_1h = df["close"].pct_change(12).shift(-12)
        long_mask = factor_values > z_threshold
        short_mask = factor_values < -z_threshold
        neutral_mask = ~long_mask & ~short_mask

        long_ret = fwd_1h[long_mask].dropna()
        short_ret = fwd_1h[short_mask].dropna()
        neutral_ret = fwd_1h[neutral_mask].dropna()

        report["long_return_1h"] = _safe_mean(long_ret)
        report["short_return_1h"] = _safe_mean(short_ret)
        report["neutral_return_1h"] = _safe_mean(neutral_ret)
        report["long_count"] = int(long_mask.sum())
        report["short_count"] = int(short_mask.sum())

        # t-test: long vs neutral, short vs neutral
        report["long_vs_neutral_tstat"], report["long_vs_neutral_pvalue"] = _safe_ttest(long_ret, neutral_ret)
        report["short_vs_neutral_tstat"], report["short_vs_neutral_pvalue"] = _safe_ttest(short_ret, neutral_ret)

        # 因子间相关性
        correlations = {}
        for other_name in self.registry.list_all():
            if other_name == factor_name:
                continue
            other_factor = self.registry.get(other_name)
            other_col = other_factor.meta.score_column
            if other_col in df.columns:
                valid = factor_values.notna() & df[other_col].notna()
                if valid.sum() >= 30:
                    corr_val = factor_values[valid].corr(df[other_col][valid].astype(float))
                    correlations[other_name] = round(float(corr_val), 4)
        report["correlations"] = correlations

        # 基础统计
        report["mean"] = round(float(factor_values.mean()), 6)
        report["std"] = round(float(factor_values.std()), 6)
        report["skew"] = round(float(factor_values.skew()), 4)
        report["kurtosis"] = round(float(factor_values.kurtosis()), 4)

        return report

    def save_report(self, report: dict, output_dir: str | Path | None = None):
        """保存评测报告到 JSON。

        Args:
            report: evaluate() 返回的报告字典。
            output_dir: 输出目录，默认 factor_library/eval_results/。
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "factor_library" / "eval_results"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        name = report.get("name", "unknown")
        filepath = output_dir / f"{name}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        return filepath

    def generate_summary(self, reports: list[dict], sort_by: str = "ts_rank_ic_12") -> pd.DataFrame:
        """汇总多因子评测结果为排名表，含 FDR 校正。

        Args:
            reports: evaluate() 报告列表。
            sort_by: 排序指标列名。

        Returns:
            汇总排名 DataFrame。
        """
        rows = []
        pvalues_12 = []

        for r in reports:
            row = {
                "name": r.get("name"),
                "ts_rank_ic_6": r.get("ts_rank_ic_6"),
                "ts_rank_ic_12": r.get("ts_rank_ic_12"),
                "ts_rank_ic_24": r.get("ts_rank_ic_24"),
                "ts_rank_ic_48": r.get("ts_rank_ic_48"),
                "ic_ir_12": r.get("ic_ir_12"),
                "ic_ir_24": r.get("ic_ir_24"),
                "t_stat_12": r.get("t_stat_12"),
                "nw_pvalue_12": r.get("nw_pvalue_12"),
                "half_life": r.get("half_life"),
                "adaptive_threshold": r.get("adaptive_threshold"),
                "trigger_1.5": r.get("trigger_rate_1.5"),
                "turnover": r.get("turnover"),
                "long_ret_1h": r.get("long_return_1h"),
                "short_ret_1h": r.get("short_return_1h"),
                "long_vs_neutral_p": r.get("long_vs_neutral_pvalue"),
            }
            rows.append(row)
            pvalues_12.append(r.get("nw_pvalue_12"))

        summary = pd.DataFrame(rows)

        # 0.1.4: Benjamini-Hochberg FDR
        valid_pvalues = [p for p in pvalues_12 if p is not None]
        if valid_pvalues:
            from statsmodels.stats.multitest import multipletests
            # 用原始位置做映射
            pv_indices = [i for i, p in enumerate(pvalues_12) if p is not None]
            reject, pvals_corrected, _, _ = multipletests(
                valid_pvalues, method="fdr_bh", alpha=0.05
            )
            fdr_col = [None] * len(pvalues_12)
            fdr_sig_col = [None] * len(pvalues_12)
            for idx, (rej, pv_corr) in zip(pv_indices, zip(reject, pvals_corrected)):
                fdr_col[idx] = round(float(pv_corr), 6)
                fdr_sig_col[idx] = bool(rej)
            summary["fdr_pvalue_12"] = fdr_col
            summary["fdr_significant"] = fdr_sig_col

        # 向后兼容 sort
        if sort_by == "rank_ic_12":
            sort_by = "ts_rank_ic_12"
        if sort_by in summary.columns:
            summary = summary.sort_values(sort_by, ascending=False, na_position="last")

        return summary


def _detect_adaptive_threshold(values: pd.Series) -> float:
    """根据因子值分布自动选择触发阈值。

    规则:
    - 离散因子（unique values < 10）: 0.5
    - 连续因子 std < 1.0: 绝对值 75th percentile
    - 连续因子 std >= 1.0: 经典 1.5

    Args:
        values: 因子值 Series。

    Returns:
        适合该因子的触发阈值。
    """
    clean = values.dropna()
    if len(clean) < 10:
        return 1.5

    n_unique = clean.nunique()
    if n_unique < 10:
        return 0.5

    std = float(clean.std())
    if std < 1.0:
        threshold = float(np.percentile(clean.abs(), 75))
        # 安全下界：至少 0.01 避免全触发
        return max(threshold, 0.01)

    return 1.5


def _newey_west_tstat(ic_array: np.ndarray) -> tuple[float | None, float | None]:
    """计算 IC 序列的 Newey-West t-stat 和 p-value。

    Args:
        ic_array: IC 值数组。

    Returns:
        (t_stat, p_value) 元组。
    """
    try:
        import statsmodels.api as sm

        n = len(ic_array)
        if n < 10:
            return None, None

        X = sm.add_constant(np.ones(n))
        model = sm.OLS(ic_array, X)
        maxlags = int(n ** (1 / 3))
        results = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

        t_stat = float(results.tvalues[0])
        p_value = float(results.pvalues[0])
        return t_stat, p_value
    except Exception as e:
        logger.debug(f"Newey-West failed: {e}")
        return None, None


def _fit_ic_halflife(ic_decay: list[float | None]) -> float | None:
    """拟合 IC 衰减的 half-life。

    对 ic_decay[0..47] 拟合 IC(h) = IC_0 * exp(-h/tau)，
    half_life = tau * ln(2)。

    Args:
        ic_decay: IC 衰减列表（1-48 bars）。

    Returns:
        half_life (bars) 或 None。
    """
    valid_h = []
    valid_ic = []
    for i, ic in enumerate(ic_decay):
        if ic is not None and ic > 0:
            valid_h.append(i + 1)
            valid_ic.append(ic)

    if len(valid_h) < 5:
        return None

    try:
        log_ic = np.log(np.array(valid_ic))
        h_arr = np.array(valid_h, dtype=float)

        # 线性回归: log(IC) = log(IC_0) - h/tau
        slope, intercept, _, _, _ = sp_stats.linregress(h_arr, log_ic)
        if slope >= 0:
            return None  # IC 不衰减

        tau = -1.0 / slope
        half_life = tau * np.log(2)
        return round(float(half_life), 2) if half_life > 0 else None
    except Exception:
        return None


def _safe_mean(series: pd.Series) -> float | None:
    """安全计算均值。

    Args:
        series: 数值 Series。

    Returns:
        均值或 None（数据不足时）。
    """
    valid = series.dropna()
    if len(valid) < 5:
        return None
    return round(float(valid.mean()), 6)


def _safe_ttest(group_a: pd.Series, group_b: pd.Series) -> tuple[float | None, float | None]:
    """安全执行双样本 t-test。

    Args:
        group_a: 第一组样本。
        group_b: 第二组样本。

    Returns:
        (t_stat, p_value) 元组。
    """
    a = group_a.dropna()
    b = group_b.dropna()
    if len(a) < 10 or len(b) < 10:
        return None, None
    try:
        t_stat, p_value = sp_stats.ttest_ind(a, b, equal_var=False)
        return round(float(t_stat), 4), round(float(p_value), 6)
    except Exception:
        return None, None
