"""OKX数据获取引擎。

通过REST API批量拉取永续合约K线数据及衍生品微观结构数据，支持分页获取长历史。
V10新增：资金费率、持仓量、多空账户比、主动买卖量、指数K线。
"""

import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from src.logging_config import get_logger

logger = get_logger("data_fetcher")

BASE_URL = "https://www.okx.com"
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
VALID_CACHE_MODES = {"prefer_local", "local_only", "refresh"}


def validate_ohlcv(df: pd.DataFrame, max_gap_ratio: float = 0.05, max_nan_ratio: float = 0.1) -> dict:
    """OHLCV 数据质量检查。

    Args:
        df: OHLCV DataFrame，需含 ts, open, high, low, close 列。
        max_gap_ratio: 最大允许时间间隙比例。
        max_nan_ratio: 最大允许 NaN 比例。

    Returns:
        质量报告字典，含 passed, issues 列表。

    Example:
        >>> report = validate_ohlcv(df)
        >>> if not report["passed"]:
        ...     logger.warning(f"Data quality issues: {report['issues']}")
    """
    issues = []

    if df.empty:
        return {"passed": False, "issues": ["Empty DataFrame"]}

    # 检查必需列
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        return {"passed": False, "issues": [f"Missing columns: {missing}"]}

    n = len(df)

    # 1. 时间戳间隙
    if "ts" in df.columns and n > 1:
        ts_sorted = df["ts"].sort_values()
        diffs = ts_sorted.diff().dropna()
        median_diff = diffs.median()
        if median_diff > 0:
            gaps = (diffs > median_diff * 2).sum()
            gap_ratio = gaps / n
            if gap_ratio > max_gap_ratio:
                issues.append(f"Time gaps: {gaps}/{n} ({gap_ratio:.1%})")

    # 2. 重复行
    if "ts" in df.columns:
        n_dupes = df["ts"].duplicated().sum()
        if n_dupes > 0:
            issues.append(f"Duplicate timestamps: {n_dupes}")

    # 3. OHLC 逻辑: high >= max(open, close), low <= min(open, close)
    ohlc_invalid = (
        (df["high"] < df["open"]) | (df["high"] < df["close"]) |
        (df["low"] > df["open"]) | (df["low"] > df["close"])
    ).sum()
    if ohlc_invalid > 0:
        issues.append(f"OHLC logic violations: {ohlc_invalid}")

    # 4. NaN 比例
    for col in ["open", "high", "low", "close"]:
        nan_ratio = df[col].isna().mean()
        if nan_ratio > max_nan_ratio:
            issues.append(f"High NaN ratio in {col}: {nan_ratio:.1%}")

    # 5. 零值/负值
    for col in ["open", "high", "low", "close"]:
        n_invalid = (df[col] <= 0).sum()
        if n_invalid > 0:
            issues.append(f"Non-positive values in {col}: {n_invalid}")

    passed = len(issues) == 0
    if not passed:
        logger.warning(f"Data quality issues ({n} bars): {issues}")
    else:
        logger.debug(f"Data quality check passed ({n} bars)")

    return {"passed": passed, "issues": issues, "n_bars": n}


def fetch_candles(
    inst_id: str,
    bar: str = "5m",
    limit: int = 300,
    after: Optional[str] = None,
    before: Optional[str] = None,
) -> list[list[str]]:
    """获取单批K线数据。

    Args:
        inst_id: 合约ID，如 'BTC-USDT-SWAP'。
        bar: K线周期，如 '5m', '15m', '1H'。
        limit: 单次最大条数（OKX上限300）。
        after: 请求此时间戳之前的数据（分页用）。
        before: 请求此时间戳之后的数据。

    Returns:
        K线数据列表，每条为 [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]。
    """
    params = {"instId": inst_id, "bar": bar, "limit": str(min(limit, 300))}
    if after:
        params["after"] = after
    if before:
        params["before"] = before

    url = f"{BASE_URL}/api/v5/market/history-candles"
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=30)
            data = resp.json()
            if data["code"] != "0":
                raise RuntimeError(f"OKX API error: {data['code']} - {data['msg']}")
            return data.get("data", [])
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def fetch_candles_batch(
    inst_id: str,
    bar: str = "5m",
    days: int = 60,
    cache_mode: str = "prefer_local",
    cache_ttl_hours: Optional[float] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """批量分页获取历史K线，自动缓存。

    Args:
        inst_id: 合约ID。
        bar: K线周期。
        days: 获取最近多少天的数据。
        cache_mode: 缓存模式。
            - prefer_local: 优先本地缓存，缓存不存在/过期才拉取网络。
            - local_only: 只读本地缓存，不访问网络。
            - refresh: 强制从网络拉取并覆盖缓存。
        cache_ttl_hours: 本地缓存有效期（小时），None 表示永不过期。
        use_cache: 兼容旧接口，False 等价于 cache_mode=refresh。

    Returns:
        包含 datetime, open, high, low, close, volume, volCcyQuote 的DataFrame。

    Raises:
        ValueError: 当 cache_mode 非法时抛出。
        FileNotFoundError: local_only 且本地缓存不存在时抛出。

    Example:
        >>> df = fetch_candles_batch("BTC-USDT-SWAP", bar="5m", days=60, cache_mode="local_only")
    """
    if not use_cache:
        cache_mode = "refresh"
    if cache_mode not in VALID_CACHE_MODES:
        raise ValueError(
            f"Invalid cache_mode: {cache_mode}. "
            f"Expected one of: {sorted(VALID_CACHE_MODES)}"
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{inst_id}_{bar}_{days}d.parquet"
    stale_cache_df = None

    if cache_file.exists():
        try:
            stale_cache_df = pd.read_parquet(cache_file)
        except Exception:
            print(f"    WARN: corrupted cache file, removing: {cache_file.name}")
            cache_file.unlink(missing_ok=True)
            stale_cache_df = None
        if stale_cache_df is not None:
            cache_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            if cache_mode == "local_only":
                return stale_cache_df
            if cache_mode == "prefer_local":
                if cache_ttl_hours is None or cache_age_hours <= cache_ttl_hours:
                    return stale_cache_df

    if cache_mode == "local_only":
        raise FileNotFoundError(
            f"Local cache not found: {cache_file}. "
            f"Run once with cache_mode=refresh or prefer_local to create it."
        )

    target_ts = int((time.time() - days * 86400) * 1000)
    all_data = []
    after_ts = None
    max_requests = 500

    try:
        for _ in range(max_requests):
            batch = fetch_candles(inst_id, bar, limit=300, after=after_ts)
            if not batch:
                break
            all_data.extend(batch)
            oldest_ts = int(batch[-1][0])
            if oldest_ts <= target_ts:
                break
            after_ts = str(oldest_ts)
            time.sleep(0.12)
    except Exception:
        if stale_cache_df is not None and cache_mode == "prefer_local":
            print(
                f"    WARN: network fetch failed, fallback to stale local cache: "
                f"{cache_file.name}"
            )
            return stale_cache_df
        raise

    if not all_data:
        if stale_cache_df is not None and cache_mode == "prefer_local":
            print(
                f"    WARN: empty network response, fallback to stale local cache: "
                f"{cache_file.name}"
            )
            return stale_cache_df
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        "ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"
    ])
    for col in ["open", "high", "low", "close", "vol", "volCcy", "volCcyQuote"]:
        df[col] = df[col].astype(float)
    df["ts"] = df["ts"].astype(np.int64)
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.sort_values("ts").reset_index(drop=True)
    df = df[df["ts"] >= target_ts].reset_index(drop=True)
    df = df.drop_duplicates(subset=["ts"]).reset_index(drop=True)

    tmp_file = cache_file.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_file, index=False)
    tmp_file.replace(cache_file)
    return df


def fetch_all_instruments(
    instruments: list[dict],
    bar: str = "5m",
    days: int = 60,
    cache_mode: str = "prefer_local",
    cache_ttl_hours: Optional[float] = None,
) -> dict[str, pd.DataFrame]:
    """批量获取多个标的的K线数据。

    Args:
        instruments: 标的配置列表，每项含 instId 和 alias。
        bar: K线周期。
        days: 天数。
        cache_mode: 缓存模式，传给 ``fetch_candles_batch``。
        cache_ttl_hours: 缓存有效期（小时），None 表示永不过期。

    Returns:
        {alias: DataFrame} 的字典。

    Raises:
        RuntimeError: local_only 且某标的缓存不存在时抛出。

    Example:
        >>> data = fetch_all_instruments(instruments, bar="15m", days=60, cache_mode="prefer_local")
    """
    result = {}
    for inst in instruments:
        inst_id = inst["instId"]
        alias = inst["alias"]
        print(f"  Fetching {alias} ({inst_id}) {bar} x {days}d [mode={cache_mode}] ...")
        try:
            df = fetch_candles_batch(
                inst_id=inst_id,
                bar=bar,
                days=days,
                cache_mode=cache_mode,
                cache_ttl_hours=cache_ttl_hours,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Missing local cache for {alias} ({inst_id}) under local_only mode."
            ) from exc
        if df.empty:
            print("    → 0 bars (empty dataset)")
        else:
            print(f"    → {len(df)} bars, {df['datetime'].min()} to {df['datetime'].max()}")
        result[alias] = df
    return result


# ============================================================
# V10: OKX 公共 API 通用请求
# ============================================================

def _okx_get(url: str, params: dict, max_retries: int = 3, sleep_sec: float = 0.12) -> dict:
    """OKX REST API 通用请求，含重试。

    Args:
        url: 完整 API URL。
        params: 请求参数字典。
        max_retries: 最大重试次数。
        sleep_sec: 请求间隔（秒）。

    Returns:
        响应 JSON 字典。

    Raises:
        RuntimeError: API 返回非 0 code。
    """
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            data = resp.json()
            if data.get("code") != "0":
                raise RuntimeError(f"OKX API error: {data.get('code')} - {data.get('msg')}")
            return data
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


def _paginated_fetch(
    url: str,
    base_params: dict,
    days: int,
    items_per_batch: int,
    sleep_sec: float,
    ts_index: int = 0,
    max_requests: int = 500,
) -> list[list[str]]:
    """通用分页获取（按时间戳向前翻页）。

    Args:
        url: API URL。
        base_params: 基础请求参数（不含分页参数）。
        days: 获取最近多少天。
        items_per_batch: 预估每批数据量（用于判断分页是否结束）。
        sleep_sec: 请求间隔。
        ts_index: 时间戳在返回数组中的位置索引。
        max_requests: 最大请求数。

    Returns:
        所有数据行的列表。
    """
    target_ts = int((time.time() - days * 86400) * 1000)
    all_data = []
    after_ts = None

    for _ in range(max_requests):
        params = dict(base_params)
        if after_ts:
            params["after"] = str(after_ts)

        data = _okx_get(url, params, sleep_sec=sleep_sec)
        batch = data.get("data", [])
        if not batch:
            break

        all_data.extend(batch)
        oldest_ts = int(batch[-1][ts_index])
        if oldest_ts <= target_ts:
            break
        after_ts = oldest_ts
        time.sleep(sleep_sec)

    return all_data


def _cache_wrapper(
    cache_file: Path,
    cache_mode: str,
    cache_ttl_hours: Optional[float],
    fetch_fn,
) -> pd.DataFrame:
    """通用缓存包装器。

    Args:
        cache_file: 缓存文件路径。
        cache_mode: prefer_local / local_only / refresh。
        cache_ttl_hours: 缓存有效期。
        fetch_fn: 无参数的获取函数，返回 pd.DataFrame。

    Returns:
        数据 DataFrame。
    """
    if cache_mode not in VALID_CACHE_MODES:
        raise ValueError(f"Invalid cache_mode: {cache_mode}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stale_df = None

    if cache_file.exists():
        try:
            stale_df = pd.read_parquet(cache_file)
        except Exception:
            print(f"    WARN: corrupted cache file, removing: {cache_file.name}")
            cache_file.unlink(missing_ok=True)
            stale_df = None
        if stale_df is not None:
            if cache_mode == "local_only":
                return stale_df
            if cache_mode == "prefer_local":
                cache_age = (time.time() - cache_file.stat().st_mtime) / 3600
                if cache_ttl_hours is None or cache_age <= cache_ttl_hours:
                    return stale_df

    if cache_mode == "local_only":
        raise FileNotFoundError(f"Local cache not found: {cache_file}")

    try:
        df = fetch_fn()
    except Exception:
        if stale_df is not None and cache_mode == "prefer_local":
            print(f"    WARN: fetch failed, fallback to stale cache: {cache_file.name}")
            return stale_df
        raise

    if df.empty:
        if stale_df is not None and cache_mode == "prefer_local":
            print(f"    WARN: empty response, fallback to stale cache: {cache_file.name}")
            return stale_df
        return df

    tmp_file = cache_file.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_file, index=False)
    tmp_file.replace(cache_file)
    return df


# ============================================================
# V10: 资金费率历史
# ============================================================

def fetch_funding_rate_history(
    inst_id: str,
    days: int = 90,
    cache_mode: str = "prefer_local",
    cache_ttl_hours: Optional[float] = None,
) -> pd.DataFrame:
    """获取资金费率历史。

    Args:
        inst_id: 合约 ID，如 'BTC-USDT-SWAP'。
        days: 获取最近多少天。
        cache_mode: 缓存模式。
        cache_ttl_hours: 缓存有效期。

    Returns:
        含 datetime, ts, fundingRate, realizedRate 的 DataFrame。

    Example:
        >>> df = fetch_funding_rate_history("BTC-USDT-SWAP", days=90)
    """
    cache_file = CACHE_DIR / f"{inst_id}_funding_rate_{days}d.parquet"

    def _fetch():
        url = f"{BASE_URL}/api/v5/public/funding-rate-history"
        target_ts = int((time.time() - days * 86400) * 1000)
        all_data = []
        after_ts = None

        for _ in range(200):
            params = {"instId": inst_id, "limit": "100"}
            if after_ts:
                params["after"] = str(after_ts)

            data = _okx_get(url, params, sleep_sec=0.12)
            batch = data.get("data", [])
            if not batch:
                break

            all_data.extend(batch)
            # funding rate 返回 dict，取 fundingTime 作为分页游标
            oldest_ts = int(batch[-1]["fundingTime"])
            if oldest_ts <= target_ts:
                break
            after_ts = oldest_ts
            time.sleep(0.12)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df["ts"] = df["fundingTime"].astype(np.int64)
        df["fundingRate"] = df["fundingRate"].astype(float)
        df["realizedRate"] = df["realizedRate"].astype(float)
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
        df = df[df["ts"] >= target_ts]
        df = df.sort_values("ts").reset_index(drop=True)
        df = df.drop_duplicates(subset=["ts"]).reset_index(drop=True)
        df = df[["datetime", "ts", "fundingRate", "realizedRate"]]
        return df

    return _cache_wrapper(cache_file, cache_mode, cache_ttl_hours, _fetch)


# ============================================================
# V10: Rubik 系列 API（OI / LS Ratio / Taker Volume）
# ============================================================

def _fetch_rubik_history(
    endpoint: str,
    ccy: str,
    period: str,
    days: int,
    columns: list[str],
    cache_tag: str,
    cache_mode: str = "prefer_local",
    cache_ttl_hours: Optional[float] = None,
) -> pd.DataFrame:
    """Rubik 系列 API 通用获取。

    Args:
        endpoint: API 路径（不含 BASE_URL）。
        ccy: 币种，如 'BTC'。
        period: 数据粒度，如 '5m'。
        days: 天数。
        columns: 返回数组的列名列表。
        cache_tag: 缓存文件名标识。
        cache_mode: 缓存模式。
        cache_ttl_hours: 缓存有效期。

    Returns:
        DataFrame。
    """
    cache_file = CACHE_DIR / f"{ccy}_{cache_tag}_{period}_{days}d.parquet"

    def _fetch():
        url = f"{BASE_URL}{endpoint}"
        target_ts = int((time.time() - days * 86400) * 1000)
        now_ts = int(time.time() * 1000)
        all_data = []

        # rubik API 用 begin+end 窗口滑动，每次请求一段时间的数据
        # 从当前时间向前滑动，每次取 1 天
        window_ms = 24 * 3600 * 1000  # 1 天
        cursor_end = now_ts

        consecutive_errors = 0
        for _ in range(days + 5):
            cursor_begin = cursor_end - window_ms
            if cursor_begin < target_ts:
                cursor_begin = target_ts

            params = {
                "instType": "SWAP",
                "ccy": ccy,
                "period": period,
                "begin": str(cursor_begin),
                "end": str(cursor_end),
            }

            try:
                data = _okx_get(url, params, sleep_sec=0.5)
                batch = data.get("data", [])
                if batch:
                    all_data.extend(batch)
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
            except RuntimeError:
                consecutive_errors += 1

            # 连续 3 次失败说明已到历史上限，停止翻页
            if consecutive_errors >= 3:
                break
            if cursor_begin <= target_ts:
                break
            cursor_end = cursor_begin
            time.sleep(0.5)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data, columns=columns)
        df["ts"] = df["ts"].astype(np.int64)
        for col in columns:
            if col != "ts":
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
        df = df[df["ts"] >= target_ts]
        df = df.sort_values("ts").reset_index(drop=True)
        df = df.drop_duplicates(subset=["ts"]).reset_index(drop=True)
        return df

    return _cache_wrapper(cache_file, cache_mode, cache_ttl_hours, _fetch)


def fetch_oi_history(
    inst_id: str,
    period: str = "5m",
    days: int = 60,
    cache_mode: str = "prefer_local",
    cache_ttl_hours: Optional[float] = None,
) -> pd.DataFrame:
    """获取持仓量历史。

    注意：此端点需要 instId（如 BTC-USDT-SWAP），非 ccy。

    Args:
        inst_id: 合约 ID，如 'BTC-USDT-SWAP'。
        period: 数据粒度。
        days: 天数。

    Returns:
        含 datetime, ts, oi 的 DataFrame。
    """
    alias = inst_id.split("-")[0]
    cache_file = CACHE_DIR / f"{alias}_oi_{period}_{days}d.parquet"

    def _fetch():
        url = f"{BASE_URL}/api/v5/rubik/stat/contracts/open-interest-history"
        target_ts = int((time.time() - days * 86400) * 1000)
        now_ts = int(time.time() * 1000)
        all_data = []
        window_ms = 24 * 3600 * 1000
        cursor_end = now_ts
        consecutive_errors = 0

        for _ in range(days + 5):
            cursor_begin = max(cursor_end - window_ms, target_ts)
            params = {
                "instId": inst_id,
                "period": period,
                "begin": str(cursor_begin),
                "end": str(cursor_end),
            }
            try:
                data = _okx_get(url, params, sleep_sec=0.5)
                batch = data.get("data", [])
                if batch:
                    all_data.extend(batch)
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
            except RuntimeError:
                consecutive_errors += 1

            if consecutive_errors >= 3:
                break
            if cursor_begin <= target_ts:
                break
            cursor_end = cursor_begin
            time.sleep(0.5)

        if not all_data:
            return pd.DataFrame()

        # [ts, oi_contracts, oi_coin, oi_usd]
        df = pd.DataFrame(all_data, columns=["ts", "oi", "oiCcy", "oiUsd"])
        df["ts"] = df["ts"].astype(np.int64)
        df["oi"] = pd.to_numeric(df["oi"], errors="coerce").fillna(0.0)
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
        df = df[df["ts"] >= target_ts]
        df = df.sort_values("ts").reset_index(drop=True)
        df = df.drop_duplicates(subset=["ts"]).reset_index(drop=True)
        df = df[["ts", "oi", "datetime"]]
        return df

    return _cache_wrapper(cache_file, cache_mode, cache_ttl_hours, _fetch)


def fetch_ls_ratio_history(
    ccy: str,
    period: str = "5m",
    days: int = 60,
    cache_mode: str = "prefer_local",
    cache_ttl_hours: Optional[float] = None,
) -> pd.DataFrame:
    """获取多空账户比历史。

    Args:
        ccy: 币种。
        period: 数据粒度。
        days: 天数。

    Returns:
        含 datetime, ts, longShortAccountRatio 的 DataFrame。
    """
    return _fetch_rubik_history(
        endpoint="/api/v5/rubik/stat/contracts/long-short-account-ratio",
        ccy=ccy,
        period=period,
        days=days,
        columns=["ts", "longShortAccountRatio"],
        cache_tag="ls_ratio",
        cache_mode=cache_mode,
        cache_ttl_hours=cache_ttl_hours,
    )


def fetch_taker_volume_history(
    inst_id: str,
    period: str = "5m",
    days: int = 60,
    cache_mode: str = "prefer_local",
    cache_ttl_hours: Optional[float] = None,
) -> pd.DataFrame:
    """获取主动买卖量历史。

    注意：此端点需要 instId（如 BTC-USDT-SWAP），非 ccy。

    Args:
        inst_id: 合约 ID，如 'BTC-USDT-SWAP'。
        period: 数据粒度。
        days: 天数。

    Returns:
        含 datetime, ts, buyVol, sellVol 的 DataFrame。
    """
    alias = inst_id.split("-")[0]
    cache_file = CACHE_DIR / f"{alias}_taker_vol_{period}_{days}d.parquet"

    def _fetch():
        url = f"{BASE_URL}/api/v5/rubik/stat/taker-volume-contract"
        target_ts = int((time.time() - days * 86400) * 1000)
        now_ts = int(time.time() * 1000)
        all_data = []
        window_ms = 24 * 3600 * 1000
        cursor_end = now_ts

        consecutive_errors = 0
        for _ in range(days + 5):
            cursor_begin = max(cursor_end - window_ms, target_ts)
            params = {
                "instId": inst_id,
                "period": period,
                "begin": str(cursor_begin),
                "end": str(cursor_end),
            }
            try:
                data = _okx_get(url, params, sleep_sec=0.5)
                batch = data.get("data", [])
                if batch:
                    all_data.extend(batch)
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
            except RuntimeError:
                consecutive_errors += 1

            if consecutive_errors >= 3:
                break
            if cursor_begin <= target_ts:
                break
            cursor_end = cursor_begin
            time.sleep(0.5)

        if not all_data:
            return pd.DataFrame()

        columns = ["ts", "buyVol", "sellVol"]
        df = pd.DataFrame(all_data, columns=columns)
        df["ts"] = df["ts"].astype(np.int64)
        df["buyVol"] = pd.to_numeric(df["buyVol"], errors="coerce").fillna(0.0)
        df["sellVol"] = pd.to_numeric(df["sellVol"], errors="coerce").fillna(0.0)
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
        df = df[df["ts"] >= target_ts]
        df = df.sort_values("ts").reset_index(drop=True)
        df = df.drop_duplicates(subset=["ts"]).reset_index(drop=True)
        return df

    return _cache_wrapper(cache_file, cache_mode, cache_ttl_hours, _fetch)


# ============================================================
# V10: 指数K线（用于基差计算）
# ============================================================

def fetch_index_candles_batch(
    inst_id: str,
    bar: str = "5m",
    days: int = 60,
    cache_mode: str = "prefer_local",
    cache_ttl_hours: Optional[float] = None,
) -> pd.DataFrame:
    """批量获取指数K线历史。

    Args:
        inst_id: 指数 ID，如 'BTC-USDT'（不带 SWAP 后缀）。
        bar: K线周期。
        days: 天数。
        cache_mode: 缓存模式。
        cache_ttl_hours: 缓存有效期。

    Returns:
        含 datetime, ts, open, high, low, close 的 DataFrame。

    Example:
        >>> df = fetch_index_candles_batch("BTC-USDT", bar="5m", days=60)
    """
    cache_file = CACHE_DIR / f"{inst_id}_index_{bar}_{days}d.parquet"

    def _fetch():
        url = f"{BASE_URL}/api/v5/market/history-index-candles"
        raw = _paginated_fetch(
            url=url,
            base_params={"instId": inst_id, "bar": bar, "limit": "100"},
            days=days,
            items_per_batch=100,
            sleep_sec=0.12,
            ts_index=0,
        )
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "confirm"])
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df["ts"] = df["ts"].astype(np.int64)
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
        target_ts = int((time.time() - days * 86400) * 1000)
        df = df[df["ts"] >= target_ts]
        df = df.sort_values("ts").reset_index(drop=True)
        df = df.drop_duplicates(subset=["ts"]).reset_index(drop=True)
        df = df[["datetime", "ts", "open", "high", "low", "close"]]
        return df

    return _cache_wrapper(cache_file, cache_mode, cache_ttl_hours, _fetch)


# ============================================================
# V10: 批量获取衍生品数据
# ============================================================

def fetch_all_derivatives_data(
    instruments: list[dict],
    days: int = 60,
    cache_mode: str = "prefer_local",
    cache_ttl_hours: Optional[float] = None,
    rubik_period: str = "5m",
) -> dict[str, dict[str, pd.DataFrame]]:
    """批量获取所有标的的衍生品微观结构数据。

    Args:
        instruments: 标的配置列表，每项含 instId 和 alias。
        days: 天数。
        cache_mode: 缓存模式。
        cache_ttl_hours: 缓存有效期。

    Returns:
        {alias: {data_type: DataFrame}} 的嵌套字典。

    Example:
        >>> data = fetch_all_derivatives_data(instruments, days=90)
        >>> data["BTC"]["funding_rate"]
    """
    result = {}
    for inst in instruments:
        inst_id = inst["instId"]
        alias = inst["alias"]
        # 指数 ID：去掉 -SWAP 后缀
        index_id = inst_id.replace("-SWAP", "")

        print(f"  Fetching derivatives data for {alias} ...")
        deriv = {}

        # 1. 资金费率
        try:
            print(f"    funding_rate ({inst_id}) ...")
            deriv["funding_rate"] = fetch_funding_rate_history(
                inst_id, days=days, cache_mode=cache_mode, cache_ttl_hours=cache_ttl_hours,
            )
            print(f"      → {len(deriv['funding_rate'])} records")
        except Exception as e:
            print(f"      WARN: funding_rate fetch failed: {e}")
            deriv["funding_rate"] = pd.DataFrame()

        # 2. 持仓量（需要 instId，非 ccy）
        try:
            print(f"    oi ({inst_id}, {rubik_period}) ...")
            deriv["oi"] = fetch_oi_history(
                inst_id, period=rubik_period, days=days, cache_mode=cache_mode, cache_ttl_hours=cache_ttl_hours,
            )
            print(f"      → {len(deriv['oi'])} records")
        except Exception as e:
            print(f"      WARN: oi fetch failed: {e}")
            deriv["oi"] = pd.DataFrame()

        # 3. 多空账户比
        try:
            print(f"    ls_ratio ({alias}, {rubik_period}) ...")
            deriv["ls_ratio"] = fetch_ls_ratio_history(
                alias, period=rubik_period, days=days, cache_mode=cache_mode, cache_ttl_hours=cache_ttl_hours,
            )
            print(f"      → {len(deriv['ls_ratio'])} records")
        except Exception as e:
            print(f"      WARN: ls_ratio fetch failed: {e}")
            deriv["ls_ratio"] = pd.DataFrame()

        # 4. 主动买卖量（需要 instId，非 ccy）
        try:
            print(f"    taker_vol ({inst_id}, {rubik_period}) ...")
            deriv["taker_vol"] = fetch_taker_volume_history(
                inst_id, period=rubik_period, days=days, cache_mode=cache_mode, cache_ttl_hours=cache_ttl_hours,
            )
            print(f"      → {len(deriv['taker_vol'])} records")
        except Exception as e:
            print(f"      WARN: taker_vol fetch failed: {e}")
            deriv["taker_vol"] = pd.DataFrame()

        # 5. 指数K线
        try:
            print(f"    index_candles ({index_id}) ...")
            deriv["index_candles"] = fetch_index_candles_batch(
                index_id, bar="5m", days=days, cache_mode=cache_mode, cache_ttl_hours=cache_ttl_hours,
            )
            print(f"      → {len(deriv['index_candles'])} records")
        except Exception as e:
            print(f"      WARN: index_candles fetch failed: {e}")
            deriv["index_candles"] = pd.DataFrame()

        result[alias] = deriv
    return result


# ============================================================
# V12: 新增 Rubik 系列 API（PCR / Margin Loan / Spot Taker /
#       Top Trader LS / Contract OI-Volume）
# ============================================================

def fetch_put_call_ratio(
    ccy: str,
    days: int = 72,
    cache_mode: str = "prefer_local",
    cache_ttl_hours: Optional[float] = None,
) -> pd.DataFrame:
    """获取期权 Put/Call Ratio（持仓量比与成交量比）。

    Args:
        ccy: 币种，如 'BTC'。
        days: 获取最近多少天（1D粒度最多72天）。
        cache_mode: 缓存模式。
        cache_ttl_hours: 缓存有效期。

    Returns:
        含 ts, datetime, pcr_oi, pcr_vol 的 DataFrame。

    Example:
        >>> df = fetch_put_call_ratio("BTC", days=60)
    """
    cache_file = CACHE_DIR / f"{ccy}_pcr_1D_{days}d.parquet"

    def _fetch():
        url = f"{BASE_URL}/api/v5/rubik/stat/option/open-interest-volume-ratio"
        data = _okx_get(url, {"ccy": ccy, "period": "1D"})
        raw = data.get("data", [])
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=["ts", "pcr_oi", "pcr_vol"])
        df["ts"] = df["ts"].astype(np.int64)
        df["pcr_oi"] = pd.to_numeric(df["pcr_oi"], errors="coerce").fillna(0.0)
        df["pcr_vol"] = pd.to_numeric(df["pcr_vol"], errors="coerce").fillna(0.0)
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
        target_ts = int((time.time() - days * 86400) * 1000)
        df = df[df["ts"] >= target_ts]
        df = df.sort_values("ts").reset_index(drop=True)
        df = df.drop_duplicates(subset=["ts"]).reset_index(drop=True)
        return df

    return _cache_wrapper(cache_file, cache_mode, cache_ttl_hours, _fetch)


def fetch_margin_loan_ratio(
    ccy: str,
    period: str = "1H",
    days: int = 30,
    cache_mode: str = "prefer_local",
    cache_ttl_hours: Optional[float] = None,
) -> pd.DataFrame:
    """获取杠杆借贷比历史。

    Args:
        ccy: 币种，如 'BTC'。
        period: 数据粒度，'1H' 最多30天，'1D' 最多180天。
        days: 获取最近多少天。
        cache_mode: 缓存模式。
        cache_ttl_hours: 缓存有效期。

    Returns:
        含 ts, datetime, margin_loan_ratio 的 DataFrame。

    Example:
        >>> df = fetch_margin_loan_ratio("BTC", period="1H", days=30)
    """
    cache_file = CACHE_DIR / f"{ccy}_margin_loan_{period}_{days}d.parquet"

    def _fetch():
        url = f"{BASE_URL}/api/v5/rubik/stat/margin/loan-ratio"
        data = _okx_get(url, {"ccy": ccy, "period": period})
        raw = data.get("data", [])
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=["ts", "margin_loan_ratio"])
        df["ts"] = df["ts"].astype(np.int64)
        df["margin_loan_ratio"] = pd.to_numeric(df["margin_loan_ratio"], errors="coerce").fillna(0.0)
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
        target_ts = int((time.time() - days * 86400) * 1000)
        df = df[df["ts"] >= target_ts]
        df = df.sort_values("ts").reset_index(drop=True)
        df = df.drop_duplicates(subset=["ts"]).reset_index(drop=True)
        return df

    return _cache_wrapper(cache_file, cache_mode, cache_ttl_hours, _fetch)


def fetch_spot_taker_volume(
    ccy: str,
    period: str = "1H",
    days: int = 30,
    cache_mode: str = "prefer_local",
    cache_ttl_hours: Optional[float] = None,
) -> pd.DataFrame:
    """获取现货主动买卖量历史。

    Args:
        ccy: 币种，如 'BTC'。
        period: 数据粒度，'1H' 最多30天。
        days: 获取最近多少天。
        cache_mode: 缓存模式。
        cache_ttl_hours: 缓存有效期。

    Returns:
        含 ts, datetime, spotBuyVol, spotSellVol 的 DataFrame。

    Example:
        >>> df = fetch_spot_taker_volume("BTC", period="1H", days=30)
    """
    cache_file = CACHE_DIR / f"{ccy}_spot_taker_{period}_{days}d.parquet"

    def _fetch():
        url = f"{BASE_URL}/api/v5/rubik/stat/taker-volume"
        data = _okx_get(url, {"ccy": ccy, "instType": "SPOT", "period": period})
        raw = data.get("data", [])
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=["ts", "spotBuyVol", "spotSellVol"])
        df["ts"] = df["ts"].astype(np.int64)
        df["spotBuyVol"] = pd.to_numeric(df["spotBuyVol"], errors="coerce").fillna(0.0)
        df["spotSellVol"] = pd.to_numeric(df["spotSellVol"], errors="coerce").fillna(0.0)
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
        target_ts = int((time.time() - days * 86400) * 1000)
        df = df[df["ts"] >= target_ts]
        df = df.sort_values("ts").reset_index(drop=True)
        df = df.drop_duplicates(subset=["ts"]).reset_index(drop=True)
        return df

    return _cache_wrapper(cache_file, cache_mode, cache_ttl_hours, _fetch)


def fetch_top_trader_ls_ratio(
    ccy: str,
    period: str = "1H",
    days: int = 30,
    cache_mode: str = "prefer_local",
    cache_ttl_hours: Optional[float] = None,
) -> pd.DataFrame:
    """获取顶级交易者多空比历史。

    Args:
        ccy: 币种，如 'BTC'。
        period: 数据粒度，'1H' 最多30天。
        days: 获取最近多少天。
        cache_mode: 缓存模式。
        cache_ttl_hours: 缓存有效期。

    Returns:
        含 ts, datetime, top_trader_ls_ratio 的 DataFrame。

    Example:
        >>> df = fetch_top_trader_ls_ratio("BTC", period="1H", days=30)
    """
    cache_file = CACHE_DIR / f"{ccy}_top_trader_ls_{period}_{days}d.parquet"

    def _fetch():
        url = f"{BASE_URL}/api/v5/rubik/stat/contracts/long-short-account-ratio"
        data = _okx_get(url, {"ccy": ccy, "period": period})
        raw = data.get("data", [])
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=["ts", "top_trader_ls_ratio"])
        df["ts"] = df["ts"].astype(np.int64)
        df["top_trader_ls_ratio"] = pd.to_numeric(df["top_trader_ls_ratio"], errors="coerce").fillna(0.0)
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
        target_ts = int((time.time() - days * 86400) * 1000)
        df = df[df["ts"] >= target_ts]
        df = df.sort_values("ts").reset_index(drop=True)
        df = df.drop_duplicates(subset=["ts"]).reset_index(drop=True)
        return df

    return _cache_wrapper(cache_file, cache_mode, cache_ttl_hours, _fetch)


def fetch_contract_oi_volume(
    ccy: str,
    period: str = "1H",
    days: int = 30,
    cache_mode: str = "prefer_local",
    cache_ttl_hours: Optional[float] = None,
) -> pd.DataFrame:
    """获取合约持仓量-成交量历史。

    Args:
        ccy: 币种，如 'BTC'。
        period: 数据粒度，'1H' 最多30天。
        days: 获取最近多少天。
        cache_mode: 缓存模式。
        cache_ttl_hours: 缓存有效期。

    Returns:
        含 ts, datetime, contract_oi, contract_vol 的 DataFrame。

    Example:
        >>> df = fetch_contract_oi_volume("BTC", period="1H", days=30)
    """
    cache_file = CACHE_DIR / f"{ccy}_contract_oi_vol_{period}_{days}d.parquet"

    def _fetch():
        url = f"{BASE_URL}/api/v5/rubik/stat/contracts/open-interest-volume"
        data = _okx_get(url, {"ccy": ccy, "period": period})
        raw = data.get("data", [])
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=["ts", "contract_oi", "contract_vol"])
        df["ts"] = df["ts"].astype(np.int64)
        df["contract_oi"] = pd.to_numeric(df["contract_oi"], errors="coerce").fillna(0.0)
        df["contract_vol"] = pd.to_numeric(df["contract_vol"], errors="coerce").fillna(0.0)
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
        target_ts = int((time.time() - days * 86400) * 1000)
        df = df[df["ts"] >= target_ts]
        df = df.sort_values("ts").reset_index(drop=True)
        df = df.drop_duplicates(subset=["ts"]).reset_index(drop=True)
        return df

    return _cache_wrapper(cache_file, cache_mode, cache_ttl_hours, _fetch)


# ============================================================
# V12: 批量获取全部 V12 数据
# ============================================================

def fetch_all_v12_data(
    instruments: list[dict],
    days: int = 30,
    cache_mode: str = "prefer_local",
    cache_ttl_hours: Optional[float] = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """批量获取所有标的的 V12 微观结构数据（PCR + Margin Loan + Spot Taker +
    Top Trader LS + Contract OI-Volume + Index Candles）。

    Args:
        instruments: 标的配置列表，每项含 instId 和 alias。
        days: 天数。
        cache_mode: 缓存模式。
        cache_ttl_hours: 缓存有效期。

    Returns:
        {alias: {data_type: DataFrame}} 的嵌套字典，keys 为
        pcr, margin_loan, spot_taker, top_trader_ls, contract_oi_vol, index_candles。

    Example:
        >>> data = fetch_all_v12_data(instruments, days=30)
        >>> data["BTC"]["pcr"]
    """
    result = {}
    for inst in instruments:
        inst_id = inst["instId"]
        alias = inst["alias"]
        ccy = alias  # e.g. "BTC"
        index_id = inst_id.replace("-SWAP", "")

        print(f"  Fetching V12 data for {alias} ...")
        v12 = {}

        # 1. Put/Call Ratio (1D, max 72d)
        pcr_days = min(days, 72)
        try:
            print(f"    pcr ({ccy}, 1D) ...")
            v12["pcr"] = fetch_put_call_ratio(
                ccy, days=pcr_days, cache_mode=cache_mode, cache_ttl_hours=cache_ttl_hours,
            )
            print(f"      -> {len(v12['pcr'])} records")
        except Exception as e:
            print(f"      WARN: pcr fetch failed: {e}")
            v12["pcr"] = pd.DataFrame()

        # 2. Margin Loan Ratio (1H, max 30d)
        ml_days = min(days, 30)
        try:
            print(f"    margin_loan ({ccy}, 1H) ...")
            v12["margin_loan"] = fetch_margin_loan_ratio(
                ccy, period="1H", days=ml_days, cache_mode=cache_mode,
                cache_ttl_hours=cache_ttl_hours,
            )
            print(f"      -> {len(v12['margin_loan'])} records")
        except Exception as e:
            print(f"      WARN: margin_loan fetch failed: {e}")
            v12["margin_loan"] = pd.DataFrame()

        # 3. Spot Taker Volume (1H, max 30d)
        st_days = min(days, 30)
        try:
            print(f"    spot_taker ({ccy}, 1H) ...")
            v12["spot_taker"] = fetch_spot_taker_volume(
                ccy, period="1H", days=st_days, cache_mode=cache_mode,
                cache_ttl_hours=cache_ttl_hours,
            )
            print(f"      -> {len(v12['spot_taker'])} records")
        except Exception as e:
            print(f"      WARN: spot_taker fetch failed: {e}")
            v12["spot_taker"] = pd.DataFrame()

        # 4. Top Trader Long/Short Ratio (1H, max 30d)
        tt_days = min(days, 30)
        try:
            print(f"    top_trader_ls ({ccy}, 1H) ...")
            v12["top_trader_ls"] = fetch_top_trader_ls_ratio(
                ccy, period="1H", days=tt_days, cache_mode=cache_mode,
                cache_ttl_hours=cache_ttl_hours,
            )
            print(f"      -> {len(v12['top_trader_ls'])} records")
        except Exception as e:
            print(f"      WARN: top_trader_ls fetch failed: {e}")
            v12["top_trader_ls"] = pd.DataFrame()

        # 5. Contract OI-Volume (1H, max 30d)
        oi_days = min(days, 30)
        try:
            print(f"    contract_oi_vol ({ccy}, 1H) ...")
            v12["contract_oi_vol"] = fetch_contract_oi_volume(
                ccy, period="1H", days=oi_days, cache_mode=cache_mode,
                cache_ttl_hours=cache_ttl_hours,
            )
            print(f"      -> {len(v12['contract_oi_vol'])} records")
        except Exception as e:
            print(f"      WARN: contract_oi_vol fetch failed: {e}")
            v12["contract_oi_vol"] = pd.DataFrame()

        # 6. Index Candles (reuse existing function)
        try:
            print(f"    index_candles ({index_id}) ...")
            v12["index_candles"] = fetch_index_candles_batch(
                index_id, bar="5m", days=days, cache_mode=cache_mode,
                cache_ttl_hours=cache_ttl_hours,
            )
            print(f"      -> {len(v12['index_candles'])} records")
        except Exception as e:
            print(f"      WARN: index_candles fetch failed: {e}")
            v12["index_candles"] = pd.DataFrame()

        result[alias] = v12
    return result


# ============================================================
# V13: Mark Price Candles
# ============================================================

def fetch_mark_price_candles_batch(
    inst_id: str,
    bar: str = "5m",
    days: int = 365,
    cache_mode: str = "prefer_local",
    cache_ttl_hours: Optional[float] = None,
) -> pd.DataFrame:
    """批量获取标记价格K线历史。

    标记价格用于计算与指数价格的溢价/折价，是永续合约基差分析的关键数据。

    Args:
        inst_id: 合约ID，如 'BTC-USDT-SWAP'。
        bar: K线周期，如 '5m', '1H', '4H', '1D'。
        days: 获取最近多少天的数据。
        cache_mode: 缓存模式。
        cache_ttl_hours: 缓存有效期。

    Returns:
        含 datetime, ts, open, high, low, close 的 DataFrame。

    Example:
        >>> df = fetch_mark_price_candles_batch("BTC-USDT-SWAP", bar="1H", days=365)
    """
    alias = inst_id.split("-")[0]
    cache_file = CACHE_DIR / f"{alias}_mark_{bar}_{days}d.parquet"

    def _fetch():
        url = f"{BASE_URL}/api/v5/market/history-mark-price-candles"
        raw = _paginated_fetch(
            url=url,
            base_params={"instId": inst_id, "bar": bar, "limit": "100"},
            days=days,
            items_per_batch=100,
            sleep_sec=0.12,
            ts_index=0,
        )
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "confirm"])
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df["ts"] = df["ts"].astype(np.int64)
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
        target_ts = int((time.time() - days * 86400) * 1000)
        df = df[df["ts"] >= target_ts]
        df = df.sort_values("ts").reset_index(drop=True)
        df = df.drop_duplicates(subset=["ts"]).reset_index(drop=True)
        df = df[["datetime", "ts", "open", "high", "low", "close"]]
        return df

    return _cache_wrapper(cache_file, cache_mode, cache_ttl_hours, _fetch)


# ============================================================
# V13: Position-based Long/Short Ratio (Rubik)
# ============================================================

def fetch_ls_position_ratio_history(
    ccy: str,
    period: str = "5m",
    days: int = 60,
    cache_mode: str = "prefer_local",
    cache_ttl_hours: Optional[float] = None,
) -> pd.DataFrame:
    """获取持仓多空比历史（按持仓量计算，区别于按账户数计算的 ls_ratio）。

    此指标反映大户持仓方向，与按账户数的多空比配合使用更有价值。

    Args:
        ccy: 币种，如 'BTC'。
        period: 数据粒度，如 '5m', '1H'。
        days: 天数。
        cache_mode: 缓存模式。
        cache_ttl_hours: 缓存有效期。

    Returns:
        含 datetime, ts, longShortPositionRatio 的 DataFrame。

    Example:
        >>> df = fetch_ls_position_ratio_history("BTC", period="1H", days=90)
    """
    return _fetch_rubik_history(
        endpoint="/api/v5/rubik/stat/contracts/long-short-account-ratio-contract-top-trader",
        ccy=ccy,
        period=period,
        days=days,
        columns=["ts", "longShortPositionRatio"],
        cache_tag="ls_position_ratio",
        cache_mode=cache_mode,
        cache_ttl_hours=cache_ttl_hours,
    )


# ============================================================
# V13: Liquidation Orders
# ============================================================

def fetch_liquidation_orders(
    inst_type: str = "SWAP",
    uly: str = "BTC-USDT",
    days: int = 7,
    cache_mode: str = "prefer_local",
    cache_ttl_hours: Optional[float] = None,
) -> pd.DataFrame:
    """获取公开爆仓单数据。

    OKX 爆仓数据 API 通常返回最近的爆仓事件，时间跨度有限。
    用于构建爆仓强度因子。

    Args:
        inst_type: 产品类型，如 'SWAP', 'FUTURES'。
        uly: 标的，如 'BTC-USDT'。
        days: 获取最近多少天（API 可能限制在 7 天内）。
        cache_mode: 缓存模式。
        cache_ttl_hours: 缓存有效期。

    Returns:
        含 instId, side, sz, price, ts, datetime 的 DataFrame。

    Example:
        >>> df = fetch_liquidation_orders(uly="BTC-USDT", days=7)
    """
    ccy = uly.split("-")[0]
    cache_file = CACHE_DIR / f"{ccy}_liquidations_{days}d.parquet"

    def _fetch():
        url = f"{BASE_URL}/api/v5/public/liquidation-orders"
        target_ts = int((time.time() - days * 86400) * 1000)
        all_data = []
        after_ts = None

        # OKX liquidation API: state=filled, paginate by after
        for _ in range(100):
            params = {
                "instType": inst_type,
                "uly": uly,
                "state": "filled",
                "limit": "100",
            }
            if after_ts:
                params["after"] = str(after_ts)

            try:
                data = _okx_get(url, params, sleep_sec=0.2)
                batch = data.get("data", [])
                if not batch:
                    break

                # liquidation-orders returns [{details: [{...}], ...}]
                for item in batch:
                    details = item.get("details", [])
                    for d in details:
                        all_data.append({
                            "instId": d.get("instId", ""),
                            "side": d.get("side", ""),
                            "sz": float(d.get("sz", 0)),
                            "price": float(d.get("bkPx", 0)),
                            "ts": int(d.get("ts", 0)),
                        })
                    # Use the item-level ts for pagination
                    item_ts = int(item.get("ts", 0))
                    if item_ts > 0:
                        after_ts = item_ts

                if after_ts and after_ts <= target_ts:
                    break
            except RuntimeError:
                break

            time.sleep(0.2)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
        df = df[df["ts"] >= target_ts]
        df = df.sort_values("ts").reset_index(drop=True)
        df = df.drop_duplicates().reset_index(drop=True)
        return df

    return _cache_wrapper(cache_file, cache_mode, cache_ttl_hours, _fetch)


# ============================================================
# V13: Margin Loan Ratio (1D, max 180d — extends the 1H/30d version)
# ============================================================

def fetch_margin_loan_ratio_daily(
    ccy: str,
    days: int = 180,
    cache_mode: str = "prefer_local",
    cache_ttl_hours: Optional[float] = None,
) -> pd.DataFrame:
    """获取杠杆借贷比历史（1D 粒度，最多 180 天）。

    与 1H 版本互补，覆盖更长时间跨度。

    Args:
        ccy: 币种，如 'BTC'。
        days: 获取最近多少天（最多180天）。
        cache_mode: 缓存模式。
        cache_ttl_hours: 缓存有效期。

    Returns:
        含 ts, datetime, margin_loan_ratio 的 DataFrame。

    Example:
        >>> df = fetch_margin_loan_ratio_daily("BTC", days=180)
    """
    days = min(days, 180)
    cache_file = CACHE_DIR / f"{ccy}_margin_loan_1D_{days}d.parquet"

    def _fetch():
        url = f"{BASE_URL}/api/v5/rubik/stat/margin/loan-ratio"
        data = _okx_get(url, {"ccy": ccy, "period": "1D"})
        raw = data.get("data", [])
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=["ts", "margin_loan_ratio"])
        df["ts"] = df["ts"].astype(np.int64)
        df["margin_loan_ratio"] = pd.to_numeric(df["margin_loan_ratio"], errors="coerce").fillna(0.0)
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
        target_ts = int((time.time() - days * 86400) * 1000)
        df = df[df["ts"] >= target_ts]
        df = df.sort_values("ts").reset_index(drop=True)
        df = df.drop_duplicates(subset=["ts"]).reset_index(drop=True)
        return df

    return _cache_wrapper(cache_file, cache_mode, cache_ttl_hours, _fetch)


# ============================================================
# Cross-Asset Data (yfinance, 零注册)
# ============================================================

def fetch_cross_asset_daily(days: int = 730) -> pd.DataFrame:
    """通过 yfinance 拉取跨资产日频数据 (QQQ/GLD/TLT/UUP/IBIT/HYG)。

    Args:
        days: 回溯天数，默认 2 年。

    Returns:
        DataFrame，含 ts (ms) 和各 ticker 的 close/volume 列。
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed, skip cross-asset fetch")
        return pd.DataFrame()

    tickers = ["QQQ", "GLD", "TLT", "UUP", "IBIT", "HYG"]

    try:
        raw = yf.download(tickers, period=f"{days}d", progress=False, auto_adjust=True)
    except Exception as e:
        logger.warning(f"yfinance download failed: {e}")
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    rows = []
    for date_idx in raw.index:
        row = {"date": date_idx}
        for ticker in tickers:
            try:
                row[f"{ticker}_close"] = float(raw.loc[date_idx, ("Close", ticker)])
                row[f"{ticker}_volume"] = float(raw.loc[date_idx, ("Volume", ticker)])
            except (KeyError, TypeError):
                pass
        rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["ts"] = (df["date"].astype(np.int64) // 10**6) + 21 * 3600 * 1000
    df = df.sort_values("ts").reset_index(drop=True)
    logger.info(f"Cross-asset: {len(df)} daily bars, {len(tickers)} tickers")
    return df


# ============================================================
# On-Chain Data (DeFiLlama + blockchain.com, 零注册)
# ============================================================

def fetch_stablecoin_supply() -> pd.DataFrame:
    """从 DeFiLlama 拉取 stablecoin 供应量历史 (USDT + 全部)。

    Returns:
        DataFrame，含 ts (ms), usdt_supply, total_sc_supply。
    """
    try:
        resp = requests.get(
            "https://stablecoins.llama.fi/stablecoincharts/all?stablecoin=1",
            timeout=15,
        )
        usdt_data = resp.json()
        resp2 = requests.get(
            "https://stablecoins.llama.fi/stablecoincharts/all",
            timeout=15,
        )
        total_data = resp2.json()
    except Exception as e:
        logger.warning(f"DeFiLlama stablecoin fetch failed: {e}")
        return pd.DataFrame()

    usdt_rows = [
        {"ts": int(e.get("date", 0)) * 1000,
         "usdt_supply": float(e.get("totalCirculating", {}).get("peggedUSD", 0))}
        for e in usdt_data
    ]
    total_rows = [
        {"ts": int(e.get("date", 0)) * 1000,
         "total_sc_supply": float(e.get("totalCirculating", {}).get("peggedUSD", 0))}
        for e in total_data
    ]

    df_usdt = pd.DataFrame(usdt_rows)
    df_total = pd.DataFrame(total_rows)
    if df_usdt.empty:
        return pd.DataFrame()

    df = pd.merge(df_usdt, df_total, on="ts", how="outer").sort_values("ts")
    df = df.drop_duplicates(subset=["ts"]).reset_index(drop=True)
    logger.info(f"Stablecoin supply: {len(df)} daily entries")
    return df


def fetch_blockchain_stats(days: int = 730) -> pd.DataFrame:
    """从 blockchain.com 拉取 BTC 链上统计 (hashrate, tx_count, mempool)。

    Args:
        days: 回溯天数。

    Returns:
        DataFrame，含 ts (ms), hashrate, tx_count, mempool_size。
    """
    bc_base = "https://api.blockchain.info/charts"
    timespan = f"{days}days"
    metrics = {
        "hash-rate": "hashrate",
        "n-transactions": "tx_count",
        "mempool-size": "mempool_size",
    }

    all_dfs = []
    for chart, col_name in metrics.items():
        try:
            resp = requests.get(
                f"{bc_base}/{chart}",
                params={"timespan": timespan, "format": "json", "sampled": "true"},
                timeout=15,
            )
            chart_data = resp.json()
            rows = [{"ts": int(v["x"]) * 1000, col_name: float(v["y"])}
                    for v in chart_data.get("values", [])]
            all_dfs.append(pd.DataFrame(rows))
        except Exception as e:
            logger.warning(f"blockchain.com {chart} fetch failed: {e}")

    if not all_dfs:
        return pd.DataFrame()

    result_df = all_dfs[0]
    for other in all_dfs[1:]:
        result_df = pd.merge(result_df, other, on="ts", how="outer")
    result_df = result_df.sort_values("ts").drop_duplicates(subset=["ts"]).reset_index(drop=True)
    logger.info(f"Blockchain stats: {len(result_df)} entries")
    return result_df
