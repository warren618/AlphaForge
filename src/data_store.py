"""统一数据仓库。

每种数据一个持续增长的 parquet 文件，所有消费者通过同一个读取接口访问。
更新时增量追加新数据，按 ts 去重。迁移 = scp data/store/。

目录结构:
    data/store/
      candles/          BTC_5m.parquet, ETH_15m.parquet, ...
      market/           BTC_funding_rate.parquet, BTC_oi_1H.parquet, ...
      macro/            fear_greed.parquet, fred_macro.parquet
      binance/          BTC_5m.parquet, ETH_5m.parquet

Usage:
    from src.data_store import DataStore
    store = DataStore()
    df = store.read_candles("BTC", bar="5m", days=90)
    store.update_candles("BTC", bar="5m")

CLI:
    python -m src.data_store info
    python -m src.data_store update
    python -m src.data_store update --candles-only
    python -m src.data_store migrate
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.logging_config import get_logger

IS_LINUX = platform.system() == "Linux"

logger = get_logger("data_store")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OLD_CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# inst_id -> (alias, index_id)
INSTRUMENT_MAP = {
    "BTC-USDT-SWAP": {"alias": "BTC", "index_id": "BTC-USDT"},
    "ETH-USDT-SWAP": {"alias": "ETH", "index_id": "ETH-USDT"},
    "SOL-USDT-SWAP": {"alias": "SOL", "index_id": "SOL-USDT"},
}

# market data types and their fetch configs
MARKET_SOURCES = {
    "funding_rate": {
        "fetch_fn": "fetch_funding_rate_history",
        "needs_inst_id": True,
        "default_days": 90,
    },
    "oi_1H": {
        "fetch_fn": "fetch_oi_history",
        "needs_inst_id": True,
        "kwargs": {"period": "1H"},
        "default_days": 90,
    },
    "ls_ratio_1H": {
        "fetch_fn": "fetch_ls_ratio_history",
        "needs_inst_id": False,
        "kwargs": {"period": "1H"},
        "default_days": 90,
    },
    "taker_vol_1H": {
        "fetch_fn": "fetch_taker_volume_history",
        "needs_inst_id": True,
        "kwargs": {"period": "1H"},
        "default_days": 90,
    },
    "index_5m": {
        "fetch_fn": "fetch_index_candles_batch",
        "needs_index_id": True,
        "kwargs": {"bar": "5m"},
        "default_days": 90,
    },
    "pcr_1D": {
        "fetch_fn": "fetch_put_call_ratio",
        "needs_inst_id": False,
        "default_days": 72,
        "symbols": ["BTC", "ETH"],
    },
    "margin_loan_1H": {
        "fetch_fn": "fetch_margin_loan_ratio",
        "needs_inst_id": False,
        "kwargs": {"period": "1H"},
        "default_days": 30,
    },
    "spot_taker_1H": {
        "fetch_fn": "fetch_spot_taker_volume",
        "needs_inst_id": False,
        "kwargs": {"period": "1H"},
        "default_days": 30,
    },
    "top_trader_ls_1H": {
        "fetch_fn": "fetch_top_trader_ls_ratio",
        "needs_inst_id": False,
        "kwargs": {"period": "1H"},
        "default_days": 30,
    },
    "contract_oi_vol_1H": {
        "fetch_fn": "fetch_contract_oi_volume",
        "needs_inst_id": False,
        "kwargs": {"period": "1H"},
        "default_days": 30,
    },
}

# Old cache filename -> new store path mapping for migration
# Format: (regex_pattern, store_subdir, new_filename_template)
MIGRATION_RULES = [
    # OHLCV candles: BTC-USDT-SWAP_5m_90d.parquet -> candles/BTC_5m.parquet
    (r"^(BTC|ETH|SOL)-USDT-SWAP_(5m|15m)_\d+d\.parquet$",
     "candles", lambda m: f"{m.group(1)}_{m.group(2)}.parquet"),
    # Binance: BINANCE_BTC_5m.parquet -> binance/BTC_5m.parquet
    (r"^BINANCE_(BTC|ETH)_5m\.parquet$",
     "binance", lambda m: f"{m.group(1)}_5m.parquet"),
    # Funding rate: BTC-USDT-SWAP_funding_rate_90d.parquet -> market/BTC_funding_rate.parquet
    (r"^(BTC|ETH|SOL)-USDT-SWAP_funding_rate_\d+d\.parquet$",
     "market", lambda m: f"{m.group(1)}_funding_rate.parquet"),
    # OI: BTC_oi_1H_90d.parquet -> market/BTC_oi_1H.parquet
    (r"^(BTC|ETH|SOL)_oi_(1H|5m)_\d+d\.parquet$",
     "market", lambda m: f"{m.group(1)}_oi_{m.group(2)}.parquet"),
    # LS ratio: BTC_ls_ratio_1H_90d.parquet -> market/BTC_ls_ratio_1H.parquet
    (r"^(BTC|ETH|SOL)_ls_ratio_(1H|5m)_\d+d\.parquet$",
     "market", lambda m: f"{m.group(1)}_ls_ratio_{m.group(2)}.parquet"),
    # Taker vol: BTC_taker_vol_1H_90d.parquet -> market/BTC_taker_vol_1H.parquet
    (r"^(BTC|ETH|SOL)_taker_vol_(1H|5m)_\d+d\.parquet$",
     "market", lambda m: f"{m.group(1)}_taker_vol_{m.group(2)}.parquet"),
    # Index candles: BTC-USDT_index_5m_90d.parquet -> market/BTC_index_5m.parquet
    (r"^(BTC|ETH|SOL)-USDT_index_5m_\d+d\.parquet$",
     "market", lambda m: f"{m.group(1)}_index_5m.parquet"),
    # PCR: BTC_pcr_1D_90d.parquet -> market/BTC_pcr_1D.parquet
    (r"^(BTC|ETH)_pcr_1D_\d+d\.parquet$",
     "market", lambda m: f"{m.group(1)}_pcr_1D.parquet"),
    # Margin loan: BTC_margin_loan_1H_90d.parquet -> market/BTC_margin_loan_1H.parquet
    (r"^(BTC|ETH|SOL)_margin_loan_1H_\d+d\.parquet$",
     "market", lambda m: f"{m.group(1)}_margin_loan_1H.parquet"),
    # Spot taker: BTC_spot_taker_1H_90d.parquet -> market/BTC_spot_taker_1H.parquet
    (r"^(BTC|ETH|SOL)_spot_taker_1H_\d+d\.parquet$",
     "market", lambda m: f"{m.group(1)}_spot_taker_1H.parquet"),
    # Top trader LS: BTC_top_trader_ls_1H_90d.parquet -> market/BTC_top_trader_ls_1H.parquet
    (r"^(BTC|ETH|SOL)_top_trader_ls_1H_\d+d\.parquet$",
     "market", lambda m: f"{m.group(1)}_top_trader_ls_1H.parquet"),
    # Contract OI vol: BTC_contract_oi_vol_1H_90d.parquet -> market/BTC_contract_oi_vol_1H.parquet
    (r"^(BTC|ETH|SOL)_contract_oi_vol_1H_\d+d\.parquet$",
     "market", lambda m: f"{m.group(1)}_contract_oi_vol_1H.parquet"),
]


def _ensure_numpy_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """将 pyarrow/nullable 列转为原生 numpy 类型。

    这样写入的 parquet 文件读回来不会有只读数组问题。

    Args:
        df: 输入 DataFrame。

    Returns:
        所有列都是原生 numpy dtype 的 DataFrame。
    """
    for col in df.columns:
        dtype = df[col].dtype
        if hasattr(dtype, "pyarrow_dtype") or dtype.name.startswith(("Int", "Float", "boolean")):
            df[col] = df[col].to_numpy(copy=True, na_value=np.nan)
    return df


def _atomic_write_parquet(df: pd.DataFrame, path: Path):
    """原子写 parquet: 确保 numpy 类型 → 写 tmp → rename。

    Args:
        df: 要写入的 DataFrame。
        path: 目标文件路径。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _ensure_numpy_dtypes(df)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _dedup_sort(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    """按 ts 去重并排序。

    Args:
        df: 输入 DataFrame。
        ts_col: 时间戳列名。

    Returns:
        去重排序后的 DataFrame。
    """
    if df.empty or ts_col not in df.columns:
        return df
    return df.drop_duplicates(subset=[ts_col], keep="last").sort_values(ts_col).reset_index(drop=True)


def _cutoff_ts(days: int) -> int:
    """计算 N 天前的毫秒时间戳。

    Args:
        days: 天数。

    Returns:
        毫秒时间戳。
    """
    return int((time.time() - days * 86400) * 1000)


class DataStore:
    """统一数据仓库。

    每种数据一个持续增长的 parquet 文件。
    读取时按 days 截取最近 N 天。
    更新时增量追加新数据到文件末尾。

    Attributes:
        store_dir: 数据仓库根目录。
    """

    def __init__(self, store_dir: Path | None = None):
        """初始化数据仓库。

        Args:
            store_dir: 数据仓库根目录，默认 PROJECT_ROOT/data/store。
        """
        self.store_dir = store_dir or PROJECT_ROOT / "data" / "store"

    def _path(self, category: str, filename: str) -> Path:
        """构建存储路径。

        Args:
            category: 子目录名 (candles/market/macro/binance)。
            filename: 文件名。

        Returns:
            完整文件路径。
        """
        return self.store_dir / category / filename

    def _read_parquet(self, path: Path, days: int | None = None) -> pd.DataFrame:
        """读 parquet，可选截取最近 N 天。

        崩溃恢复: 如果 .parquet.tmp 存在但 .parquet 不存在，自动恢复。

        Args:
            path: 文件路径。
            days: 截取最近 N 天，None 表示全部。

        Returns:
            DataFrame，文件不存在返回空 DataFrame。
        """
        tmp = path.with_suffix(".parquet.tmp")
        if tmp.exists() and not path.exists():
            logger.warning(f"Recovering orphaned tmp: {tmp.name}")
            tmp.replace(path)
        elif tmp.exists() and path.exists():
            tmp.unlink(missing_ok=True)

        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        if days and "ts" in df.columns and not df.empty:
            cutoff = _cutoff_ts(days)
            df = df[df["ts"] >= cutoff].reset_index(drop=True)
        return df

    def _append(self, path: Path, new_df: pd.DataFrame, ts_col: str = "ts"):
        """增量追加: 加锁 → 读已有 → 合并 → 去重排序 → 原子写回 → 释放锁。

        使用文件锁防止 paper_trade 和 cron 同时写导致数据丢失。

        Args:
            path: 文件路径。
            new_df: 新数据。
            ts_col: 时间戳列名。
        """
        if new_df.empty:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(".lock")

        lock_fd = open(lock_path, "w")
        try:
            if IS_LINUX:
                import fcntl
                fcntl.flock(lock_fd, fcntl.LOCK_EX)

            existing = self._read_parquet(path)
            if existing.empty:
                combined = new_df
            else:
                combined = pd.concat([existing, new_df], ignore_index=True)
            combined = _dedup_sort(combined, ts_col)
            _atomic_write_parquet(combined, path)
            logger.info(f"  {path.name}: {len(combined)} rows")
        finally:
            if IS_LINUX:
                import fcntl
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    # ============================================================
    # 读取接口
    # ============================================================

    def read_candles(self, symbol: str, bar: str = "5m", days: int | None = None,
                     source: str = "okx") -> pd.DataFrame:
        """读 K 线数据，可选截取最近 N 天。

        Args:
            symbol: 币种别名，如 "BTC"。
            bar: K 线周期，如 "5m", "15m"。
            days: 截取最近 N 天，None 表示全部。
            source: 数据来源，"okx" 或 "binance"。

        Returns:
            OHLCV DataFrame。

        Example:
            >>> store = DataStore()
            >>> df = store.read_candles("BTC", bar="5m", days=90)
        """
        if source == "binance":
            path = self._path("binance", f"{symbol}_{bar}.parquet")
        else:
            path = self._path("candles", f"{symbol}_{bar}.parquet")
        df = self._read_parquet(path, days=days)
        if df.empty:
            logger.warning(f"No data: {path}")
        return df

    def read_market(self, symbol: str, days: int | None = None) -> dict[str, pd.DataFrame]:
        """读市场微观结构全家桶。

        返回所有存在的市场数据文件，key 为数据类型名。

        Args:
            symbol: 币种别名，如 "BTC"。
            days: 截取最近 N 天。

        Returns:
            {data_type: DataFrame} 字典。
            Keys 映射到因子系统的旧 key:
              funding_rate, oi, ls_ratio, taker_vol, index_candles,
              pcr, margin_loan, spot_taker, top_trader_ls, contract_oi_vol

        Example:
            >>> market = store.read_market("BTC", days=90)
            >>> market["funding_rate"]
        """
        market_dir = self.store_dir / "market"
        if not market_dir.exists():
            return {}

        # 文件名 -> 返回 key 的映射
        file_key_map = {
            f"{symbol}_funding_rate.parquet": "funding_rate",
            f"{symbol}_oi_1H.parquet": "oi",
            f"{symbol}_ls_ratio_1H.parquet": "ls_ratio",
            f"{symbol}_taker_vol_1H.parquet": "taker_vol",
            f"{symbol}_index_5m.parquet": "index_candles",
            f"{symbol}_pcr_1D.parquet": "pcr",
            f"{symbol}_margin_loan_1H.parquet": "margin_loan",
            f"{symbol}_spot_taker_1H.parquet": "spot_taker",
            f"{symbol}_top_trader_ls_1H.parquet": "top_trader_ls",
            f"{symbol}_contract_oi_vol_1H.parquet": "contract_oi_vol",
        }

        result = {}
        for filename, key in file_key_map.items():
            path = market_dir / filename
            if path.exists():
                df = self._read_parquet(path, days=days)
                if not df.empty:
                    result[key] = df
        return result

    def read_macro(self, days: int | None = None) -> dict[str, pd.DataFrame]:
        """读宏观/情绪数据。

        Args:
            days: 截取最近 N 天。

        Returns:
            {name: DataFrame} 字典 (fear_greed, fred_macro)。
        """
        result = {}
        for name in ["fear_greed", "fred_macro"]:
            path = self._path("macro", f"{name}.parquet")
            df = self._read_parquet(path, days=days)
            if not df.empty:
                result[name] = df
        return result

    def read_cross_asset(self, days: int | None = None) -> pd.DataFrame:
        """读跨资产日频数据 (QQQ/GLD/TLT/UUP/IBIT/HYG)。"""
        path = self._path("cross_asset", "daily.parquet")
        return self._read_parquet(path, days=days)

    def read_onchain(self, days: int | None = None) -> dict[str, pd.DataFrame]:
        """读链上数据 (stablecoin + blockchain stats)。"""
        result = {}
        for name in ["stablecoin_supply", "blockchain_stats"]:
            path = self._path("onchain", f"{name}.parquet")
            df = self._read_parquet(path, days=days)
            if not df.empty:
                result[name] = df
        return result

    def update_cross_asset(self):
        """拉取并存储跨资产日频数据。"""
        from src.data_fetcher import fetch_cross_asset_daily
        df = fetch_cross_asset_daily(days=730)
        if not df.empty:
            self._append(self._path("cross_asset", "daily.parquet"), df)

    def update_onchain(self):
        """拉取并存储链上数据。"""
        from src.data_fetcher import fetch_stablecoin_supply, fetch_blockchain_stats
        df_sc = fetch_stablecoin_supply()
        if not df_sc.empty:
            self._append(self._path("onchain", "stablecoin_supply.parquet"), df_sc)
        df_bc = fetch_blockchain_stats(days=730)
        if not df_bc.empty:
            self._append(self._path("onchain", "blockchain_stats.parquet"), df_bc)

    # ============================================================
    # 更新接口 (增量追加)
    # ============================================================

    def update_candles(self, symbol: str, bar: str = "5m"):
        """从 OKX API 增量追加最新 K 线。

        读已有文件 → 取 last ts → 从 last_ts 分页拉取新 bar → 去重 → 存回。

        Args:
            symbol: 币种别名，如 "BTC"。
            bar: K 线周期。
        """
        from src.data_fetcher import fetch_candles

        inst_id = f"{symbol}-USDT-SWAP"
        path = self._path("candles", f"{symbol}_{bar}.parquet")

        existing = self._read_parquet(path)
        last_ts = int(existing["ts"].max()) if not existing.empty else 0

        new_bars = self._fetch_candles_since(inst_id, bar, last_ts)
        if not new_bars.empty:
            self._append(path, new_bars)
            logger.info(f"  {symbol} {bar}: +{len(new_bars)} bars")
        else:
            logger.info(f"  {symbol} {bar}: up to date")

    def _fetch_candles_since(self, inst_id: str, bar: str, since_ts: int) -> pd.DataFrame:
        """从 since_ts 之后拉取所有新 K 线。

        OKX API 返回数据从新到旧排列。用 after 参数从当前时间向后翻页，
        直到翻到 since_ts 为止。

        Args:
            inst_id: 合约 ID。
            bar: K 线周期。
            since_ts: 起始毫秒时间戳 (不含此 ts)。

        Returns:
            新 K 线 DataFrame。
        """
        from src.data_fetcher import fetch_candles

        all_data = []
        after_ts = None  # 从最新开始向后翻页

        for _ in range(500):
            batch = fetch_candles(inst_id, bar=bar, limit=300, after=after_ts)
            if not batch:
                break
            all_data.extend(batch)
            # batch[-1] 是本批最旧的一条
            oldest_ts = int(batch[-1][0])
            # 如果最旧的一条已经 <= since_ts，说明已覆盖到了
            if oldest_ts <= since_ts:
                break
            if len(batch) < 300:
                break
            after_ts = str(oldest_ts)
            time.sleep(0.12)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data, columns=[
            "ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"
        ])
        for col in ["open", "high", "low", "close", "vol", "volCcy", "volCcyQuote"]:
            df[col] = df[col].astype(float)
        df["ts"] = df["ts"].astype(np.int64)
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms")

        # 只保留比 since_ts 更新的数据
        if since_ts > 0:
            df = df[df["ts"] > since_ts]

        return _dedup_sort(df)

    def update_market(self, symbol: str):
        """增量更新所有市场微观结构数据。

        Args:
            symbol: 币种别名，如 "BTC"。
        """
        from src.data_fetcher import (
            fetch_funding_rate_history,
            fetch_oi_history,
            fetch_ls_ratio_history,
            fetch_taker_volume_history,
            fetch_index_candles_batch,
            fetch_put_call_ratio,
            fetch_margin_loan_ratio,
            fetch_spot_taker_volume,
            fetch_top_trader_ls_ratio,
            fetch_contract_oi_volume,
        )

        inst_id = f"{symbol}-USDT-SWAP"
        index_id = f"{symbol}-USDT"

        # (store_filename, fetch_call)
        tasks = [
            (f"{symbol}_funding_rate.parquet",
             lambda: fetch_funding_rate_history(inst_id, days=90, cache_mode="refresh")),
            (f"{symbol}_oi_1H.parquet",
             lambda: fetch_oi_history(inst_id, period="1H", days=90, cache_mode="refresh")),
            (f"{symbol}_ls_ratio_1H.parquet",
             lambda: fetch_ls_ratio_history(symbol, period="1H", days=90, cache_mode="refresh")),
            (f"{symbol}_taker_vol_1H.parquet",
             lambda: fetch_taker_volume_history(inst_id, period="1H", days=90, cache_mode="refresh")),
            (f"{symbol}_index_5m.parquet",
             lambda: fetch_index_candles_batch(index_id, bar="5m", days=90, cache_mode="refresh")),
            (f"{symbol}_margin_loan_1H.parquet",
             lambda: fetch_margin_loan_ratio(symbol, period="1H", days=30, cache_mode="refresh")),
            (f"{symbol}_spot_taker_1H.parquet",
             lambda: fetch_spot_taker_volume(symbol, period="1H", days=30, cache_mode="refresh")),
            (f"{symbol}_top_trader_ls_1H.parquet",
             lambda: fetch_top_trader_ls_ratio(symbol, period="1H", days=30, cache_mode="refresh")),
            (f"{symbol}_contract_oi_vol_1H.parquet",
             lambda: fetch_contract_oi_volume(symbol, period="1H", days=30, cache_mode="refresh")),
        ]

        # PCR only for BTC/ETH
        if symbol in ("BTC", "ETH"):
            tasks.append(
                (f"{symbol}_pcr_1D.parquet",
                 lambda: fetch_put_call_ratio(symbol, days=72, cache_mode="refresh"))
            )

        logger.info(f"Updating market data for {symbol}...")
        for filename, fetch_fn in tasks:
            path = self._path("market", filename)
            try:
                fresh = fetch_fn()
                if not fresh.empty:
                    self._append(path, fresh)
            except Exception as e:
                logger.warning(f"  {filename}: FAILED ({e})")

    def update_all(self, symbols: list[str] | None = None):
        """更新所有标的的所有数据。

        Args:
            symbols: 币种列表，默认 ["BTC", "ETH", "SOL"]。
        """
        symbols = symbols or ["BTC", "ETH", "SOL"]

        logger.info("=== DataStore Update ===")
        for symbol in symbols:
            logger.info(f"\n[{symbol}] Candles...")
            self.update_candles(symbol, bar="5m")
            self.update_candles(symbol, bar="15m")

            logger.info(f"[{symbol}] Market data...")
            self.update_market(symbol)

        logger.info("\nUpdate complete.")
        self.info()

    # ============================================================
    # 工具
    # ============================================================

    def info(self) -> dict:
        """打印所有表的行数、时间范围、文件大小。

        Returns:
            {category: {filename: stats}} 字典。
        """
        stats = {}
        total_size = 0
        total_rows = 0

        for category in ["candles", "market", "macro", "binance"]:
            cat_dir = self.store_dir / category
            if not cat_dir.exists():
                continue

            cat_stats = {}
            for f in sorted(cat_dir.glob("*.parquet")):
                size = f.stat().st_size
                try:
                    df = pd.read_parquet(f)
                    rows = len(df)
                    if "ts" in df.columns and rows > 0:
                        ts_min = pd.to_datetime(df["ts"].min(), unit="ms")
                        ts_max = pd.to_datetime(df["ts"].max(), unit="ms")
                        time_range = f"{ts_min} ~ {ts_max}"
                    elif "datetime" in df.columns and rows > 0:
                        time_range = f"{df['datetime'].min()} ~ {df['datetime'].max()}"
                    else:
                        time_range = "N/A"
                except Exception:
                    rows = -1
                    time_range = "ERROR"

                cat_stats[f.name] = {
                    "rows": rows,
                    "size_mb": round(size / 1024 / 1024, 2),
                    "range": time_range,
                }
                total_size += size
                if rows > 0:
                    total_rows += rows

            if cat_stats:
                stats[category] = cat_stats

        # Print
        print(f"\n{'=' * 80}")
        print(f"DataStore: {self.store_dir}")
        print(f"{'=' * 80}")

        for category, files in stats.items():
            print(f"\n  [{category}]")
            for name, s in files.items():
                print(f"    {name:<45} {s['rows']:>8} rows  {s['size_mb']:>6.1f} MB  {s['range']}")

        print(f"\n  Total: {total_rows:,} rows, {total_size / 1024 / 1024:.1f} MB")
        return stats

    def migrate_from_cache(self):
        """一次性迁移: data/cache/*.parquet -> data/store/。

        对于同一个目标文件有多个源文件时（如 _90d 和 _30d），
        合并所有源文件取最多数据。
        """
        if not OLD_CACHE_DIR.exists():
            logger.warning("No cache directory found")
            return

        files = list(OLD_CACHE_DIR.glob("*.parquet"))
        migrated = 0
        skipped = 0

        # Group by target to merge multiple day variants
        target_sources: dict[str, list[Path]] = {}

        for f in files:
            matched = False
            for pattern, subdir, name_fn in MIGRATION_RULES:
                m = re.match(pattern, f.name)
                if m:
                    new_name = name_fn(m)
                    target_key = f"{subdir}/{new_name}"
                    target_sources.setdefault(target_key, []).append(f)
                    matched = True
                    break
            if not matched:
                logger.debug(f"  Skip (no rule): {f.name}")
                skipped += 1

        for target_key, sources in target_sources.items():
            subdir, new_name = target_key.split("/", 1)
            target_path = self.store_dir / subdir / new_name

            # Read all source files and merge (pick the one with most rows)
            dfs = []
            for src in sources:
                try:
                    df = pd.read_parquet(src)
                    if not df.empty:
                        dfs.append(df)
                except Exception as e:
                    logger.warning(f"  Read failed: {src.name} ({e})")

            if not dfs:
                continue

            # Merge all variants
            combined = pd.concat(dfs, ignore_index=True)
            if "ts" in combined.columns:
                combined["ts"] = combined["ts"].astype(np.int64)
                combined = _dedup_sort(combined)

            target_path.parent.mkdir(parents=True, exist_ok=True)

            # If target already exists, merge with it
            if target_path.exists():
                existing = pd.read_parquet(target_path)
                combined = pd.concat([existing, combined], ignore_index=True)
                if "ts" in combined.columns:
                    combined = _dedup_sort(combined)

            _atomic_write_parquet(combined, target_path)
            src_names = [s.name for s in sources]
            logger.info(f"  {' + '.join(src_names)} -> {target_key} ({len(combined)} rows)")
            migrated += 1

        # Migrate macro data (jsonl -> parquet)
        self._migrate_macro()

        logger.info(f"\nMigration complete: {migrated} tables migrated, {skipped} files skipped")
        self.info()

    def _migrate_macro(self):
        """迁移宏观数据: data/fear_greed.jsonl + data/fred_macro.jsonl -> store/macro/*.parquet。"""
        import json

        # Fear & Greed
        fng_path = PROJECT_ROOT / "data" / "fear_greed.jsonl"
        if fng_path.exists():
            records = []
            for line in fng_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
            if records:
                df = pd.DataFrame(records)
                if "ts" not in df.columns and "timestamp" in df.columns:
                    df["ts"] = df["timestamp"]
                target = self._path("macro", "fear_greed.parquet")
                target.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write_parquet(df, target)
                logger.info(f"  fear_greed.jsonl -> macro/fear_greed.parquet ({len(df)} rows)")

        # FRED Macro
        fred_path = PROJECT_ROOT / "data" / "fred_macro.jsonl"
        if fred_path.exists():
            records = []
            for line in fred_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
            if records:
                df = pd.DataFrame(records)
                target = self._path("macro", "fred_macro.parquet")
                target.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write_parquet(df, target)
                logger.info(f"  fred_macro.jsonl -> macro/fred_macro.parquet ({len(df)} rows)")


# ============================================================
# CLI
# ============================================================

def main():
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="DataStore — Unified Data Warehouse")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # info
    subparsers.add_parser("info", help="Show all tables status")

    # update
    update_parser = subparsers.add_parser("update", help="Update all data")
    update_parser.add_argument("--candles-only", action="store_true", help="Only update OHLCV candles")
    update_parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to update (default: BTC ETH SOL)")

    # migrate
    subparsers.add_parser("migrate", help="Migrate from data/cache to data/store")

    args = parser.parse_args()
    store = DataStore()

    if args.command == "info":
        store.info()
    elif args.command == "update":
        symbols = args.symbols or ["BTC", "ETH", "SOL"]
        if args.candles_only:
            for sym in symbols:
                store.update_candles(sym, bar="5m")
                store.update_candles(sym, bar="15m")
            store.info()
        else:
            store.update_all(symbols)
    elif args.command == "migrate":
        store.migrate_from_cache()


if __name__ == "__main__":
    main()
