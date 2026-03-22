"""因子基类与元数据定义。

Factor ABC 只负责计算，不负责评分。
评分规则在 config.yaml 的 scoring_rules 里定义。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FactorMeta:
    """因子元数据。

    Attributes:
        name: 因子唯一标识，如 "basis_dislocation"。
        family: 因子族，对应 factor_library/ 子目录，如 "derivatives"。
        output_columns: 因子计算输出的所有列名。
        score_column: 用于评测和策略评分的主列。
        data_source: 数据来源，"ohlcv" | "v12" | "derived"。
    """

    name: str
    family: str
    output_columns: list[str] = field(default_factory=list)
    score_column: str = ""
    data_source: str = "ohlcv"


class Factor(ABC):
    """因子抽象基类。

    子类必须定义 meta 属性和 compute() 方法。
    因子只负责计算，不定义评分逻辑。

    Attributes:
        meta: 因子元数据。

    Example:
        >>> class MyFactor(Factor):
        ...     meta = FactorMeta(name="my_factor", family="trend",
        ...                       output_columns=["my_col"], score_column="my_col")
        ...     def compute(self, df, cfg, extra_data=None):
        ...         df["my_col"] = df["close"].pct_change()
        ...         return df
    """

    meta: FactorMeta

    @abstractmethod
    def compute(self, df: pd.DataFrame, cfg: dict, extra_data: dict | None = None) -> pd.DataFrame:
        """追加因子列到 df 并返回。

        Args:
            df: 含 OHLCV 的主 DataFrame。
            cfg: 因子参数配置字典（config.yaml 的 factors 部分）。
            extra_data: 额外数据源，如 {"index_candles": df_index}。

        Returns:
            追加了因子列的 DataFrame。
        """
