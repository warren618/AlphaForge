"""因子注册表，自动发现 factors/ 下所有 Factor 子类。"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from .base import Factor
from src.logging_config import get_logger

logger = get_logger("registry")


class FactorRegistry:
    """因子注册与查找。

    自动扫描 src/factor_pipeline/factors/ 下所有模块，
    注册其中的 Factor 子类实例。

    Attributes:
        _factors: 已注册因子字典 {name: Factor}。

    Example:
        >>> registry = FactorRegistry()
        >>> registry.auto_discover()
        >>> registry.list_all()
        ['my_factor_1', 'my_factor_2', ...]
    """

    def __init__(self):
        """初始化空注册表。"""
        self._factors: dict[str, Factor] = {}

    def register(self, factor: Factor):
        """注册单个因子实例。

        Args:
            factor: Factor 子类实例。

        Raises:
            ValueError: 当因子名重复时抛出。
        """
        name = factor.meta.name
        if name in self._factors:
            raise ValueError(f"Factor '{name}' already registered")
        self._factors[name] = factor

    def auto_discover(self):
        """扫描 factors/ 目录下所有模块，注册 Factor 子类实例。"""
        factors_pkg = importlib.import_module("src.factor_pipeline.factors")
        factors_path = Path(factors_pkg.__file__).parent

        for _, module_name, _ in pkgutil.iter_modules([str(factors_path)]):
            module = importlib.import_module(f"src.factor_pipeline.factors.{module_name}")
            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, Factor)
                    and obj is not Factor
                ):
                    # 检查类级别是否定义了 meta (FactorMeta 实例)
                    class_meta = obj.__dict__.get("meta")
                    if class_meta is None:
                        continue
                    try:
                        instance = obj()
                        if instance.meta.name not in self._factors:
                            self.register(instance)
                    except Exception as e:
                        logger.warning(
                            f"Failed to instantiate factor {attr_name} "
                            f"from {module_name}: {e}"
                        )

    def get(self, name: str) -> Factor:
        """按名称获取因子。

        Args:
            name: 因子名称。

        Returns:
            对应的 Factor 实例。

        Raises:
            KeyError: 因子未注册时抛出。
        """
        if name not in self._factors:
            raise KeyError(f"Factor '{name}' not found. Available: {list(self._factors.keys())}")
        return self._factors[name]

    def list_all(self) -> list[str]:
        """返回所有已注册因子名称。

        Returns:
            因子名称列表。
        """
        return list(self._factors.keys())

    def list_by_source(self, source: str) -> list[Factor]:
        """按数据源筛选因子。

        Args:
            source: 数据源标识，如 "ohlcv"、"v12"。

        Returns:
            匹配的 Factor 实例列表。
        """
        return [f for f in self._factors.values() if f.meta.data_source == source]
