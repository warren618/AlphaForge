"""因子流水线包。

提供 Factor ABC、FactorRegistry、FactorEvaluator、FactorPipeline。
"""

from __future__ import annotations

from .base import Factor, FactorMeta
from .evaluator import FactorEvaluator
from .pipeline import FactorPipeline
from .registry import FactorRegistry

__all__ = [
    "Factor",
    "FactorMeta",
    "FactorEvaluator",
    "FactorPipeline",
    "FactorRegistry",
]
