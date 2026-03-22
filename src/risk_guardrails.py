"""风控铁底层。

硬编码不可覆盖的风控限制，防止配置文件中设置不合理的参数。
任何 config 中超过铁底的值会被 clamp 到安全范围。
"""

from src.logging_config import get_logger

logger = get_logger("risk_guardrails")

# 硬编码风控铁底 — 不在 config 中，不可覆盖
RISK_FLOOR = {
    "max_loss_per_trade": 0.02,     # 单笔最大亏损 <= 2% of equity
    "max_daily_loss": 0.05,         # 日最大亏损 <= 5% of equity
    "max_drawdown": 0.30,           # 最大回撤 <= 30%
    "max_leverage": 20,             # 最大杠杆 <= 20x
}


def enforce_risk_floor(config: dict) -> dict:
    """强制风控铁底，将 config 中超限的值 clamp 到安全范围。

    Args:
        config: 运行时配置字典。

    Returns:
        修正后的配置字典（原地修改）。

    Example:
        >>> config = {"risk": {"max_drawdown": 0.90}}
        >>> enforce_risk_floor(config)
        >>> config["risk"]["max_drawdown"]
        0.3
    """
    risk = config.setdefault("risk", {})

    for key, floor_val in RISK_FLOOR.items():
        current = risk.get(key)
        if current is None:
            risk[key] = floor_val
            logger.info(f"Risk param '{key}' not set, defaulting to {floor_val}")
            continue

        current = float(current)
        if key == "max_leverage":
            # leverage: clamp 上限
            if current > floor_val:
                logger.warning(
                    f"Risk floor: {key}={current} exceeds max {floor_val}, clamped"
                )
                risk[key] = floor_val
        else:
            # loss/drawdown: clamp 上限（值越大越危险）
            if current > floor_val:
                logger.warning(
                    f"Risk floor: {key}={current} exceeds max {floor_val}, clamped"
                )
                risk[key] = floor_val

    return config
