"""结构化日志工厂。

提供统一的 JSON 格式日志，console + rotating file handler。
所有模块通过 `get_logger(name)` 获取 logger 实例。
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


LOG_DIR = Path(__file__).parent.parent / "data" / "logs"


class _JsonFormatter(logging.Formatter):
    """JSON 格式化器。"""

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为 JSON 行。

        Args:
            record: 日志记录对象。

        Returns:
            JSON 字符串。
        """
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_data"):
            entry["data"] = record.extra_data
        return json.dumps(entry, ensure_ascii=False)


class _ConsoleFormatter(logging.Formatter):
    """简洁的控制台格式化器。"""

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为可读行。

        Args:
            record: 日志记录对象。

        Returns:
            格式化字符串。
        """
        ts = datetime.now().strftime("%H:%M:%S")
        return f"{ts} [{record.levelname[0]}] {record.name}: {record.getMessage()}"


_initialized: set[str] = set()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """获取结构化 logger 实例。

    Args:
        name: logger 名称，通常为模块名。
        level: 日志级别，默认 INFO。

    Returns:
        配置好的 Logger 实例。

    Example:
        >>> logger = get_logger("backtest")
        >>> logger.info("Trade executed", extra={"extra_data": {"symbol": "BTC"}})
    """
    if name in _initialized:
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(_ConsoleFormatter())
    logger.addHandler(console)

    # Rotating file handler (JSON)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / f"{name}.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(_JsonFormatter())
    logger.addHandler(file_handler)

    _initialized.add(name)
    return logger
