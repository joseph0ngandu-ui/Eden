import logging
import sys
from pathlib import Path


def configure_logging(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "eden.log", mode="a", encoding="utf-8"),
        ],
    )
