# -*- coding: utf-8 -*-
"""顶层入口：唯一复现命令 `python main.py`（在 PROJECT_ROOT 下执行）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cli import main  # noqa: E402

if __name__ == "__main__":
    main()