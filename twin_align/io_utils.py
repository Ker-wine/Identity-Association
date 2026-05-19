"""文件内容：本文件包含通用 I/O 工具。
主要职责：负责保存 JSON 等小工具函数。
前置文件：无。
后置文件：model.py、cli.py。
"""

import json
from pathlib import Path
from typing import Any, Dict


def save_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
