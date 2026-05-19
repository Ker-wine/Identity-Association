"""文件内容：本文件包含标准数据读取与基础字段处理。
主要职责：负责读取并校验 users.csv/posts.csv，统一平台和用户字段格式。
前置文件：constants.py。
后置文件：features.py、model.py。
"""

from pathlib import Path
from typing import Any, Tuple

import pandas as pd


def normalize_platform(value: Any) -> str:
    return str(value or "").strip().lower()


def load_standard_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    users_path = data_dir / "users.csv"
    posts_path = data_dir / "posts.csv"
    if not users_path.exists() or not posts_path.exists():
        raise FileNotFoundError(
            f"Expected {users_path} and {posts_path}. Run make-demo-data or convert TWIN first."
        )

    users = pd.read_csv(users_path).fillna("")
    posts = pd.read_csv(posts_path).fillna("")

    required_users = {"entityId", "platformId", "userId", "username"}
    required_posts = {"platformId", "userId", "postId", "text", "timestamp"}
    missing_users = required_users - set(users.columns)
    missing_posts = required_posts - set(posts.columns)
    if missing_users:
        raise ValueError(f"users.csv missing columns: {sorted(missing_users)}")
    if missing_posts:
        raise ValueError(f"posts.csv missing columns: {sorted(missing_posts)}")

    if "imagePath" not in posts.columns:
        posts["imagePath"] = ""
    if "imageEmbedding" not in posts.columns:
        posts["imageEmbedding"] = ""

    users["platformId"] = users["platformId"].map(normalize_platform)
    users["userId"] = users["userId"].astype(str)
    users["entityId"] = users["entityId"].astype(str)
    users["username"] = users["username"].astype(str)
    posts["platformId"] = posts["platformId"].map(normalize_platform)
    posts["userId"] = posts["userId"].astype(str)
    posts["text"] = posts["text"].astype(str)
    return users, posts
