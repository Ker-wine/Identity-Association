"""文件内容：本文件包含 platform_a/platform_b上传数据转标准数据的逻辑。
主要职责：负责读取每个平台的 upload.csv，并生成训练需要的 users.csv/posts.csv。
前置文件：依赖 data/platform_a/upload.csv、data/platform_b/upload.csv。
后置文件：被 cli.py 调用。
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


REQUIRED_UPLOAD_COLUMNS = {"userId", "postId", "imagePath", "text"}
OPTIONAL_UPLOAD_COLUMNS = {"timestamp", "imageEmbedding"}
OUTPUT_POST_COLUMNS = [
    "platformId",
    "userId",
    "postId",
    "text",
    "timestamp",
    "imagePath",
    "imageEmbedding",
]


def make_entity_id(user_id: Any) -> str:
    """把 A/B 两个平台相同的用户编号映射为同一个 entityId。"""
    value = str(user_id).strip()
    if value.isdigit():
        return f"u{int(value):03d}"
    return f"u_{value}"


def make_username(platform_id: str, user_id: Any) -> str:
    """为上传数据生成一个稳定的默认用户名。"""
    return f"{platform_id}_user_{str(user_id).strip()}"


def resolve_image_path(platform_dir: Path, image_path: Any) -> str:
    """把相对图片路径转换为以平台目录为基准的路径，绝对路径保持不变。"""
    text = str(image_path or "").strip()
    if not text:
        return ""
    path = Path(text)
    if path.is_absolute():
        return str(path)
    return str(platform_dir / path)


def load_platform_upload(platform_dir: Path, platform_id: str, file_name: str) -> pd.DataFrame:
    """读取单个平台的 upload.csv，并补齐可选字段。"""
    upload_path = platform_dir / file_name
    if not upload_path.exists():
        raise FileNotFoundError(f"Upload file not found: {upload_path}")

    frame = pd.read_csv(upload_path).fillna("")
    missing = REQUIRED_UPLOAD_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{upload_path} missing columns: {sorted(missing)}")

    for column in OPTIONAL_UPLOAD_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""

    frame["platformId"] = platform_id
    frame["userId"] = frame["userId"].astype(str).str.strip()
    frame["postId"] = frame["postId"].astype(str).str.strip()
    frame["text"] = frame["text"].astype(str)
    frame["timestamp"] = frame["timestamp"].astype(str)
    frame["imagePath"] = frame["imagePath"].map(lambda value: resolve_image_path(platform_dir, value))
    frame["imageEmbedding"] = frame["imageEmbedding"].astype(str)

    empty_post_ids = frame["postId"].eq("")
    if empty_post_ids.any():
        frame.loc[empty_post_ids, "postId"] = [
            f"{platform_id}_{idx + 1:04d}" for idx in range(int(empty_post_ids.sum()))
        ]

    return frame


def build_standard_tables(
    platform_a: pd.DataFrame,
    platform_b: pd.DataFrame,
    platform_a_id: str,
    platform_b_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """把两个平台的上传表转换成标准 users/posts 两张表。"""
    all_posts = pd.concat([platform_a, platform_b], ignore_index=True)
    all_posts = all_posts[all_posts["userId"].astype(str).str.strip() != ""].copy()

    users_rows: List[Dict[str, str]] = []
    for platform_id, group in all_posts.groupby("platformId", sort=True):
        for user_id in sorted(group["userId"].unique(), key=lambda value: str(value)):
            users_rows.append(
                {
                    "entityId": make_entity_id(user_id),
                    "platformId": platform_id,
                    "userId": str(user_id),
                    "username": make_username(platform_id, user_id),
                }
            )

    users = pd.DataFrame(users_rows, columns=["entityId", "platformId", "userId", "username"])
    posts = all_posts[OUTPUT_POST_COLUMNS].copy()

    expected_entities = {
        make_entity_id(user_id)
        for user_id in [*platform_a["userId"].tolist(), *platform_b["userId"].tolist()]
        if str(user_id).strip()
    }
    if not expected_entities:
        raise ValueError("No users found in uploaded platform files.")

    observed_platforms = set(posts["platformId"].unique())
    missing_platforms = {platform_a_id, platform_b_id} - observed_platforms
    if missing_platforms:
        raise ValueError(f"No valid posts found for platforms: {sorted(missing_platforms)}")

    return users, posts


def prepare_platform_data(args: Any) -> None:
    """命令行入口：生成 data/twin_std/users.csv 和 data/twin_std/posts.csv。"""
    input_root = Path(args.inputRoot)
    platform_a_dir = input_root / args.platformADir
    platform_b_dir = input_root / args.platformBDir
    out_dir = Path(args.outDir)
    out_dir.mkdir(parents=True, exist_ok=True)

    platform_a = load_platform_upload(platform_a_dir, args.platformAId, args.inputFile)
    platform_b = load_platform_upload(platform_b_dir, args.platformBId, args.inputFile)
    users, posts = build_standard_tables(platform_a, platform_b, args.platformAId, args.platformBId)

    users.to_csv(out_dir / "users.csv", index=False)
    posts.to_csv(out_dir / "posts.csv", index=False)

    print(f"Wrote standard users: {out_dir / 'users.csv'}")
    print(f"Wrote standard posts: {out_dir / 'posts.csv'}")
    print(f"Users: {len(users)}, posts: {len(posts)}")
