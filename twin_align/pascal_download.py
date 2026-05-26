"""文件内容：本文件包含 Pascal Sentences 数据下载与平台数据生成逻辑。
主要职责：负责下载前 10 类图片的前两张，组织为 platform_a/platform_b上传数据。
前置文件：依赖 Pascal Sentences 官网页面。
后置文件：被 cli.py 调用。
"""

from __future__ import annotations

import csv
import html
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


PASCAL_URL = "https://vision.cs.uiuc.edu/pascal-sentences/"
FIRST_TEN_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
]
UPLOAD_COLUMNS = ["userId", "postId", "imagePath", "text", "timestamp", "imageEmbedding"]


def fetch_text(url: str) -> str:
    """下载网页 HTML 文本。"""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_bytes(url: str) -> bytes:
    """下载图片二进制内容。"""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def clean_caption(value: str) -> str:
    """清理 HTML caption，得到普通文本。"""
    text = re.sub(r"<[^>]+>", " ", value)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_pascal_page(page_html: str) -> Dict[str, List[Dict[str, Any]]]:
    """解析官网页面，按类别收集图片路径和对应句子。"""
    records: Dict[str, List[Dict[str, Any]]] = {}
    pattern = re.compile(
        r'<tr>\s*<td><img src="([^"]+)"></td>\s*<td><table>(.*?)</table></td>\s*</tr>',
        re.DOTALL | re.IGNORECASE,
    )
    caption_pattern = re.compile(r"<tr><td>(.*?)</td></tr>", re.DOTALL | re.IGNORECASE)

    for match in pattern.finditer(page_html):
        image_src = match.group(1).strip()
        class_name = image_src.split("/", 1)[0]
        captions = [clean_caption(item.group(1)) for item in caption_pattern.finditer(match.group(2))]
        captions = [caption for caption in captions if caption]
        records.setdefault(class_name, []).append({"imageSrc": image_src, "captions": captions})
    return records


def write_upload_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    """写入平台 upload.csv。"""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=UPLOAD_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def download_pascal_data(args: Any) -> None:
    """命令行入口：下载 Pascal Sentences 并生成 A/B 平台上传数据。"""
    output_root = Path(args.outputRoot)
    platform_a_dir = output_root / args.platformADir
    platform_b_dir = output_root / args.platformBDir
    platform_a_dir.mkdir(parents=True, exist_ok=True)
    platform_b_dir.mkdir(parents=True, exist_ok=True)

    page_html = fetch_text(args.url)
    records = parse_pascal_page(page_html)

    platform_rows = {"A": [], "B": []}
    mapping_rows: List[Dict[str, str]] = []
    for user_index, class_name in enumerate(FIRST_TEN_CLASSES, start=1):
        images = records.get(class_name, [])
        if len(images) < 2:
            raise ValueError(f"Class {class_name} has fewer than 2 images on Pascal page.")

        for platform_key, platform_dir, item_index in [
            ("A", platform_a_dir, 0),
            ("B", platform_b_dir, 1),
        ]:
            item = images[item_index]
            image_src = item["imageSrc"]
            image_name = Path(image_src).name
            relative_path = f"images/user_{user_index:02d}/{image_name}"
            image_out = platform_dir / relative_path
            image_out.parent.mkdir(parents=True, exist_ok=True)
            image_out.write_bytes(fetch_bytes(args.url.rstrip("/") + "/" + image_src))

            post_id = f"{platform_key}{user_index:02d}_01"
            platform_rows[platform_key].append(
                {
                    "userId": str(user_index),
                    "postId": post_id,
                    "imagePath": relative_path,
                    "text": " ".join(item["captions"]),
                    "timestamp": "",
                    "imageEmbedding": "",
                }
            )
            mapping_rows.append(
                {
                    "userId": str(user_index),
                    "className": class_name,
                    "platform": platform_key,
                    "postId": post_id,
                    "imagePath": relative_path,
                    "captionCount": str(len(item["captions"])),
                }
            )

    write_upload_csv(platform_a_dir / "upload.csv", platform_rows["A"])
    write_upload_csv(platform_b_dir / "upload.csv", platform_rows["B"])

    with (output_root / "pascal_class_mapping.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["userId", "className", "platform", "postId", "imagePath", "captionCount"],
        )
        writer.writeheader()
        writer.writerows(mapping_rows)

    print(f"Wrote platform A data: {platform_a_dir}")
    print(f"Wrote platform B data: {platform_b_dir}")
    print(f"Wrote mapping: {output_root / 'pascal_class_mapping.csv'}")
