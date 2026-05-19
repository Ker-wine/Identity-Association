
"""文件内容：本文件包含身份对齐特征工程。
主要职责：负责构建用户画像，计算文本、图像、时间、风格和用户名相似度。
前置文件：constants.py、schemas.py、data.py。
后置文件：pairs.py、inference.py。
"""

from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .constants import EPS, FEATURE_NAMES
from .schemas import UserProfile


def parse_embedding(value: Any) -> Optional[np.ndarray]:
    """解析帖子中的 imageEmbedding 字段。

    输入可能是空值、Python list、numpy 数组、JSON 字符串或类似 "[0.1, 0.2]" 的文本。
    解析成功时返回一维 numpy 向量；解析失败或字段为空时返回 None。
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float)
        return arr if arr.size else None

    text = str(value).strip()
    if not text:
        return None
    try:   # 先尝试当成 JSON 解析，适合标准的 JSON 数组字符串。
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None
    try:  # 最后把解析结果转成 numpy 向量，要求是一维且非空。
        arr = np.asarray(parsed, dtype=float)     # 适合解析 "[0.1, 0.2]" 这种文本格式的向量。
    except (TypeError, ValueError):            # 解析失败可能是因为文本格式不对，或者解析结果不是数值列表。
        return None
    if arr.ndim != 1 or arr.size == 0:        # 解析成功但不是一维向量，或者向量长度为 0，都视作无效输入。
        return None
    return arr


def safe_cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """安全计算两个向量的余弦相似度。

    如果任意向量为空、维度不同或向量长度为 0，就返回 0.0，避免训练和推理时崩溃。
    余弦相似度越接近 1，表示两个向量越相似。
    """
    if a is None or b is None:
        return 0.0
    if len(a) != len(b):
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))  # 计算两个向量的模长乘积，作为余弦相似度的分母。如果模长为 0，说明至少有一个向量是零向量，无法计算相似度。
    if denom <= EPS:
        return 0.0
    return float(np.dot(a, b) / denom)


def sparse_cosine(matrix: sparse.spmatrix, idx_a: int, idx_b: int) -> float: 
    """计算稀疏矩阵中两行 TF-IDF 向量的余弦相似度。

    text_matrix 里的每一行代表一个用户的文本向量。
    idx_a 和 idx_b 是两个用户在矩阵中的行号，用来比较两个用户发帖文本的主题相似度。
    """
    a = matrix[idx_a]
    b = matrix[idx_b]
    denom = math.sqrt(float(a.multiply(a).sum())) * math.sqrt(float(b.multiply(b).sum()))
    if denom <= EPS:
        return 0.0
    return float(a.multiply(b).sum() / denom)


def normalized_hist(values: Sequence[int], bins: int) -> np.ndarray:   
    """把离散取值统计成归一化直方图。

    例如小时取值 0-23 会变成 24 维分布，星期取值 0-6 会变成 7 维分布。
    归一化后，所有格子的和为 1，方便比较两个用户的活跃时间习惯。
    """
    hist = np.zeros(bins, dtype=float)
    for value in values:
        if 0 <= int(value) < bins:
            hist[int(value)] += 1.0
    total = hist.sum()
    return hist / total if total > 0 else hist


def style_features(text: str) -> np.ndarray:
    """提取用户文本的轻量写作风格特征。

    这些特征不关心文本具体说了什么，而是看用户怎么写：
    平均句长、标点比例、数字比例、大写比例、特殊字符比例、平均词长和词数量。
    """
    chars = list(text or "")
    total_chars = max(len(chars), 1)
    tokens = re.findall(r"\b\w+\b", text.lower())
    token_count = max(len(tokens), 1)
    punct = sum(1 for c in chars if c in ".,!?;:'\"()[]{}")
    digits = sum(1 for c in chars if c.isdigit())
    uppers = sum(1 for c in chars if c.isupper())
    special = sum(1 for c in chars if (not c.isalnum() and not c.isspace() and c not in ".,!?;:'\"()[]{}"))
    avg_word_len = sum(len(t) for t in tokens) / token_count
    sentences = max(len(re.findall(r"[.!?]+", text)), 1)
    avg_sentence_len = token_count / sentences
    return np.asarray(
        [
            avg_sentence_len,
            punct / total_chars,
            digits / total_chars,
            uppers / total_chars,
            special / total_chars,
            avg_word_len,
            token_count,
        ],
        dtype=float,
    )


def make_profiles(
    users: pd.DataFrame, posts: pd.DataFrame
) -> Tuple[Dict[Tuple[str, str], UserProfile], TfidfVectorizer, sparse.spmatrix]:
    """把 users.csv 和 posts.csv 转换成用户画像集合。

    每个 UserProfile 表示一个平台账号，里面包含：
    用户名、合并后的发帖文本、图片均值向量、发帖数量、发帖频率、时间分布和写作风格。
    函数同时训练 TF-IDF 向量器，并返回所有用户的文本向量矩阵。
    """
    # 先按 平台 + 用户ID 把帖子分组，后面可以快速找到某个账号的全部帖子。
    grouped_posts = {
        key: group.copy()
        for key, group in posts.groupby(["platformId", "userId"], dropna=False)
    }

    raw_profiles: List[Dict[str, Any]] = []
    all_texts: List[str] = []
    for _, user in users.iterrows():
        # 取出当前账号的所有帖子，并把帖子文本合并成一个用户级文本。
        key = (user["platformId"], str(user["userId"]))
        group = grouped_posts.get(key, pd.DataFrame(columns=posts.columns))
        text = " ".join(group.get("text", pd.Series(dtype=str)).astype(str).tolist()).strip()

        # 将时间戳转成小时分布、星期分布和发帖间隔统计，用来刻画用户活跃习惯。
        timestamps = pd.to_datetime(group.get("timestamp", pd.Series(dtype=str)), errors="coerce").dropna()
        hours = timestamps.dt.hour.astype(int).tolist() if len(timestamps) else []
        weekdays = timestamps.dt.weekday.astype(int).tolist() if len(timestamps) else []
        sorted_ts = timestamps.sort_values()
        if len(sorted_ts) >= 2:
            deltas = sorted_ts.diff().dropna().dt.total_seconds().to_numpy(dtype=float)
            inter_mean = float(np.mean(deltas))
            inter_std = float(np.std(deltas))
            burstiness = (inter_std - inter_mean) / (inter_std + inter_mean + EPS)
            duration_days = max((sorted_ts.max() - sorted_ts.min()).total_seconds() / 86400.0, 1.0)
        else:
            inter_mean = 0.0
            inter_std = 0.0
            burstiness = 0.0
            duration_days = 1.0

        # 如果 posts.csv 已经提供 imageEmbedding，就把该用户所有图片向量做平均池化。
        embeddings = [parse_embedding(v) for v in group.get("imageEmbedding", pd.Series(dtype=str)).tolist()]
        embeddings = [v for v in embeddings if v is not None]
        image_vector = None
        if embeddings:
            dim = max(set(len(v) for v in embeddings), key=[len(v) for v in embeddings].count)
            same_dim = [v for v in embeddings if len(v) == dim]
            image_vector = np.mean(np.vstack(same_dim), axis=0)

        post_count = int(len(group))
        raw_profiles.append(
            {
                "entity_id": str(user["entityId"]),
                "platform_id": user["platformId"],
                "user_id": str(user["userId"]),
                "username": str(user.get("username", "")),
                "text": text,
                "image_vector": image_vector,
                "post_count": post_count,
                "post_frequency": post_count / duration_days,
                "hour_hist": normalized_hist(hours, 24),
                "weekday_hist": normalized_hist(weekdays, 7),
                "burstiness": float(burstiness),
                "inter_event_mean": inter_mean,
                "inter_event_std": inter_std,
                "style_vector": style_features(text),
            }
        )
        all_texts.append(text if text else " ")

    # 用所有用户的合并文本训练 TF-IDF，每个用户得到一行文本向量。
    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=30000,
        token_pattern=r"(?u)\b\w+\b",
    )
    text_matrix = vectorizer.fit_transform(all_texts)

    profiles: Dict[Tuple[str, str], UserProfile] = {}
    for idx, item in enumerate(raw_profiles):
        # text_vector_index 记录该用户在 text_matrix 中对应哪一行。
        profile = UserProfile(text_vector_index=idx, **item)
        profiles[(profile.platform_id, profile.user_id)] = profile
    return profiles, vectorizer, text_matrix


def sequence_similarity(a: str, b: str) -> float:
    """计算两个用户名的整体字符串相似度。

    会先去掉非字母数字字符并转成小写，再用 SequenceMatcher 计算相似度。
    适合比较 alice_tw 和 alice_ig 这种整体结构相近的用户名。
    """
    from difflib import SequenceMatcher

    a_norm = re.sub(r"[^a-z0-9]+", "", (a or "").lower())
    b_norm = re.sub(r"[^a-z0-9]+", "", (b or "").lower())
    if not a_norm or not b_norm:
        return 0.0
    return float(SequenceMatcher(None, a_norm, b_norm).ratio())


def token_jaccard(a: str, b: str) -> float:
    """计算两个用户名的三字符片段 Jaccard 相似度。

    先把用户名拆成连续 3 个字符的小片段，再比较两个片段集合的交并比。
    适合捕捉用户名中局部相同的部分，例如 claire_code 和 claire_dev。
    """
    def grams(value: str) -> set:
        value = re.sub(r"[^a-z0-9]+", "", (value or "").lower())
        if len(value) <= 2:
            return {value} if value else set()
        return {value[i : i + 3] for i in range(len(value) - 2)}

    ga = grams(a)
    gb = grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def temporal_similarity(a: UserProfile, b: UserProfile) -> Tuple[float, float, float, float]:
    """计算两个用户的时间行为相似度。

    综合三部分信息：
    1. 24 小时活跃分布相似度；
    2. 一周 7 天活跃分布相似度；
    3. burstiness 发帖爆发性相似度。
    返回总时间相似度和三个子分数，供模型和解释信息使用。
    """
    hour_sim = safe_cosine(a.hour_hist, b.hour_hist)
    weekday_sim = safe_cosine(a.weekday_hist, b.weekday_hist)
    burst_sim = math.exp(-abs(a.burstiness - b.burstiness))
    total = 0.5 * hour_sim + 0.3 * weekday_sim + 0.2 * burst_sim
    return float(total), float(hour_sim), float(weekday_sim), float(burst_sim)


def build_pair_features(
    a: UserProfile, b: UserProfile, text_matrix: sparse.spmatrix
) -> Tuple[np.ndarray, Dict[str, float]]:
    """构造两个账号之间的候选对特征。

    输入两个 UserProfile，输出两份内容：
    1. numpy 特征向量：喂给机器学习模型训练或预测；
    2. 字典形式的特征分数：用于 API 输出和人工解释。
    """
    # 文本、图像、时间和风格是四类主要相似度证据。
    text_sim = sparse_cosine(text_matrix, a.text_vector_index, b.text_vector_index)
    img_sim = safe_cosine(a.image_vector, b.image_vector)
    temp_sim, hour_sim, weekday_sim, burst_sim = temporal_similarity(a, b)
    style_sim = safe_cosine(a.style_vector, b.style_vector)

    # 差异类特征越小越像，用 log1p 缩小极端发帖数量/频率带来的影响。
    hour_distance = float(np.linalg.norm(a.hour_hist - b.hour_hist))
    post_freq_diff = abs(math.log1p(a.post_frequency) - math.log1p(b.post_frequency))
    post_count_diff = abs(math.log1p(a.post_count) - math.log1p(b.post_count))
    text_len_ratio = min(len(a.text), len(b.text)) / max(max(len(a.text), len(b.text)), 1)

    # values 的 key 必须和 FEATURE_NAMES 保持一致，最后会按固定顺序组装成模型输入。
    values = {
        "username_similarity": sequence_similarity(a.username, b.username),
        "username_token_jaccard": token_jaccard(a.username, b.username),
        "text_similarity": text_sim,
        "image_content_similarity": img_sim,
        "temporal_similarity": temp_sim,
        "hour_similarity": hour_sim,
        "weekday_similarity": weekday_sim,
        "burstiness_similarity": burst_sim,
        "style_similarity": style_sim,
        "post_frequency_diff": post_freq_diff,
        "post_count_diff": post_count_diff,
        "active_hour_distance": hour_distance,
        "has_text_feature": float(bool(a.text.strip()) and bool(b.text.strip())),
        "has_image_feature": float(a.image_vector is not None and b.image_vector is not None),
        "has_temporal_feature": float(a.post_count > 0 and b.post_count > 0),
        "text_length_ratio": float(text_len_ratio),
    }
    return np.asarray([values[name] for name in FEATURE_NAMES], dtype=float), values
