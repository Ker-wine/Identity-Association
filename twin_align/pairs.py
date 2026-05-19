 """文件内容：本文件包含候选账号对构造逻辑。
主要职责：负责生成正负样本、构造训练矩阵。
前置文件：schemas.py、features.py。
后置文件：model.py。
"""

import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from .data import normalize_platform
from .features import build_pair_features
from .schemas import UserProfile


def make_training_pairs(users: pd.DataFrame, negative_ratio: int, seed: int) -> pd.DataFrame:
    """根据 users.csv 构造训练用的账号对。

    输出的每一行表示一个 Twitter-Instagram 候选对：
    label=1 表示两个账号的 entityId 相同，是同一个人；
    label=0 表示两个账号的 entityId 不同，不是同一个人。
    negative_ratio 控制每个 Twitter 账号随机采样多少个负样本。
    """
    # 第一版先只处理 TWIN 的 twitter-instagram 双平台匹配任务。
    twitter = users[users["platformId"] == "twitter"].copy()
    instagram = users[users["platformId"] == "instagram"].copy()

    # 相同 entityId 的 Twitter 和 Instagram 账号天然就是正样本。
    positives = twitter.merge(instagram, on="entityId", suffixes=("_a", "_b"))
    rows: List[Dict[str, Any]] = []
    for _, row in positives.iterrows():
        rows.append(
            {
                "platform_a": row["platformId_a"],
                "user_a": str(row["userId_a"]),
                "platform_b": row["platformId_b"],
                "user_b": str(row["userId_b"]),
                "label": 1,
            }
        )

    # 负样本从不同 entityId 的账号中随机抽取，用 seed 保证每次采样可复现。
    rng = random.Random(seed)
    ig_records = instagram[["entityId", "platformId", "userId"]].to_dict("records")
    for _, tw in twitter.iterrows():
        candidates = [ig for ig in ig_records if ig["entityId"] != tw["entityId"]]
        rng.shuffle(candidates)
        for ig in candidates[: max(1, negative_ratio)]:
            rows.append(
                {
                    "platform_a": tw["platformId"],
                    "user_a": str(tw["userId"]),
                    "platform_b": ig["platformId"],
                    "user_b": str(ig["userId"]),
                    "label": 0,
                }
            )

    # 去重后打乱顺序，避免模型训练时先看到一批正样本再看到一批负样本。
    pairs = pd.DataFrame(rows).drop_duplicates(["platform_a", "user_a", "platform_b", "user_b"])
    if pairs["label"].nunique() < 2:
        raise ValueError("Need at least one positive pair and one negative pair for training.")
    return pairs.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def materialize_pair_matrix(
    pairs: pd.DataFrame,
    profiles: Dict[Tuple[str, str], UserProfile],
    text_matrix: sparse.spmatrix,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """把候选账号对转换成模型能训练的矩阵。

    输入 pairs 是账号对清单，profiles 是每个账号的用户画像。
    输出 x 是特征矩阵，y 是标签数组，evidence 是每个账号对的可解释特征分数字典。
    """
    xs: List[np.ndarray] = []
    ys: List[int] = []
    evidence: List[Dict[str, float]] = []
    skipped = 0
    for _, row in pairs.iterrows():
        # 用 平台 + 用户ID 定位对应的 UserProfile。
        key_a = (normalize_platform(row["platform_a"]), str(row["user_a"]))
        key_b = (normalize_platform(row["platform_b"]), str(row["user_b"]))
        if key_a not in profiles or key_b not in profiles:
            skipped += 1
            continue

        # build_pair_features 会把两个账号画像合成一个固定长度的特征向量。
        features, values = build_pair_features(profiles[key_a], profiles[key_b], text_matrix)
        xs.append(features)
        ys.append(int(row["label"]))
        evidence.append(values)
    if skipped:
        print(f"Skipped {skipped} pairs because profiles were missing.")
    return np.vstack(xs), np.asarray(ys, dtype=int), evidence
