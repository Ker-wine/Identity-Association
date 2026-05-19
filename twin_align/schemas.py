"""文件内容：本文件包含核心数据结构。
主要职责：负责定义 UserProfile 用户画像结构。
前置文件：constants.py。
后置文件：features.py、pairs.py、inference.py。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class UserProfile:
    entity_id: str
    platform_id: str
    user_id: str
    username: str
    text: str
    text_vector_index: int
    image_vector: Optional[np.ndarray]
    post_count: int
    post_frequency: float
    hour_hist: np.ndarray
    weekday_hist: np.ndarray
    burstiness: float
    inter_event_mean: float
    inter_event_std: float
    style_vector: np.ndarray
