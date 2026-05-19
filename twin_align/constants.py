"""文件内容：本文件包含项目全局常量。
主要职责：负责定义 MODEL_VERSION、EPS、FEATURE_NAMES。
前置文件：无。
后置文件：features.py、model.py、inference.py、api.py。
"""

MODEL_VERSION = "twin-align-baseline-0.1.0"
EPS = 1e-9

FEATURE_NAMES = [
    "username_similarity",
    "username_token_jaccard",
    "text_similarity",
    "image_content_similarity",
    "temporal_similarity",
    "hour_similarity",
    "weekday_similarity",
    "burstiness_similarity",
    "style_similarity",
    "post_frequency_diff",
    "post_count_diff",
    "active_hour_distance",
    "has_text_feature",
    "has_image_feature",
    "has_temporal_feature",
    "text_length_ratio",
]
