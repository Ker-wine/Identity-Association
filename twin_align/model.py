"""文件内容：本文件包含模型训练与评估。
主要职责：负责训练分类器、计算指标、保存模型 artifact。
前置文件：data.py、features.py、pairs.py、inference.py、io_utils.py。
后置文件：cli.py。
"""

import json
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .constants import FEATURE_NAMES, MODEL_VERSION
from .data import load_standard_data
from .features import make_profiles
from .inference import build_api_response, predict_pair
from .io_utils import save_json
from .pairs import make_training_pairs, materialize_pair_matrix


def evaluate_model(model: HistGradientBoostingClassifier, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """评估二分类模型效果。

    输入测试集特征 x 和真实标签 y，输出 accuracy、precision、recall、f1、auc。
    这些指标用来快速判断模型是否已经学到“同一人/非同一人”的区分能力。
    """
    # predict_proba 返回每个样本属于 0/1 类的概率，这里取 label=1 的概率作为匹配分数。
    prob = model.predict_proba(x)[:, 1]

    # 评估阶段用 0.5 做默认分类阈值；实际身份合并时会使用 mergeThreshold。
    pred = (prob >= 0.5).astype(int)
    metrics = {               
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }
    try:
        # AUC 要求测试集中同时存在正负样本；如果只有单一类别，就记为 nan。
        metrics["auc"] = roc_auc_score(y, prob)
    except ValueError:
        metrics["auc"] = float("nan")
    return {k: float(v) for k, v in metrics.items()}


def train_model(args: Any) -> None:
    """执行完整训练流程。

    训练流程包括：
    1. 读取标准 users.csv/posts.csv；
    2. 构建用户画像和 TF-IDF 文本矩阵；
    3. 构造正负账号对；
    4. 把账号对转换成特征矩阵；
    5. 训练 HistGradientBoostingClassifier；
    6. 保存模型 artifact、评估指标和一份 API 示例输出。
    """
    started = time.time()
    data_dir = Path(args.dataDir)
    out_dir = Path(args.outDir)
    out_dir.mkdir(parents=True, exist_ok=True)



    # 第一步：读取标准化后的 TWIN 数据。
    users, posts = load_standard_data(data_dir)

    # 第二步：把每个账号变成 UserProfile，同时得到文本 TF-IDF 矩阵。
    profiles, vectorizer, text_matrix = make_profiles(users, posts)

    # 第三步：根据 entityId 自动生成正样本，并随机采样负样本。
    pairs = make_training_pairs(users, args.negativeRatio, args.seed)

    # 第四步：把账号对转换成模型输入 x 和标签 y。
    x, y, _ = materialize_pair_matrix(pairs, profiles, text_matrix)

    # 数据足够时切分训练集/测试集；demo 数据太小时用同一批数据做 smoke 评估。
    class_counts = pd.Series(y).value_counts()
    can_split = len(y) >= 10 and class_counts.min() >= 2
    if can_split:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=args.testSize, stratify=y, random_state=args.seed
        )
    else:
        x_train, x_test, y_train, y_test = x, x, y, y

    # 第一版使用 scikit-learn 的梯度提升树，适合表格特征、依赖轻、不需要 GPU。
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=160,
        max_leaf_nodes=15,
        min_samples_leaf=1,
        l2_regularization=0.01,
        random_state=args.seed,
    )
    model.fit(x_train, y_train)
    metrics = evaluate_model(model, x_test, y_test)

    # artifact 保存了推理需要的全部对象：模型、文本向量器、用户画像和特征名。
    artifact = {
        "modelVersion": MODEL_VERSION,
        "model": model,
        "vectorizer": vectorizer,
        "text_matrix": text_matrix,
        "profiles": profiles,
        "featureNames": FEATURE_NAMES,
        "trainedAt": pd.Timestamp.utcnow().isoformat(),
        "metrics": metrics,
    }
    model_path = out_dir / "twin_align_baseline.joblib"
    joblib.dump(artifact, model_path)

    # 额外生成一份正样本的 API 返回示例，方便快速查看输出格式。
    demo_pred = []
    positive_pair = pairs[pairs["label"] == 1].iloc[0]
    demo_pred.append(
        predict_pair(
            artifact,
            positive_pair["platform_a"],
            positive_pair["user_a"],
            positive_pair["platform_b"],
            positive_pair["user_b"],
        )
    )
    response = build_api_response(
        args.taskId,
        demo_pred,
        args.mergeThreshold,
        int((time.time() - started) * 1000),
    )
    save_json(out_dir / "api_response_demo.json", response)
    save_json(out_dir / "metrics.json", metrics)

    print(f"Saved model: {model_path}")
    print(f"Saved API demo: {out_dir / 'api_response_demo.json'}")
    
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
