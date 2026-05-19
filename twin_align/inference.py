"""文件内容：本文件包含推理与 API 响应组装。
主要职责：负责单对预测、身份簇聚合、生成 entityId 和解释证据。
前置文件：constants.py、schemas.py、features.py。
后置文件：model.py、api.py、cli.py。
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .constants import FEATURE_NAMES, MODEL_VERSION
from .data import normalize_platform
from .features import build_pair_features


def stable_entity_id(members: Sequence[Tuple[str, str]]) -> str:
    joined = "|".join(f"{p}:{u}" for p, u in sorted(members))
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    return f"entity_{digest}"


def evidence_to_explanation(evidence: Dict[str, float]) -> List[str]:
    ranked = [
        ("username", evidence["username_similarity"]),
        ("text", evidence["text_similarity"]),
        ("temporal", evidence["temporal_similarity"]),
        ("style", evidence["style_similarity"]),
        ("image", evidence["image_content_similarity"]),
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [f"{name} similarity={score:.3f}" for name, score in ranked[:4]]


def predict_pair(
    artifact: Dict[str, Any],
    platform_a: str,
    user_a: str,
    platform_b: str,
    user_b: str,
) -> Dict[str, Any]:
    profiles = artifact["profiles"]
    text_matrix = artifact["text_matrix"]
    key_a = (normalize_platform(platform_a), str(user_a))
    key_b = (normalize_platform(platform_b), str(user_b))
    if key_a not in profiles:
        raise KeyError(f"Unknown user: {key_a}")
    if key_b not in profiles:
        raise KeyError(f"Unknown user: {key_b}")
    features, evidence = build_pair_features(profiles[key_a], profiles[key_b], text_matrix)
    score = float(artifact["model"].predict_proba(features.reshape(1, -1))[0, 1])
    return {
        "platformA": key_a[0],
        "userA": key_a[1],
        "platformB": key_b[0],
        "userB": key_b[1],
        "confidence": score,
        "features": {name: float(evidence[name]) for name in FEATURE_NAMES},
    }


def build_api_response(
    task_id: str,
    predictions: List[Dict[str, Any]],
    merge_threshold: float,
    inference_time_ms: int,
) -> Dict[str, Any]:
    parent: Dict[Tuple[str, str], Tuple[str, str]] = {}

    def find(x: Tuple[str, str]) -> Tuple[str, str]:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: Tuple[str, str], b: Tuple[str, str]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    accepted = [p for p in predictions if p["confidence"] >= merge_threshold]
    for pred in accepted:
        union((pred["platformA"], pred["userA"]), (pred["platformB"], pred["userB"]))

    groups: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for pred in accepted:
        for member in [(pred["platformA"], pred["userA"]), (pred["platformB"], pred["userB"])]:
            groups.setdefault(find(member), [])
            if member not in groups[find(member)]:
                groups[find(member)].append(member)

    clusters = []
    for members in groups.values():
        pair_scores = [
            p["confidence"]
            for p in accepted
            if (p["platformA"], p["userA"]) in members and (p["platformB"], p["userB"]) in members
        ]
        representative = max(
            [
                p
                for p in accepted
                if (p["platformA"], p["userA"]) in members and (p["platformB"], p["userB"]) in members
            ],
            key=lambda item: item["confidence"],
        )
        clusters.append(
            {
                "entityId": stable_entity_id(members),
                "confidence": float(np.mean(pair_scores)) if pair_scores else 0.0,
                "members": [
                    {"platformId": platform_id, "userId": user_id}
                    for platform_id, user_id in sorted(members)
                ],
                "alignmentEvidence": {
                    "pairScore": representative["confidence"],
                    "featureScores": representative["features"],
                },
                "physicalExplanation": evidence_to_explanation(representative["features"]),
            }
        )

    rejected_users = []
    for pred in predictions:
        if pred["confidence"] < merge_threshold:
            rejected_users.extend(
                [
                    {"platformId": pred["platformA"], "userId": pred["userA"]},
                    {"platformId": pred["platformB"], "userId": pred["userB"]},
                ]
            )
    seen = set()
    unmatched = []
    for item in rejected_users:
        key = (item["platformId"], item["userId"])
        if key not in seen:
            seen.add(key)
            unmatched.append(item)

    return {
        "taskId": task_id,
        "modelVersion": MODEL_VERSION,
        "inferenceTimeMs": inference_time_ms,
        "identityClusters": clusters,
        "unmatchedUsers": unmatched,
    }
