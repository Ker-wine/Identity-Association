"""文件内容：本文件包含命令行入口。
主要职责：负责注册 make-demo-data/download-pascal-data/prepare-upload-data/train/predict/predict-all/serve 子命令。
前置文件：model.py、inference.py、api.py、demo_data.py、pascal_download.py、platform_import.py。
后置文件：twin_align_baseline.py。
"""

import argparse
import json
import time
from pathlib import Path

import joblib

from .api import serve
from .demo_data import make_demo_data
from .inference import build_api_response, predict_pair
from .io_utils import save_json
from .model import train_model
from .pascal_download import PASCAL_URL, download_pascal_data
from .platform_import import prepare_platform_data


def cmd_predict(args: argparse.Namespace) -> None:
    started = time.time()
    artifact = joblib.load(args.modelPath)
    pred = predict_pair(
        artifact,
        args.platformA,
        args.userA,
        args.platformB,
        args.userB,
    )
    response = build_api_response(
        args.taskId,
        [pred],
        args.mergeThreshold,
        int((time.time() - started) * 1000),
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))


def cmd_predict_all(args: argparse.Namespace) -> None:
    started = time.time()
    artifact = joblib.load(args.modelPath)
    profiles = artifact["profiles"]
    left = [p for p in profiles.values() if p.platform_id == args.platformA]
    right = [p for p in profiles.values() if p.platform_id == args.platformB]
    predictions = [
        predict_pair(artifact, a.platform_id, a.user_id, b.platform_id, b.user_id)
        for a in left
        for b in right
    ]
    response = build_api_response(
        args.taskId,
        predictions,
        args.mergeThreshold,
        int((time.time() - started) * 1000),
    )
    if args.out:
        save_json(Path(args.out), response)
    print(json.dumps(response, ensure_ascii=False, indent=2))


def cmd_serve(args: argparse.Namespace) -> None:
    serve(args.modelPath, args.host, args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TWIN cross-platform identity alignment baseline")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("make-demo-data")
    demo.add_argument("--outDir", default="./data/twin_std")
    demo.set_defaults(func=make_demo_data)

    pascal = sub.add_parser("download-pascal-data")
    pascal.add_argument("--url", default=PASCAL_URL)
    pascal.add_argument("--outputRoot", default="./data")
    pascal.add_argument("--platformADir", default="platform_a")
    pascal.add_argument("--platformBDir", default="platform_b")
    pascal.set_defaults(func=download_pascal_data)

    prepare = sub.add_parser("prepare-upload-data")
    prepare.add_argument("--inputRoot", default="./data")
    prepare.add_argument("--platformADir", default="platform_a")
    prepare.add_argument("--platformBDir", default="platform_b")
    prepare.add_argument("--inputFile", default="upload.csv")
    prepare.add_argument("--platformAId", default="a_platform")
    prepare.add_argument("--platformBId", default="b_platform")
    prepare.add_argument("--outDir", default="./data/twin_std")
    prepare.set_defaults(func=prepare_platform_data)

    train = sub.add_parser("train")
    train.add_argument("--dataDir", default="./data/twin_std")
    train.add_argument("--outDir", default="./runs/twin_align")
    train.add_argument("--negativeRatio", type=int, default=3)
    train.add_argument("--mergeThreshold", type=float, default=0.85)
    train.add_argument("--testSize", type=float, default=0.25)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--taskId", default="demo-task-001")
    train.set_defaults(func=train_model)

    predict = sub.add_parser("predict")
    predict.add_argument("--modelPath", default="./runs/twin_align/twin_align_baseline.joblib")
    predict.add_argument("--platformA", default="twitter")
    predict.add_argument("--userA", required=True)
    predict.add_argument("--platformB", default="instagram")
    predict.add_argument("--userB", required=True)
    predict.add_argument("--taskId", default="demo-task-001")
    predict.add_argument("--mergeThreshold", type=float, default=0.85)
    predict.set_defaults(func=cmd_predict)

    predict_all = sub.add_parser("predict-all")
    predict_all.add_argument("--modelPath", default="./runs/twin_align/twin_align_baseline.joblib")
    predict_all.add_argument("--platformA", default="twitter")
    predict_all.add_argument("--platformB", default="instagram")
    predict_all.add_argument("--taskId", default="demo-task-all")
    predict_all.add_argument("--mergeThreshold", type=float, default=0.85)
    predict_all.add_argument("--out", default="")
    predict_all.set_defaults(func=cmd_predict_all)

    serve_parser = sub.add_parser("serve")
    serve_parser.add_argument("--modelPath", default="./runs/twin_align/twin_align_baseline.joblib")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.set_defaults(func=cmd_serve)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
