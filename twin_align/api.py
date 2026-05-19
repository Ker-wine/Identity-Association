"""文件内容：本文件包含 HTTP 服务定义。
主要职责：负责创建 FastAPI app，提供 /health 和 /api/v1/identity/align。
前置文件：inference.py、constants.py。
后置文件：cli.py。
"""

import time
from typing import Any, Dict, List

import joblib

from .constants import MODEL_VERSION
from .inference import build_api_response, predict_pair


def create_app(model_path: str) -> Any:
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise SystemExit("Install optional API dependencies: pip install fastapi uvicorn") from exc

    artifact = joblib.load(model_path)
    app = FastAPI(title="TWIN Identity Alignment API", version=MODEL_VERSION)

    class Candidate(BaseModel):
        sourcePlatformId: str = Field(default="twitter")
        sourceUserId: str
        targetPlatformId: str = Field(default="instagram")
        targetUserId: str

    class AlignConfig(BaseModel):
        mergeThreshold: float = 0.85

    class AlignRequest(BaseModel):
        taskId: str = "demo-task"
        candidates: List[Candidate]
        alignConfig: AlignConfig = Field(default_factory=AlignConfig)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "modelVersion": MODEL_VERSION}

    @app.post("/api/v1/identity/align")
    def align(request: AlignRequest) -> Dict[str, Any]:
        started = time.time()
        predictions = [
            predict_pair(
                artifact,
                item.sourcePlatformId,
                item.sourceUserId,
                item.targetPlatformId,
                item.targetUserId,
            )
            for item in request.candidates
        ]
        return build_api_response(
            request.taskId,
            predictions,
            request.alignConfig.mergeThreshold,
            int((time.time() - started) * 1000),
        )

    return app


def serve(model_path: str, host: str, port: int) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install optional API dependencies: pip install fastapi uvicorn") from exc

    uvicorn.run(create_app(model_path), host=host, port=port)
