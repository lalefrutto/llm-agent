"""
main.py — FastAPI-обёртка над agent/graph.py (контейнер agent-app).
Запуск: uvicorn agent.main:app --host 0.0.0.0 --port 8080

Эндпоинты:
  GET  /         — минимальный веб-чат (одна HTML-страница, без сборки)
  POST /chat     — JSON API {user_id, thread_id?, message} -> {thread_id, answer, route_history, step_count}
  GET  /metrics  — Prometheus (см. agent/metrics.py, observability/README.md)
  GET  /health   — liveness для docker healthcheck
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from agent import metrics as M
from agent.graph import build_graph, run_request

_graph = None  # ленивая инициализация — граф требует доступный Ollama
_CHAT_HTML = (Path(__file__).parent / "static" / "chat.html").read_text(encoding="utf-8")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _graph
    _graph = build_graph()
    yield


app = FastAPI(title="Atlas Agent API", lifespan=_lifespan)


class ChatRequest(BaseModel):
    user_id: str
    thread_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    route_history: list[str] = []
    step_count: int = 0
    latency_s: float = 0.0


@app.get("/", response_class=HTMLResponse)
def index():
    return _CHAT_HTML


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    thread_id = req.thread_id or str(uuid.uuid4())
    t0 = time.perf_counter()
    try:
        result = run_request(_graph, req.user_id, thread_id, req.message)
    except Exception as e:  # noqa: BLE001
        M.REQUESTS_TOTAL.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e
    M.REQUESTS_TOTAL.labels(status="ok").inc()
    return ChatResponse(
        thread_id=thread_id,
        answer=result["final_answer"] or "",
        route_history=result.get("route_history") or [],
        step_count=result.get("step_count", 0),
        latency_s=round(time.perf_counter() - t0, 2),
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok", "graph_ready": _graph is not None}
