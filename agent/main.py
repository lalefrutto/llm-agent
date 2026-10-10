"""
main.py — FastAPI-обёртка над agent/graph.py (контейнер agent-app).
Запуск: uvicorn agent.main:app --host 0.0.0.0 --port 8080

Эндпоинты:
  GET  /         — минимальный веб-чат (одна HTML-страница, без сборки)
  POST /chat     — JSON API {user_id, thread_id?, message} -> {thread_id, answer, route_history, step_count, trace}
  GET  /threads?user_id=…  — список тредов пользователя (название, время), новые сверху
  GET  /threads/{id}       — история треда из checkpointer'а (для переключения тредов)
  PATCH /threads/{id}      — переименовать тред {user_id, title}
  DELETE /threads/{id}?user_id=…&forget_memory=0|1 — удалить тред (и, по желанию, факты из него)
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

from agent import graph as G
from agent import metrics as M
from agent.graph import build_graph, delete_thread, run_request, thread_history
from agent.threads import ThreadStore, make_title

_graph = None  # ленивая инициализация — граф требует доступный Ollama
_threads = ThreadStore()
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
    skills_used: list[str] = []
    trace: list[dict] = []        # пошаговый путь к ответу (см. graph._traced)
    title: str = ""               # название треда (придумывается моделью по первому сообщению)


class RenameRequest(BaseModel):
    user_id: str
    title: str


def _title_chat(messages: list[dict]) -> str:
    # Напрямую через LLM-клиент: G._chat пишет в текущий спан Langfuse, а
    # название генерируется уже вне трейса запроса.
    return G.llm.chat(messages, model=G._model_for("orchestrator")).content or ""


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
    meta = _threads.get(thread_id)
    if meta is None:
        meta = _threads.create(req.user_id, thread_id, make_title(_title_chat, req.message))
    else:
        _threads.touch(thread_id)
    return ChatResponse(
        thread_id=thread_id,
        answer=result["final_answer"] or "",
        route_history=result.get("route_history") or [],
        step_count=result.get("step_count", 0),
        latency_s=round(time.perf_counter() - t0, 2),
        skills_used=result.get("skills_used") or [],
        trace=result.get("trace") or [],
        title=meta["title"],
    )


@app.get("/threads")
def list_threads(user_id: str) -> list[dict]:
    return _threads.list(user_id)


@app.get("/threads/{thread_id}")
def get_thread(thread_id: str) -> dict:
    meta = _threads.get(thread_id) or {}
    return {"thread_id": thread_id, "title": meta.get("title", ""), "turns": thread_history(_graph, thread_id)}


@app.patch("/threads/{thread_id}")
def rename_thread(thread_id: str, req: RenameRequest) -> dict:
    meta = _threads.get(thread_id)
    if meta is None or meta["user_id"] != req.user_id:
        raise HTTPException(status_code=404, detail="тред не найден")
    return _threads.rename(thread_id, req.title)


@app.delete("/threads/{thread_id}")
def remove_thread(thread_id: str, user_id: str, forget_memory: bool = False) -> dict:
    meta = _threads.get(thread_id)
    if meta is None or meta["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="тред не найден")
    forgotten = delete_thread(_graph, user_id, thread_id, forget_memory)
    _threads.delete(thread_id)
    return {"deleted": thread_id, "facts_forgotten": forgotten}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok", "graph_ready": _graph is not None}
