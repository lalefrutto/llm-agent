"""
memory_manager.py — обёртка над долгосрочной памятью агента.

Основной бэкенд: Mem0 (https://github.com/mem0ai/mem0), опционально с
Qdrant как vector store. Если библиотека mem0 недоступна (например,
локальная разработка без сети) — используется LocalJSONMemoryStore,
реализующий тот же интерфейс, чтобы остальной код агента не менялся
при переключении бэкенда.

Короткосрочная память (буфер диалога) НЕ живёт здесь — она управляется
LangGraph checkpointer'ом на уровне графа (см. agent/graph.py).
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

MemoryOp = Literal["ADD", "UPDATE", "DELETE", "NOOP"]

# Ключевые слова-триггеры, при обнаружении которых факт НЕ уходит в
# долгосрочную память (см. memory/policy.md — "Что НИКОГДА не
# сохраняется"). Это грубый защитный фильтр поверх решения LLM, а не
# замена ему.
_SENSITIVE_PATTERNS = (
    "password", "пароль", "api_key", "api key", "secret", "токен",
    "card number", "номер карты", "ssn", "снилс", "паспорт",
)


@dataclass
class MemoryItem:
    id: str
    user_id: str
    text: str
    created_at: float
    metadata: dict = field(default_factory=dict)


def _role_model(role: str, default: str = "hermes3:8b-llama3.1-q4_K_M") -> str:
    """Модель роли из config/models.yaml (role_assignment); файл может
    отсутствовать в контейнере/тестах — тогда default."""
    try:
        import yaml  # type: ignore
        with open("config/models.yaml", encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("role_assignment", {}).get(role) or default
    except Exception:  # noqa: BLE001
        return default


def _contains_sensitive(text: str) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in _SENSITIVE_PATTERNS)


class LocalJSONMemoryStore:
    """Fallback-хранилище для офлайн-разработки/тестов. Реализует тот
    же контракт, что и обёртка над реальным Mem0 клиентом ниже."""

    def __init__(self, path: str = "./memory_store.json"):
        self.path = Path(path)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def _load(self) -> list[dict]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, items: list[dict]) -> None:
        self.path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, user_id: str, text: str, metadata: dict | None = None) -> MemoryItem | None:
        if _contains_sensitive(text):
            return None  # policy: секреты не сохраняем, тихо игнорируем
        item = MemoryItem(id=str(uuid.uuid4()), user_id=user_id, text=text,
                           created_at=time.time(), metadata=metadata or {})
        items = self._load()
        items.append(asdict(item))
        self._save(items)
        return item

    def search(self, user_id: str, query: str, top_k: int = 5) -> list[MemoryItem]:
        # Наивный fallback: подстрочный поиск. В реальном Mem0-бэкенде
        # здесь используется embedding similarity + recency/importance
        # weighting — см. класс Mem0Backend ниже.
        items = [MemoryItem(**i) for i in self._load() if i["user_id"] == user_id]
        query_tokens = set(query.lower().split())
        scored = []
        for it in items:
            overlap = len(query_tokens & set(it.text.lower().split()))
            recency_bonus = it.created_at / 1e10  # монотонно даёт небольшой приоритет свежим
            scored.append((overlap + recency_bonus, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for score, it in scored[:top_k] if score > 0]

    def update(self, item_id: str, new_text: str) -> bool:
        items = self._load()
        for it in items:
            if it["id"] == item_id:
                it["text"] = new_text
                it["metadata"]["updated_at"] = time.time()
                self._save(items)
                return True
        return False

    def delete(self, item_id: str) -> bool:
        items = self._load()
        new_items = [it for it in items if it["id"] != item_id]
        if len(new_items) == len(items):
            return False
        self._save(new_items)
        return True


class Mem0Backend:
    """Тонкая обёртка над реальным mem0ai клиентом. Требует:
    `pip install mem0ai` и запущенный Qdrant (см. docker-compose.yml).
    Не выполняется в текущей офлайн-песочнице — предназначен для
    запуска на машине разработчика/в Claude Code.
    """

    # nomic-embed-text отдаёт 768-мерные векторы через нативный Ollama API
    # (mem0 не обрезает их под Matryoshka) — Qdrant-коллекция и embedder
    # должны быть согласованы на это же число, иначе insert падает.
    EMBEDDING_DIMS = 768

    def __init__(self, qdrant_url: str = os.environ.get("QDRANT_URL", "http://localhost:6333")):
        try:
            from mem0 import Memory  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "mem0ai не установлен. Установите: pip install mem0ai, "
                "или используйте LocalJSONMemoryStore для офлайн-режима."
            ) from e

        # agent/llm_client.py использует OpenAI-совместимый путь
        # (.../v1) для OLLAMA_BASE_URL, а mem0's Ollama-провайдер (LLM и
        # embedder) ходит напрямую через нативный клиент `ollama`
        # (pip install ollama), которому нужен base_url БЕЗ суффикса /v1.
        ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        if ollama_base_url.endswith("/v1"):
            ollama_base_url = ollama_base_url[: -len("/v1")]

        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": qdrant_url.split("://")[-1].split(":")[0],
                    "port": 6333,
                    "embedding_model_dims": self.EMBEDDING_DIMS,
                },
            },
            # LLM для извлечения фактов и решений ADD/UPDATE/DELETE —
            # указываем на тот же локальный Ollama-эндпоинт, что и
            # основной агент (см. config/models.yaml).
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": os.environ.get("MEM0_LLM_MODEL") or _role_model("memory_extraction"),
                    "ollama_base_url": ollama_base_url,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": "nomic-embed-text",
                    "ollama_base_url": ollama_base_url,
                    "embedding_dims": self.EMBEDDING_DIMS,
                },
            },
        }
        self._client = Memory.from_config(config)

    def add(self, user_id: str, text: str, metadata: dict | None = None):
        if _contains_sensitive(text):
            return None
        return self._client.add(text, user_id=user_id, metadata=metadata or {})

    def search(self, user_id: str, query: str, top_k: int = 5) -> list[MemoryItem]:
        # mem0>=2.x убрал позиционный user_id/limit из search() в пользу
        # filters={"user_id": ...} и top_k=, и возвращает сырой
        # {"results": [{"id":..., "memory": "...", ...}]}, а не список
        # объектов с полем .text (контракт этого класса, см.
        # LocalJSONMemoryStore) — нормализуем форму ответа.
        raw = self._client.search(query, filters={"user_id": user_id}, top_k=top_k)
        items = raw.get("results", []) if isinstance(raw, dict) else (raw or [])
        return [
            MemoryItem(
                id=it.get("id", ""),
                user_id=user_id,
                text=it.get("memory", ""),
                created_at=0.0,
                metadata=it.get("metadata") or {},
            )
            for it in items
        ]

    def update(self, item_id: str, new_text: str) -> bool:
        self._client.update(item_id, new_text)
        return True

    def delete(self, item_id: str) -> bool:
        self._client.delete(item_id)
        return True


def get_memory_backend():
    """Фабрика: реальный Mem0, если библиотека и Qdrant доступны,
    иначе — локальный JSON fallback (для разработки без сети).
    MEMORY_BACKEND=local принудительно включает fallback (evals без Qdrant),
    MEMORY_BACKEND=mem0 — запрещает fallback (падать громко, а не тихо)."""
    import logging
    log = logging.getLogger("atlas.memory")
    mode = os.environ.get("MEMORY_BACKEND", "auto")
    if mode == "local":
        log.warning("memory backend: LocalJSONMemoryStore (MEMORY_BACKEND=local)")
        return LocalJSONMemoryStore()
    try:
        backend = Mem0Backend()
        log.info("memory backend: Mem0Backend (Qdrant %s)", os.environ.get("QDRANT_URL", "http://localhost:6333"))
        return backend
    except Exception as e:  # noqa: BLE001
        if mode == "mem0":
            raise
        log.warning("memory backend: fallback LocalJSONMemoryStore — Mem0 недоступен: %s: %s", type(e).__name__, e)
        return LocalJSONMemoryStore()


if __name__ == "__main__":
    # Небольшой smoke-test, работает офлайн через LocalJSONMemoryStore.
    store = LocalJSONMemoryStore(path="/tmp/agent_memory_demo.json")
    store.add("user_1", "Пользователь предпочитает ответы на русском языке.")
    store.add("user_1", "Пароль от базы: hunter2")  # должен быть отфильтрован
    results = store.search("user_1", "на каком языке отвечать")
    print("Найдено записей:", len(results))
    for r in results:
        print("-", r.text)
