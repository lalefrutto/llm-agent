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
import re
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


# Правила извлечения фактов для LLM внутри Mem0 (custom_instructions).
# Без них экстрактор сам отвечал на вопрос пользователя и сохранял ответ как
# «факт»: из «На основе только этого текста: 'Кошки — млекопитающие' —
# сколько лап у кошки?» появлялся факт «A cat has 4 paws based on the text»,
# который в следующем треде агент выдавал за содержание текста (кейс t10);
# из «посчитай среднее» — неверное «average was 130.5» (на деле 132).
#
# Итерации (каждая проверялась прогоном одних и тех же фраз по 3 раза):
#  1) правила-запреты -> 8B-экстрактор отбрасывал всё, включая «Я живу в
#     Москве» (t06 упал до 1/5);
#  2) позитивные правила с примерами на реальных именах -> факты «утекали»
#     из примеров («User's name is Ильназ» у пользователя, который этого не
#     говорил), а на анализ продаж экстрактор выдумал месяцы и среднее.
# Итог: примеры — с обезличенными заглушками, а вопросы и поручения до
# экстрактора не доходят вообще (is_memory_worthy, детерминированно).
MEMORY_EXTRACTION_RULES = """\
Extract facts the user tells about THEMSELVES: name, city, study, work, projects, \
plans, deadlines, preferences. A statement like "I live in <CITY>" or "I moved to \
<CITY>" IS such a fact — always extract it. Never answer questions, never compute, \
never add general knowledge. The examples below use placeholders — never copy \
anything from them into memories, and never output words in angle brackets: \
mention only what the user actually said.

Examples:
"Я живу в <CITY>" -> ["User lives in <CITY>"]
"Я переехал в <CITY>" -> ["User moved to <CITY>"]
"Меня зовут <NAME>, я работаю <JOB>" -> ["User's name is <NAME>", "User works as <JOB>"]
"Запомни: <PREFERENCE>" -> ["User prefers: <PREFERENCE>"]"""

# Фразы-поручения: их смысл — задача для агента, а не факт о пользователе.
_TASK_VERBS = re.compile(
    r"^\s*(посчитай|вычисли|рассчитай|проанализируй|найди|сравни|выполни|сделай|напиши|"
    r"покажи|объясни|переведи|удали|прочитай|сохрани|составь|проверь|построй|сгенерируй|"
    r"расскажи|ответь|подскажи|помоги)", re.IGNORECASE)
_FIRST_PERSON = re.compile(r"(?<!\w)(я|меня|мне|мной|мой|моя|моё|мое|мои|моих|у меня|мы|нас|наш|наша)(?!\w)",
                           re.IGNORECASE)


def is_memory_worthy(text: str) -> bool:
    """Стоит ли отдавать реплику экстрактору Mem0. Вопросы и поручения — нет
    (иначе экстрактор на них отвечает и сохраняет ответ как факт); явная
    просьба «запомни» — да; иначе нужна речь от первого лица."""
    t = text.strip()
    if re.match(r"^\s*запомни", t, re.IGNORECASE):
        return True
    if "?" in t or _TASK_VERBS.match(t):
        return False
    return bool(_FIRST_PERSON.search(t))


@dataclass
class MemoryItem:
    id: str
    user_id: str
    text: str
    created_at: float
    metadata: dict = field(default_factory=dict)


def _role_model(role: str, default: str = "atlas-hermes3-12k") -> str:
    """Модель роли из config/models.yaml (role_assignment); файл может
    отсутствовать в контейнере/тестах — тогда default."""
    try:
        import yaml  # type: ignore
        with open("config/models.yaml", encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("role_assignment", {}).get(role) or default
    except Exception:  # noqa: BLE001
        return default


def _ts(iso: str | None) -> float:
    """ISO-время из payload Mem0 -> unix timestamp (0.0, если нет)."""
    from datetime import datetime
    try:
        return datetime.fromisoformat(iso).timestamp() if iso else 0.0
    except ValueError:
        return 0.0


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

    def __init__(self, qdrant_url: str | None = None):
        try:
            from mem0 import Memory  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "mem0ai не установлен. Установите: pip install mem0ai, "
                "или используйте LocalJSONMemoryStore для офлайн-режима."
            ) from e

        # ПРИМЕЧАНИЕ: os.environ.get() как значение по умолчанию в сигнатуре
        # функции вычисляется ОДИН РАЗ при импорте модуля, а не при каждом
        # вызове — на практике в docker-compose это не страшно (переменные
        # окружения контейнера выставлены ДО старта Python), но это хрупко
        # для тестов/повторного использования класса. Поэтому резолвим
        # внутри __init__, а не в сигнатуре.
        if qdrant_url is None:
            qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")

        # agent/llm_client.py использует OpenAI-совместимый путь (.../v1)
        # для OLLAMA_BASE_URL, а mem0's Ollama-провайдер (LLM и embedder)
        # ходит напрямую через нативный клиент `ollama` (pip install ollama),
        # которому нужен base_url БЕЗ суффикса /v1. Без этого клиент внутри
        # mem0 использует свой дефолт http://localhost:11434 — а localhost
        # внутри контейнера agent-app — это сам agent-app. В docker-compose
        # OLLAMA_BASE_URL задан явно; дефолт совпадает с agent/llm_client.py
        # (запуск evals с хоста).
        ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        if ollama_base_url.endswith("/v1"):
            ollama_base_url = ollama_base_url[: -len("/v1")]

        config = {
            "custom_instructions": MEMORY_EXTRACTION_RULES,
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": qdrant_url.split("://")[-1].split(":")[0],
                    "port": 6333,
                    "embedding_model_dims": self.EMBEDDING_DIMS,
                },
            },
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
                created_at=_ts(it.get("created_at")),
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