"""
threads.py — список тредов пользователя для веб-чата (как в обычном чат-боте):
название, время создания и последней активности. Сами сообщения живут в
checkpointer'е LangGraph (см. graph.thread_history), здесь — только реестр.

Хранилище — отдельный SQLite-файл рядом с checkpoints.sqlite: реестр нужен
до того, как граф что-то запишет, и его удобно читать одним SELECT'ом.
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading
import time

TITLE_MAX = 60
# Без примеров 8B-модель склеивала слова («МурзумКот») или отвечала целой фразой.
_TITLE_PROMPT = ("Придумай короткое название (2–5 слов через пробел) для диалога, который начинается "
                 "с этого сообщения пользователя. Ответь только названием: с заглавной буквы, без кавычек, "
                 "без точки в конце, на языке сообщения.\n"
                 "Примеры:\n"
                 "«Посчитай 18% от 4500» -> Расчёт процента от числа\n"
                 "«Сравни Ollama, llama.cpp и vLLM» -> Сравнение движков инференса\n"
                 "«У меня завтра экзамен по физике» -> Экзамен по физике")


def clean_title(raw: str | None, fallback: str) -> str:
    """Ответ модели -> аккуратное название; при мусоре — начало сообщения."""
    t = (raw or "").strip().splitlines()[0] if (raw or "").strip() else ""
    t = re.sub(r"^(название|title)\s*[:：]\s*", "", t, flags=re.IGNORECASE)
    t = t.strip(" \"'«»“”`*#.").strip()
    if not t or len(t) > TITLE_MAX:
        t = fallback.strip()
        t = t if len(t) <= 40 else t[:40].rstrip() + "…"
    return t or "Новый тред"


def make_title(chat_fn, first_message: str) -> str:
    """chat_fn(messages) -> str. Ошибка модели не должна ронять ответ в чат."""
    try:
        raw = chat_fn([{"role": "system", "content": _TITLE_PROMPT},
                       {"role": "user", "content": first_message[:1000]}])
    except Exception:  # noqa: BLE001
        raw = None
    return clean_title(raw, first_message)


class ThreadStore:
    def __init__(self, path: str | None = None):
        path = path or os.environ.get("THREADS_DB", "./threads.sqlite")
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS threads (thread_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, "
                "title TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL)")
            self._conn.commit()

    def get(self, thread_id: str) -> dict | None:
        row = self._conn.execute("SELECT thread_id, user_id, title, created_at, updated_at FROM threads "
                                 "WHERE thread_id = ?", (thread_id,)).fetchone()
        return dict(zip(("thread_id", "user_id", "title", "created_at", "updated_at"), row)) if row else None

    def create(self, user_id: str, thread_id: str, title: str) -> dict:
        now = time.time()
        with self._lock:
            self._conn.execute("INSERT OR IGNORE INTO threads VALUES (?, ?, ?, ?, ?)",
                               (thread_id, user_id, title[:TITLE_MAX], now, now))
            self._conn.commit()
        return self.get(thread_id)

    def touch(self, thread_id: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE threads SET updated_at = ? WHERE thread_id = ?", (time.time(), thread_id))
            self._conn.commit()

    def rename(self, thread_id: str, title: str) -> dict | None:
        title = title.strip()[:TITLE_MAX]
        if not title:
            return self.get(thread_id)
        with self._lock:
            self._conn.execute("UPDATE threads SET title = ? WHERE thread_id = ?", (title, thread_id))
            self._conn.commit()
        return self.get(thread_id)

    def delete(self, thread_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
            self._conn.commit()

    def list(self, user_id: str) -> list[dict]:
        rows = self._conn.execute("SELECT thread_id, user_id, title, created_at, updated_at FROM threads "
                                  "WHERE user_id = ? ORDER BY updated_at DESC", (user_id,)).fetchall()
        return [dict(zip(("thread_id", "user_id", "title", "created_at", "updated_at"), r)) for r in rows]
