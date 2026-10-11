"""
migrate_embeddings.py — перенос фактов Mem0 в коллекцию с новой моделью эмбеддингов.

При смене эмбеддера (nomic-embed-text 768 dims -> bge-m3 1024 dims) старые
векторы несовместимы с новой коллекцией. Скрипт берёт текст и payload каждого
факта из старой коллекции, считает новый эмбеддинг и записывает факт с тем же
id и payload (дата, user_id, thread_id сохраняются) в новую коллекцию.
Повторный запуск безопасен (upsert по id).

    python memory/migrate_embeddings.py --users demo_user
    python memory/migrate_embeddings.py              # все пользователи
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def embed(text: str, model: str, ollama_url: str) -> list[float]:
    req = urllib.request.Request(ollama_url.rstrip("/") + "/api/embed",
                                 data=json.dumps({"model": model, "input": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["embeddings"][0]


def main():
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchAny, PointStruct

    from memory.memory_manager import Mem0Backend

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="mem0", help="старая коллекция")
    ap.add_argument("--users", default="", help="через запятую; по умолчанию — все")
    args = ap.parse_args()

    Mem0Backend()  # создаёт целевую коллекцию с правильной размерностью, если её нет
    target, model = Mem0Backend.COLLECTION, Mem0Backend.EMBED_MODEL
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1").removesuffix("/v1")
    qdrant = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))

    users = [u.strip() for u in args.users.split(",") if u.strip()]
    flt = Filter(must=[FieldCondition(key="user_id", match=MatchAny(any=users))]) if users else None
    moved, offset = 0, None
    while True:
        points, offset = qdrant.scroll(args.source, scroll_filter=flt, limit=100, offset=offset, with_payload=True)
        batch = [PointStruct(id=p.id, vector=embed(p.payload.get("data", ""), model, ollama_url), payload=p.payload)
                 for p in points if p.payload.get("data")]
        if batch:
            qdrant.upsert(target, points=batch)
            moved += len(batch)
        if offset is None:
            break
    print(f"перенесено {moved} фактов: {args.source} -> {target} (эмбеддер {model})")


if __name__ == "__main__":
    main()
