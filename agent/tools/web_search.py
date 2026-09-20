"""
web_search.py — инструмент веб-поиска для под-агента Researcher.

Провайдер выбирается через SEARCH_PROVIDER:
  - "ddgs"   (по умолчанию) — DuckDuckGo через пакет `ddgs`, без API-ключа;
  - "tavily" — Tavily, нужен TAVILY_API_KEY;
  - "none"   — поиск отключён (RuntimeError, researcher честно сообщит об этом).
Интерфейс инструмента для LangGraph/LLM tool-calling не зависит от провайдера.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Найти актуальную информацию в интернете. Использовать для "
            "фактов, которые могут устареть в параметрических знаниях "
            "модели (текущие события, версии ПО, цены, статусы)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Поисковый запрос, 3-8 слов"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
}


def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    provider = os.environ.get("SEARCH_PROVIDER", "ddgs")

    if provider == "ddgs":
        from ddgs import DDGS  # type: ignore

        raw = DDGS().text(query, max_results=max_results)
        return [
            SearchResult(title=r.get("title", ""), url=r.get("href", ""), snippet=r.get("body", ""))
            for r in raw or []
        ]

    if provider == "tavily":
        from tavily import TavilyClient  # type: ignore

        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        raw = client.search(query=query, max_results=max_results)
        return [
            SearchResult(title=r.get("title", ""), url=r.get("url", ""), snippet=r.get("content", ""))
            for r in raw.get("results", [])
        ]

    raise RuntimeError(
        f"SEARCH_PROVIDER='{provider}' не настроен. Установите SEARCH_PROVIDER=ddgs "
        "(без ключа) или SEARCH_PROVIDER=tavily + TAVILY_API_KEY."
    )
