"""
metrics.py — Prometheus-метрики агента (см. observability/README.md,
таблица "Что мониторим"). Экспортируются через GET /metrics в
agent/main.py, Prometheus скрейпит их по job_name "agent-app".

Метрики намеренно "агентные", а не HTTP-шные: невалидный JSON роутера,
достижение MAX_STEPS, блокировки sandbox — это "мягкие" сбои, которых
не видно в кодах ответа.
"""
from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUESTS_TOTAL = Counter(
    "agent_requests_total", "Запросы к /chat по статусу", ["status"])
ROUTING_INVALID_JSON = Counter(
    "agent_routing_invalid_json_total", "Роутер вернул нераспарсиваемый JSON (fallback answer_directly)")
MAX_STEPS_REACHED = Counter(
    "agent_max_steps_reached_total", "Достигнут лимит MAX_STEPS делегирования")
SANDBOX_BLOCKED = Counter(
    "agent_sandbox_blocked_total", "Заблокированные вызовы sandbox/file_ops", ["reason"])
ROUTE_DECISIONS = Counter(
    "agent_route_decisions_total", "Решения роутера", ["action"])
CRITIC_VERDICTS = Counter(
    "agent_critic_verdict_total", "Вердикты критика", ["verdict"])
TOOL_CALLS = Counter(
    "agent_tool_calls_total", "Вызовы инструментов под-агентами", ["tool", "status"])
LLM_LATENCY = Histogram(
    "agent_llm_latency_seconds", "Латентность одного вызова LLM по узлу графа", ["node"],
    buckets=(0.5, 1, 2, 4, 8, 16, 32, 64))
GRAPH_LATENCY = Histogram(
    "agent_graph_latency_seconds", "Латентность полного прохода графа на один запрос",
    buckets=(1, 2, 5, 10, 20, 40, 80, 160))
TOKENS_TOTAL = Counter(
    "agent_tokens_total", "Токены по типу (prompt/completion)", ["type"])
GUARDRAIL_TRIGGERED = Counter(
    "agent_guardrail_triggered_total", "Необратимое действие остановлено до LLM-роутера (ask_user)", ["reason"])
SKILLS_LOADED = Counter(
    "agent_skills_loaded_total", "Какие SKILL.md подмешаны в промпт узла", ["skill", "agent"])
