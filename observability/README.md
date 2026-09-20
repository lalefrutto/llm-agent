# Observability — что и как мониторим

## Стек (гибрид, а не выбор "или-или")

- **Langfuse** — LLM-нативный слой: трейсы каждого inference-вызова
  (промпт, completion, токены, latency, стоимость), группировка по
  сессии/треду, встроенные evals/скоринг ответов.
- **OpenTelemetry Collector → Prometheus (метрики) + Loki (логи)
  → Grafana (дашборды) + Alertmanager (алерты)** — стандартный,
  LLM-агностичный стек. Нужен, потому что Langfuse не покажет
  системные метрики контейнеров (CPU/RAM sandbox, доступность
  Qdrant/Ollama) и не заменяет alerting-инфраструктуру, которая уже
  может быть принята в компании.

Почему оба, а не один: Langfuse отвечает на вопрос "что именно
ответила модель и почему" (для отладки промптов и evals), стандартный
стек — на вопрос "жива ли система и укладывается ли в SLA" (для
дежурного инженера). Для мультиагентной системы это разные аудитории
и разные вопросы, закрывать одним инструментом обе роли неудобно.

## Что мониторим (метрики)

| Метрика | Зачем | Где хранится |
|---|---|---|
| Latency per step / p50/p95/p99 | SLA, деградация | Prometheus |
| Token usage & cost per request | Контроль расходов | Langfuse + Prometheus |
| `agent_requests_total{status}` | Error rate | Prometheus |
| `agent_routing_invalid_json_total` | Деградация модели/промпта роутинга | Prometheus |
| `agent_max_steps_reached_total` | Циклы делегирования, зависшие задачи | Prometheus |
| `agent_sandbox_blocked_total` | Попытки выйти за пределы песочницы | Prometheus (+ алерт) |
| Critic verdict distribution (approve/revise/reject) | Качество под-агентов | Langfuse |
| Memory ADD/UPDATE/DELETE rate | Здоровье памяти, аномальный рост | Langfuse/логи |
| Eval-метрики в проде (сэмплированные) | Дрейф качества модели со временем | Langfuse evals |

## Специфика мониторинга именно агента (не обычного backend-сервиса)

1. **Нужно трейсить не запрос, а цепочку решений** — один
   пользовательский запрос может пройти через 3-5 узлов графа
   (router → researcher → critic → finalize); плоский APM-трейс
   "запрос-ответ" здесь бесполезен, нужен span-per-node (это и даёт
   связка LangGraph + OTel + Langfuse).
2. **Нужно мониторить "мягкие" сбои**, которых нет в HTTP-статусах:
   невалидный JSON от модели, зацикливание делегирования, растущая
   доля "revise/reject" от критика — обычный uptime-мониторинг их не
   увидит.
3. **Нужен sampling-based eval в проде**, а не только на этапе
   разработки — модель/промпты дрейфуют, эталонный датасет из
   `evals/` не покрывает продакшн-трафик целиком.
4. **Security-специфичные события** — попытки выхода из sandbox
   критичнее обычной 500-ошибки и должны алертить немедленно
   (`SandboxEscapeAttempt`, `for: 0m` — без задержки на "for").

## Куда прилетают алерты

Alertmanager настроен с заглушкой webhook (`observability/prometheus/alertmanager.yml`)
— на реальном развёртывании подставляется Slack/Telegram/PagerDuty.
Критичные алерты (`severity: critical`) — sandbox escape, error rate —
не должны иметь `repeat_interval` больше 15 минут; предупреждающие
(`warning`) агрегируются раз в несколько часов, чтобы не создавать
alert fatigue.
