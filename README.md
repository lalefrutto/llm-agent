# Atlas — запуск

Требования: Windows 10/11, NVIDIA GPU 8 GB+ VRAM, Docker Desktop, Python 3.11, Ollama.

## 1. Ollama и модели

```powershell
winget install Ollama.Ollama
ollama pull hermes3:8b-llama3.1-q4_K_M
ollama pull nomic-embed-text
```

Для сравнения моделей (`evals/compare_models.py`) дополнительно:

```powershell
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
```

## 2. Python-окружение

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. Инфраструктура

Docker Desktop должен быть запущен.

```powershell
docker build -t agent-exec-sandbox:latest -f docker/Dockerfile.sandbox .
docker compose -f docker/docker-compose.yml up -d --build
docker compose -f docker/docker-compose.yml ps
```

Дождаться `healthy` у `agent-app`, `qdrant`, `langfuse`, `langfuse-db`, `prometheus`, `grafana`, `alertmanager`, `llm-engine` (`loki` и `otel-collector` — без healthcheck).

## 4. Использование

| Сервис | Адрес | Доступ |
|---|---|---|
| Веб-чат / API | http://localhost:8080 | — |
| Langfuse (трейсы) | http://localhost:3000 | atlas@example.com / atlas-password |
| Grafana (дашборд *Atlas → Atlas Agent*) | http://localhost:3001 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| Qdrant | http://localhost:6333 | — |

API:

```powershell
curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" `
  -d '{"user_id":"u1","message":"Посчитай 18% от 4500"}'
```

## 5. Evals

```powershell
# сравнение моделей -> evals/results/model_comparison.md
python evals/compare_models.py

# оценка агентной системы (нужны запущенные Ollama и docker compose)
$env:MEMORY_BACKEND="mem0"
$env:SANDBOX_WORKSPACE_VOLUME="docker_agent_workspace"
python evals/run_evals.py --dataset evals/dataset.jsonl --live
```

## Переменные окружения

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | адрес Ollama |
| `AGENT_DEFAULT_MODEL` | `hermes3:8b-llama3.1-q4_K_M` | модель по умолчанию; роли — `config/models.yaml` |
| `AGENT_MODEL_OVERRIDE` | — | одна модель на все роли |
| `MEMORY_BACKEND` | `auto` | `mem0` / `local` |
| `QDRANT_URL` | `http://localhost:6333` | |
| `SEARCH_PROVIDER` | `ddgs` | `tavily` (+`TAVILY_API_KEY`) / `none` |
| `SANDBOX_WORKSPACE_VOLUME` | — | named volume для sandbox |
| `LANGFUSE_HOST` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | — | трейсинг (без ключей выключен) |

## Остановка

```powershell
docker compose -f docker/docker-compose.yml down
```
