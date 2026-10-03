"""
llm_client.py — тонкая обёртка над локальным движком инференса.

Почему Ollama (обоснование движка, кратко — полная версия в
docs/ТЗ_и_отчёт.docx, раздел "Выбор движка"):

- Ollama оборачивает llama.cpp и даёт: (1) OpenAI-совместимый API
  из коробки — не нужно писать свой сервер вокруг llama.cpp;
  (2) простое переключение между GGUF-моделями одной командой —
  критично, когда по ТЗ нужно протестировать 3+ модели;
  (3) разумный дефолтный оффлоад на GPU/CPU без ручной настройки
  тензорного параллелизма, что достаточно для одного пользователя
  на 8GB VRAM.
- vLLM даёт лучший throughput при батчинге многих одновременных
  запросов (continuous batching, paged attention) — это плюс для
  продакшн-serving многих пользователей, но не даёт ощутимого
  выигрыша для одного агента с одним активным запросом за раз, а
  настройка (особенно на Windows/WSL) заметно тяжелее.
- Прямой llama.cpp даёт максимальный контроль и минимальный
  оверхед, но требует ручной работы, которую Ollama уже сделал
  (менеджмент моделей, HTTP API, автоматический offload).

Итог: Ollama — для разработки и защиты проекта; при переходе к
реальному продакшену с несколькими одновременными пользователями
рекомендован пересмотр в пользу vLLM (см. docs, раздел "Ограничения
и дальнейшее развитие").
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from openai import OpenAI  # ollama поддерживает OpenAI-совместимый клиент
except ImportError:
    OpenAI = None  # позволяет импортировать модуль для тестов без установленного пакета


@dataclass
class LLMConfig:
    base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    api_key: str = "ollama"  # Ollama не проверяет ключ, но клиенту OpenAI он нужен формально
    model: str = os.environ.get("AGENT_DEFAULT_MODEL", "atlas-hermes3-12k")  # см. config/models.yaml
    temperature: float = 0.2
    max_tokens: int = 1024
    extra_headers: dict = field(default_factory=dict)


class LLMClient:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        if OpenAI is None:
            raise RuntimeError("Пакет 'openai' не установлен: pip install openai")
        self._client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key)
        # usage последнего вызова (prompt/completion tokens) — нужен для
        # замера токенов/сек в evals/compare_models.py и для метрик
        # Prometheus (agent_tokens_total) в agent/main.py.
        self.last_usage = None

    def chat(self, messages: list[dict], model: str | None = None, tools: list[dict] | None = None,
             json_mode: bool = False):
        # json_mode включает grammar-constrained JSON-вывод Ollama
        # (OpenAI-совместимый response_format=json_object) — без него
        # модели нередко просто игнорируют инструкцию вернуть JSON и
        # отвечают обычным текстом (см. route()/run_critic() в graph.py,
        # где нужен строго распарсиваемый ответ).
        response = self._client.chat.completions.create(
            model=model or self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            tools=tools or None,
            response_format={"type": "json_object"} if json_mode else None,
        )
        self.last_usage = response.usage
        return response.choices[0].message


if __name__ == "__main__":
    # Требует запущенный `ollama serve` и `ollama pull hermes3:8b-llama3.1-q4_K_M`
    # на машине разработчика — в текущей офлайн-песочнице сети нет.
    client = LLMClient()
    msg = client.chat([{"role": "user", "content": "Скажи 'ок' одним словом."}])
    print(msg.content)
