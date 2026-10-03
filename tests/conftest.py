"""Детерминированные тесты (без Ollama/Qdrant/Docker): pytest tests/

Окружение выставляется ДО импорта agent.graph: модуль на импорте создаёт
LLM-клиент (сеть не трогает) и бэкенд памяти — здесь это локальный JSON.
Промпты читаются по относительным путям, поэтому cwd = корень репозитория.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="atlas_tests_"))

os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ["MEMORY_BACKEND"] = "local"
os.environ["CHECKPOINT_DB"] = str(TMP / "checkpoints.sqlite")
os.environ["AGENT_WORKSPACE"] = str(TMP / "workspace")
(TMP / "workspace").mkdir()
os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
