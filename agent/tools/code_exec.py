"""
code_exec.py — выполнение кода в изолированном sandbox-контейнере.

Важно: этот модуль НЕ выполняет код в процессе агента. Он отправляет
код в отдельный, заранее поднятый эфемерный Docker-контейнер
(`exec-sandbox`, см. docker/docker-compose.yml) через docker exec /
короткоживущий `docker run --rm`. Так соблюдается принцип
"минимальные полномочия" из identity/VALUES.md — даже если сам агент
скомпрометирован, у него нет прямого доступа к shell хоста.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

from agent.guardrails import destructive_pattern

SANDBOX_IMAGE = "agent-exec-sandbox:latest"
DEFAULT_TIMEOUT_S = 30

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "code_exec",
        "description": (
            "Выполнить код Python в изолированном контейнере без сети. "
            "Использовать для вычислений, обработки данных, проверки "
            "гипотез кодом — не для операций с сетью или системой хоста."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "timeout_s": {"type": "integer", "default": DEFAULT_TIMEOUT_S},
            },
            "required": ["code"],
        },
    },
}


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


# Грубый предварительный фильтр (defence in depth поверх docker
# --network=none) — соответствует skill `code-review`.
_BLOCKED_SUBSTRINGS = (
    "import socket", "import requests", "urllib.request", "os.environ",
    "subprocess.Popen", "/etc/passwd", "~/.ssh",
)


# REPL-семантика: если последняя инструкция — выражение без print(), его
# значение печатается (как в Jupyter/ipython). Иначе модели регулярно
# шлют `a + b` без print, получают пустой stdout и выдумывают результат
# (наблюдалось вживую: "Результат: 338350.0" при пустом stdout).
_REPL_WRAPPER = """import ast, sys
src = sys.argv[1]; tree = ast.parse(src); g = {"__name__": "__main__"}
last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
exec(compile(tree, "<code>", "exec"), g)
if last is not None:
    v = eval(compile(ast.Expression(last.value), "<code>", "eval"), g)
    if v is not None: print(repr(v))
"""


def _static_check(code: str) -> str | None:
    for pattern in _BLOCKED_SUBSTRINGS:
        if pattern in code:
            return f"Код заблокирован статической проверкой: обнаружен паттерн '{pattern}'."
    return None


def code_exec(code: str, timeout_s: int = DEFAULT_TIMEOUT_S, workspace: str | None = None,
              allow_destructive: bool = False) -> ExecResult:
    block_reason = _static_check(code)
    if block_reason:
        return ExecResult(exit_code=-1, stdout="", stderr=block_reason)
    # Разрушающие операции (os.remove, shutil.rmtree, ...) — только после явного
    # подтверждения пользователя (см. agent/guardrails.py).
    destructive = None if allow_destructive else destructive_pattern(code)
    if destructive:
        return ExecResult(exit_code=-1, stdout="", stderr=(
            f"Код заблокирован: разрушающая операция '{destructive}' без подтверждения пользователя."))

    # Внутри контейнера agent-app хостовый путь /workspace не имеет смысла для
    # docker-демона хоста — там монтируем тот же named volume, что и у
    # agent-app (SANDBOX_WORKSPACE_VOLUME, см. docker-compose.yml).
    workspace = (workspace or os.environ.get("SANDBOX_WORKSPACE_VOLUME")
                 or os.environ.get("AGENT_WORKSPACE", "/tmp/agent_workspace"))
    timeout_s = min(int(timeout_s or DEFAULT_TIMEOUT_S), DEFAULT_TIMEOUT_S)  # правило 2 из subagent_executor.md

    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--memory", "512m",
        "--cpus", "1",
        "--read-only",
        "-v", f"{workspace}:/workspace",
        "-w", "/workspace",
        SANDBOX_IMAGE,
        "python3", "-c", _REPL_WRAPPER, code,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s + 5,  # +5с на старт контейнера
        )
        return ExecResult(exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    except subprocess.TimeoutExpired:
        return ExecResult(exit_code=-1, stdout="", stderr=f"Превышен таймаут {timeout_s}с", timed_out=True)
    except FileNotFoundError:
        return ExecResult(
            exit_code=-1, stdout="",
            stderr="Docker недоступен в этом окружении. Запустите на машине с Docker (см. README.md).",
        )


if __name__ == "__main__":
    print(code_exec("print(1 + 1)"))
    print(code_exec("import socket; socket.socket()"))  # должно быть заблокировано
