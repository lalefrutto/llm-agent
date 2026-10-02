"""
file_ops.py — файловые операции для под-агента Executor.

Все пути принудительно резолвятся относительно WORKSPACE_ROOT и
проверяются на выход за его пределы (защита от path traversal). Это
соответствует skill `code-review` и guardrails в
`prompts/subagent_executor.md`.
"""
from __future__ import annotations

import os
from pathlib import Path

WORKSPACE_ROOT = Path(os.environ.get("AGENT_WORKSPACE", "/workspace")).resolve()

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Прочитать файл внутри рабочей директории песочницы.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": "Записать файл внутри рабочей директории песочницы.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
]


class PathEscapeError(Exception):
    pass


def _safe_resolve(path: str) -> Path:
    candidate = (WORKSPACE_ROOT / path).resolve()
    if WORKSPACE_ROOT not in candidate.parents and candidate != WORKSPACE_ROOT:
        raise PathEscapeError(
            f"Путь '{path}' резолвится за пределы WORKSPACE_ROOT ({WORKSPACE_ROOT}). "
            "Операция заблокирована политикой изоляции."
        )
    return candidate


def file_read(path: str) -> str:
    target = _safe_resolve(path)
    if not target.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    return target.read_text(encoding="utf-8")


def exists(path: str) -> bool:
    """Есть ли файл (для запрета перезаписи без подтверждения, см. graph._dispatch_tool)."""
    return _safe_resolve(path).exists()


def file_write(path: str, content: str) -> bool:
    target = _safe_resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return True


if __name__ == "__main__":
    # smoke test с локальным временным workspace
    os.environ["AGENT_WORKSPACE"] = "/tmp/agent_workspace_demo"
    Path("/tmp/agent_workspace_demo").mkdir(exist_ok=True)
    import importlib
    import agent.tools.file_ops as fo  # noqa: E402
    importlib.reload(fo)
    fo.file_write("test.txt", "hello")
    print(fo.file_read("test.txt"))
    try:
        fo.file_read("../../etc/passwd")
    except PathEscapeError as e:
        print("Заблокировано корректно:", e)
