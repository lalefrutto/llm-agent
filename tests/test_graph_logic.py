"""Логика графа с подменённой LLM: проверяем то, что должно быть гарантировано
кодом, а не поведением модели (guardrails, остановка при блокировке, память,
skills). Скриптованная LLM отвечает по имени узла графа."""
import json
import uuid
from types import SimpleNamespace

import pytest

import agent.graph as g


class FakeToolCall:
    def __init__(self, name, args):
        self.id = "call_" + uuid.uuid4().hex[:6]
        self.function = SimpleNamespace(name=name, arguments=json.dumps(args))

    def model_dump(self):
        return {"id": self.id, "type": "function",
                "function": {"name": self.function.name, "arguments": self.function.arguments}}


def reply(content="", tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


class FakeMemory:
    def __init__(self):
        self.added = []

    def search(self, user_id, query, top_k=5):
        return []

    def add(self, user_id, text, metadata=None):
        self.added.append(text)


@pytest.fixture
def fake(monkeypatch):
    """calls — журнал (узел, системный промпт); script[node] — очередь ответов."""
    state = SimpleNamespace(calls=[], script={}, memory=FakeMemory())

    def _chat(node, messages, **kw):
        state.calls.append((node, messages[0]["content"], messages))
        queue = state.script.get(node) or []
        return queue.pop(0) if queue else reply("ок")

    monkeypatch.setattr(g, "_chat", _chat)
    monkeypatch.setattr(g, "memory", state.memory)
    state.app = g.build_graph()
    return state


def run(fake, msg, thread="t-" + uuid.uuid4().hex[:6], user="u1"):
    return g.run_request(fake.app, user, thread, msg), thread


def nodes(fake):
    return [n for n, _, _ in fake.calls]


def test_irreversible_request_never_reaches_llm_router_or_executor(fake):
    res, _ = run(fake, "Удали все файлы в моей рабочей директории")
    assert res["route_history"] == ["ask_user"]
    assert res["pending_action"] == "Удали все файлы в моей рабочей директории"
    assert "router" not in nodes(fake) and "executor" not in nodes(fake)
    finalize_msgs = [m for n, _, m in fake.calls if n == "finalize"][0]
    assert any("да, подтверждаю" in m["content"] for m in finalize_msgs if m["role"] == "system")


def test_confirmation_unlocks_destructive_code(fake):
    _, thread = run(fake, "Удали все файлы в папке tmp")
    fake.calls.clear()
    fake.script["router"] = [reply('{"action": "delegate_executor", "reasoning": "подтверждено", '
                                   '"subtask": "удалить файлы в tmp"}')]
    fake.script["executor"] = [reply(tool_calls=[FakeToolCall("code_exec", {"code": "import os; os.remove('tmp/a')"})]),
                               reply("Удалил tmp/a")]
    executed = []
    g.code_exec_tool.code_exec, orig = (lambda code, timeout_s=30, allow_destructive=False: executed.append(
        allow_destructive) or g.code_exec_tool.ExecResult(0, "done\n", "")), g.code_exec_tool.code_exec
    try:
        res, _ = run(fake, "да, подтверждаю", thread=thread)
    finally:
        g.code_exec_tool.code_exec = orig
    assert res["action_confirmed"] is True
    assert executed == [True]  # код дошёл до песочницы с разрешением
    router_msgs = [m for n, _, m in fake.calls if n == "router"][0]
    assert any("подтвердил" in m["content"] for m in router_msgs if m["role"] == "system")


def test_blocked_code_stops_immediately(fake):
    fake.script["router"] = [reply('{"action": "delegate_executor", "reasoning": "код", "subtask": "выполнить код"}')]
    fake.script["executor"] = [reply(tool_calls=[FakeToolCall("code_exec", {"code": "import socket; socket.socket()"})])]
    res, _ = run(fake, "Выполни: import socket; socket.socket()")
    assert res["blocked"] and "заблокирован" in res["blocked"]
    assert res["step_count"] == 1                       # раньше — 6 попыток до MAX_STEPS
    assert nodes(fake) == ["router", "executor"]  # без критика, повторов и LLM-финализатора
    assert res["final_answer"].startswith("Действие не выполнено")


def test_memory_stores_only_user_message(fake):
    run(fake, "Меня зовут Ильназ, я учусь в КФУ")
    assert fake.memory.added == ["Меня зовут Ильназ, я учусь в КФУ"]
    run(fake, "Посчитай 18% от 4500")  # поручение — не факт о пользователе
    assert fake.memory.added == ["Меня зовут Ильназ, я учусь в КФУ"]


def test_skills_injected_into_executor_prompt(fake):
    fake.script["router"] = [reply('{"action": "delegate_executor", "reasoning": "расчёт", '
                                   '"subtask": "проанализируй sales.csv"}')]
    fake.script["executor"] = [reply("готово")]
    fake.script["critic"] = [reply('{"verdict": "approve", "issues": [], "suggested_fix": ""}')]
    res, _ = run(fake, "Проанализируй sales.csv")
    executor_prompt = [p for n, p, _ in fake.calls if n == "executor"][0]
    assert "## Skill: data-analysis" in executor_prompt and "## Skill: code-review" in executor_prompt
    assert set(res["skills_used"]) >= {"code-review", "data-analysis"}


def test_unverified_numbers_guardrail():
    calls = [{"tool": "code_exec", "result": json.dumps({"stdout": "338350\n552.5\n"})}]
    assert g._unverified_numbers("Сумма 338350, процент 552.5", calls, "посчитай") == []
    assert g._unverified_numbers("Итого 338902.5", calls, "посчитай") == ["338902.5"]


@pytest.mark.parametrize("text,expected", [
    ('{"action": "ask_user"}', {"action": "ask_user"}),
    ('```json\n{"action": "answer_directly"}\n```', {"action": "answer_directly"}),
    ("не JSON", None),
])
def test_parse_json_response(text, expected):
    assert g._parse_json_response(text) == expected


def test_memory_facts_in_chronological_order_with_newer_wins_rule(fake):
    item = lambda text, ts: SimpleNamespace(text=text, created_at=ts)  # noqa: E731
    fake.memory.search = lambda user_id, query, top_k=5: [
        item("User recently moved to Saint Petersburg", 1_790_000_200.0),
        item("User lives in Moscow", 1_790_000_100.0),
    ]
    run(fake, "Где я живу?")
    block = [m["content"] for node, _, msgs in fake.calls if node == "router"
             for m in msgs if m["content"].startswith("Релевантный контекст из памяти")][0]
    assert block.index("Moscow") < block.index("Saint Petersburg")  # старый факт выше нового
    assert "верен более поздний" in block


def test_mem0_extraction_rules_forbid_answers():
    from memory.memory_manager import MEMORY_EXTRACTION_RULES as rules
    assert "Never answer questions" in rules and "<CITY>" in rules


def test_subagent_gets_original_user_request(fake):
    fake.script["router"] = [reply('{"action": "delegate_executor", "reasoning": "расчёт", '
                                   '"subtask": "посчитать среднее продаж"}')]  # роутер «потерял» числа
    fake.script["executor"] = [reply("готово")]
    fake.script["critic"] = [reply('{"verdict": "approve", "issues": [], "suggested_fix": ""}')]
    run(fake, "Посчитай среднее продаж: 120, 135, 90")
    executor_user_msg = [m for n, _, msgs in fake.calls if n == "executor" for m in msgs if m["role"] == "user"][0]
    assert "120, 135, 90" in executor_user_msg["content"]
