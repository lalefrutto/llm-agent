"""
graph.py — оркестратор мультиагентной системы на LangGraph.

Почему LangGraph как фреймворк (кратко — полная версия в отчёте,
раздел "Выбор фреймворка"):

- Явный граф состояний вместо неявного "чата агентов" (как в AutoGen)
  — легче гарантировать отсутствие несвязанных single-агентов (см.
  red flag в ТЗ: "несколько сингл агентов, не объединённых в
  систему") — маршрутизация здесь ЯВНО закодирована рёбрами графа.
- Встроенный checkpointer даёт короткосрочную память/персистентность
  состояния треда "из коробки", без самодельного решения.
- По сравнению с CrewAI (декларативные роли, скрытая оркестрация)
  даёт больше контроля над условными переходами и циклами — важно для
  guardrail'ов (лимит шагов, критик с повторной проверкой).
- По сравнению с "без фреймворка" — не пишем с нуля персистентность,
  стриминг, визуализацию графа и retry-логику.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import asdict
from typing import Literal, TypedDict

import yaml
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END

from agent import metrics as M
from agent.llm_client import LLMClient
from agent.tools import code_exec as code_exec_tool
from agent.tools import file_ops as file_ops_tool
from agent.tools import web_search as web_search_tool
from memory.memory_manager import get_memory_backend

# Langfuse (SDK v2, сервер langfuse/langfuse:2 из docker-compose). Если
# ключи не заданы, декоратор @observe тихо отключается — код не меняется.
try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:  # langfuse не установлен — no-op декоратор
    def observe(*_a, **_k):
        def deco(fn):
            return fn
        return deco
    langfuse_context = None

MAX_STEPS = 6  # guardrail против бесконечных циклов делегирования (см. prompts/system_prompt.md)
MAX_TOOL_ITERS = 3  # сколько раундов tool-calling даём под-агенту на одну подзадачу


class AgentState(TypedDict):
    user_id: str
    thread_id: str
    messages: list[dict]          # полная история сообщений треда
    memory_context: list[str]     # факты, подтянутые из долгосрочной памяти
    route: str                    # решение роутера на текущем шаге
    route_history: list[str]      # все решения роутера за запрос (для evals routing accuracy)
    subtask: str                  # формулировка подзадачи для под-агента
    subagent_result: dict | None  # последний результат под-агента
    critic_verdict: dict | None
    critic_feedback: str | None    # issues+fix последнего revise/reject — передаётся под-агенту напрямую
    step_count: int
    final_answer: str | None


def _load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_json_response(text: str) -> dict | None:
    """Модели нередко оборачивают JSON в ```json ... ``` code fence или
    добавляют пояснительный текст вокруг — прямой json.loads() на это
    ломается и решение всегда уходит в fallback. Пробуем как есть, затем
    вырезаем первую сбалансированную {...} подстроку."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    if not isinstance(text, str):
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


SYSTEM_PROMPT = _load_prompt("prompts/system_prompt.md")
RESEARCHER_PROMPT = _load_prompt("prompts/subagent_researcher.md")
EXECUTOR_PROMPT = _load_prompt("prompts/subagent_executor.md")
CRITIC_PROMPT = _load_prompt("prompts/subagent_critic.md")
SOUL = _load_prompt("identity/SOUL.md")

llm = LLMClient()
memory = get_memory_backend()

with open("config/models.yaml", encoding="utf-8") as _f:
    ROLE_MODELS: dict[str, str] = (yaml.safe_load(_f) or {}).get("role_assignment", {})


def _model_for(role: str) -> str:
    """Модель для роли из config/models.yaml; AGENT_MODEL_OVERRIDE (env)
    принудительно переопределяет все роли (удобно для evals одной моделью)."""
    return os.environ.get("AGENT_MODEL_OVERRIDE") or ROLE_MODELS.get(role) or llm.config.model


def _chat(node: str, messages: list[dict], **kw):
    """Обёртка над llm.chat с метриками латентности/токенов по узлу графа
    и generation-спаном в Langfuse."""
    t0 = time.perf_counter()
    reply = llm.chat(messages, **kw)
    dt = time.perf_counter() - t0
    M.LLM_LATENCY.labels(node=node).observe(dt)
    usage = llm.last_usage
    if usage is not None:
        M.TOKENS_TOTAL.labels(type="prompt").inc(getattr(usage, "prompt_tokens", 0) or 0)
        M.TOKENS_TOTAL.labels(type="completion").inc(getattr(usage, "completion_tokens", 0) or 0)
    if langfuse_context is not None:
        try:
            langfuse_context.update_current_observation(
                model=kw.get("model") or llm.config.model,
                input=messages,
                output=reply.content if reply.content else [tc.model_dump() for tc in (reply.tool_calls or [])],
                usage={"input": getattr(usage, "prompt_tokens", 0), "output": getattr(usage, "completion_tokens", 0)}
                if usage else None,
                metadata={"node": node, "latency_s": round(dt, 3)},
            )
        except Exception:  # noqa: BLE001 — трейсинг не должен ронять граф
            pass
    return reply


# ---------------------------------------------------------------- tools ---

def _dispatch_tool(name: str, args: dict) -> str:
    """Выполняет инструмент под-агента и возвращает строку для tool-сообщения."""
    try:
        if name == "code_exec":
            res = code_exec_tool.code_exec(args.get("code", ""), args.get("timeout_s", code_exec_tool.DEFAULT_TIMEOUT_S))
            if res.exit_code == -1 and "заблокирован" in res.stderr:
                M.SANDBOX_BLOCKED.labels(reason="static_check").inc()
                M.TOOL_CALLS.labels(tool=name, status="blocked").inc()
            else:
                M.TOOL_CALLS.labels(tool=name, status="ok" if res.exit_code == 0 else "error").inc()
            return json.dumps(asdict(res), ensure_ascii=False)
        if name == "file_read":
            out = file_ops_tool.file_read(args["path"])
            M.TOOL_CALLS.labels(tool=name, status="ok").inc()
            return out[:8000]
        if name == "file_write":
            file_ops_tool.file_write(args["path"], args.get("content", ""))
            M.TOOL_CALLS.labels(tool=name, status="ok").inc()
            return json.dumps({"ok": True, "files_changed": [args["path"]]}, ensure_ascii=False)
        if name == "web_search":
            results = web_search_tool.web_search(args["query"], int(args.get("max_results", 5)))
            M.TOOL_CALLS.labels(tool=name, status="ok").inc()
            return json.dumps([asdict(r) for r in results], ensure_ascii=False)
    except file_ops_tool.PathEscapeError as e:
        M.SANDBOX_BLOCKED.labels(reason="path_escape").inc()
        M.TOOL_CALLS.labels(tool=name, status="blocked").inc()
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        M.TOOL_CALLS.labels(tool=name, status="error").inc()
        return json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
    M.TOOL_CALLS.labels(tool=name, status="unknown").inc()
    return json.dumps({"error": f"неизвестный инструмент {name}"}, ensure_ascii=False)


def _tool_loop(node: str, messages: list[dict], tools: list[dict], model: str) -> tuple[str, list[dict]]:
    """Стандартный цикл tool-calling: модель -> tool_calls -> результаты ->
    модель, максимум MAX_TOOL_ITERS раундов. Возвращает финальный текст
    под-агента и журнал вызовов инструментов (уходит в subagent_result,
    чтобы критик и финализатор видели реальные stdout/stderr, а не
    пересказ модели)."""
    calls_log: list[dict] = []
    for _ in range(MAX_TOOL_ITERS):
        reply = _chat(node, messages, tools=tools, model=model)
        tool_calls = reply.tool_calls or []
        if not tool_calls:
            return reply.content or "", calls_log
        messages.append({
            "role": "assistant", "content": reply.content or "",
            "tool_calls": [tc.model_dump() for tc in tool_calls],
        })
        for tc in tool_calls:
            args = _parse_json_response(tc.function.arguments) or {}
            result = _dispatch_tool(tc.function.name, args)
            calls_log.append({"tool": tc.function.name, "args": args, "result": result[:2000]})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    # Лимит раундов исчерпан — просим модель подвести итог без инструментов.
    reply = _chat(node, messages, model=model)
    return reply.content or "", calls_log


_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def _unverified_numbers(output: str, calls: list[dict], subtask: str) -> list[str]:
    """Детерминированный guardrail поверх критика-LLM: числа, которые executor
    сообщает в output, но которых нет ни в stdout его code_exec-вызовов, ни в
    самой формулировке задачи, — выдуманы (наблюдалось вживую: модель
    считает одно выражение, а в ответе "расщепляет" результат на две
    величины из головы). 8B-критик такое стабильно пропускает."""
    stdout_parts = []
    for c in calls:
        if c["tool"] == "code_exec":
            try:
                stdout_parts.append(json.loads(c["result"]).get("stdout", ""))
            except (json.JSONDecodeError, TypeError):
                stdout_parts.append(str(c["result"]))
        else:
            stdout_parts.append(str(c["result"]))
    stdout = " ".join(stdout_parts)
    allowed = {n.replace(",", ".") for n in _NUM_RE.findall(stdout)}
    allowed |= {n.replace(",", ".") for n in _NUM_RE.findall(subtask)}
    bad = []
    for n in _NUM_RE.findall(output):
        n2 = n.replace(",", ".")
        if n2 in allowed or len(n2.lstrip("-")) < 2:  # одиночные цифры (нумерация списков) не считаем
            continue
        # 338902 допустимо, если в stdout было 338902.5 (усечение дробной части)
        if any(a.split(".")[0] == n2.split(".")[0] for a in allowed):
            continue
        bad.append(n)
    return sorted(set(bad))


# ---------------------------------------------------------------- nodes ---

@observe(name="memory_retrieval")
def retrieve_memory(state: AgentState) -> AgentState:
    last_user_msg = next((m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), "")
    results = memory.search(state["user_id"], last_user_msg, top_k=5)
    state["memory_context"] = [r.text for r in results]
    return state


@observe(name="router", as_type="generation")
def route(state: AgentState) -> AgentState:
    if state["step_count"] >= MAX_STEPS:
        M.MAX_STEPS_REACHED.inc()
        state["route"] = "finalize"
        return state

    context_block = "\n".join(f"- {c}" for c in state["memory_context"]) or "(память пуста)"
    messages = [
        {"role": "system", "content": SOUL + "\n\n" + SYSTEM_PROMPT},
        {"role": "system", "content": f"Релевантный контекст из памяти:\n{context_block}"},
        *state["messages"],
    ]
    verdict = state.get("critic_verdict") or {}
    if verdict.get("verdict") in ("revise", "reject"):
        # Цикл критик -> роутер: переделегируем с учётом issues, а не отвечаем
        # пользователю непроверенным результатом.
        messages.append({"role": "system", "content": (
            "Критик вернул verdict=" + verdict["verdict"] + " для результата под-агента "
            + str((state.get("subagent_result") or {}).get("agent"))
            + ". Issues: " + json.dumps(verdict.get("issues", []), ensure_ascii=False)
            + ". Рекомендация: " + str(verdict.get("suggested_fix", ""))
            + "\nОБЯЗАТЕЛЬНО переделегируй задачу тому же под-агенту (action=delegate_executor "
              "или delegate_researcher), включив в subtask исходную задачу И требуемое исправление. "
              "Не отвечай пользователю сам.")})
        # 8B-роутер стабильно игнорирует просьбу включить fix в subtask —
        # передаём фидбэк под-агенту программно через state (см. run_executor).
        state["critic_feedback"] = ("Предыдущая попытка отклонена критиком. Issues: "
                                    + "; ".join(map(str, verdict.get("issues", [])))
                                    + ". Требуемое исправление: " + str(verdict.get("suggested_fix", "")))
        state["critic_verdict"] = None  # вердикт использован
    elif state.get("subagent_result"):
        messages.append({"role": "system", "content": (
            "Результат предыдущего шага под-агента: "
            + json.dumps(state["subagent_result"], ensure_ascii=False)[:4000]
            + "\nЕсли этого достаточно для ответа — верни action=answer_directly.")})
    reply = _chat("router", messages, json_mode=True, model=_model_for("orchestrator"))
    decision = _parse_json_response(reply.content)
    if decision is None:
        # Модель не вернула валидный JSON — безопасный дефолт: ответить напрямую,
        # не пытаясь угадать делегирование, и залогировать инцидент (см. observability/).
        M.ROUTING_INVALID_JSON.inc()
        decision = {"action": "answer_directly", "reasoning": "fallback: invalid routing JSON", "subtask": ""}

    state["route"] = decision.get("action", "answer_directly")
    state.setdefault("route_history", []).append(state["route"])
    M.ROUTE_DECISIONS.labels(action=state["route"]).inc()
    state["subtask"] = decision.get("subtask") or ""
    state["step_count"] += 1
    return state


@observe(name="researcher", as_type="generation")
def run_researcher(state: AgentState) -> AgentState:
    messages = [
        {"role": "system", "content": SOUL + "\n\n" + RESEARCHER_PROMPT},
        {"role": "user", "content": state["subtask"]},
    ]
    if state.get("critic_feedback"):
        messages.append({"role": "user", "content": state["critic_feedback"]})
        state["critic_feedback"] = None
    output, calls = _tool_loop("researcher", messages, [web_search_tool.TOOL_SCHEMA], _model_for("researcher"))
    state["subagent_result"] = {"agent": "researcher", "output": output, "tool_calls": calls}
    return state


@observe(name="executor", as_type="generation")
def run_executor(state: AgentState) -> AgentState:
    messages = [
        {"role": "system", "content": SOUL + "\n\n" + EXECUTOR_PROMPT},
        {"role": "user", "content": state["subtask"]},
    ]
    if state.get("critic_feedback"):
        messages.append({"role": "user", "content": state["critic_feedback"]
                         + " Напиши НОВЫЙ код: каждую величину — отдельной строкой print(...)."})
        state["critic_feedback"] = None
    tools = [code_exec_tool.TOOL_SCHEMA, *file_ops_tool.TOOL_SCHEMAS]
    output, calls = _tool_loop("executor", messages, tools, _model_for("executor"))
    state["subagent_result"] = {"agent": "executor", "output": output, "tool_calls": calls}
    if any(c["tool"] == "code_exec" for c in calls):
        bad = _unverified_numbers(output, calls, state["subtask"])
        if bad:
            state["subagent_result"]["unverified_numbers"] = bad
    return state


@observe(name="critic", as_type="generation")
def run_critic(state: AgentState) -> AgentState:
    messages = [
        {"role": "system", "content": CRITIC_PROMPT},
        {"role": "user", "content": json.dumps(state["subagent_result"], ensure_ascii=False)[:6000]},
    ]
    reply = _chat("critic", messages, json_mode=True, model=_model_for("critic"))
    verdict = _parse_json_response(reply.content)
    state["critic_verdict"] = verdict if verdict is not None else {
        "verdict": "approve", "issues": [], "suggested_fix": ""}
    bad = (state.get("subagent_result") or {}).get("unverified_numbers")
    if bad and state["critic_verdict"].get("verdict") == "approve":
        # LLM-критик пропустил выдуманные числа — принудительный revise.
        state["critic_verdict"] = {
            "verdict": "revise",
            "issues": [f"Числа {bad} в ответе executor отсутствуют в stdout code_exec — они не получены кодом."],
            "suggested_fix": "Каждую требуемую величину вычислить отдельно и вывести отдельной строкой через print(...); "
                             "в ответе использовать только значения из stdout.",
            "auto": True,
        }
        M.CRITIC_VERDICTS.labels(verdict="revise_auto").inc()
    else:
        M.CRITIC_VERDICTS.labels(verdict=str(state["critic_verdict"].get("verdict", "unknown"))).inc()
    return state


@observe(name="finalize", as_type="generation")
def finalize(state: AgentState) -> AgentState:
    context_block = json.dumps(state.get("subagent_result") or {}, ensure_ascii=False)[:6000]
    memory_block = "\n".join(f"- {c}" for c in state["memory_context"]) or "(память пуста)"
    messages = [
        {"role": "system", "content": SOUL + "\n\n" + SYSTEM_PROMPT},
        {"role": "system", "content": f"Релевантный контекст из памяти:\n{memory_block}"},
        {"role": "system", "content": f"Результаты под-агентов: {context_block}"},
        *state["messages"],
        {"role": "system", "content": (
            "Это финальный шаг: маршрутизация уже завершена. Ответь пользователю "
            "обычным текстом по существу его вопроса — БЕЗ служебного JSON и без "
            "упоминания action/reasoning/subtask (см. раздел 'Формат финального "
            "ответа пользователю'). Если под-агент выполнял код — опирайся на его "
            "реальный stdout, а не считай заново. Если пользователь ограничил источник "
            "('только на основе этого текста/данных') — сначала проверь, есть ли ответ "
            "буквально в этом источнике; если нет — начни ответ с фразы «В приведённом "
            "тексте это не указано» и не приписывай источнику своих знаний."
        )},
    ]
    bad = (state.get("subagent_result") or {}).get("unverified_numbers")
    if bad:
        # Дошли до финализации с неподтверждёнными числами (например, по MAX_STEPS):
        # не выдаём их за результат вычислений.
        messages.append({"role": "system", "content": (
            f"ВНИМАНИЕ: числа {bad} из ответа под-агента НЕ подтверждены выполнением кода "
            "(их нет в stdout). Сообщи пользователю только значения из stdout code_exec и явно "
            "скажи, что остальные величины вычислить не удалось — не приводи их как результат.")})
    reply = _chat("finalize", messages, model=_model_for("orchestrator"))
    state["final_answer"] = reply.content

    last_user_msg = next((m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), "")
    memory.add(state["user_id"], f"Q: {last_user_msg}\nA: {state['final_answer']}")
    return state


def _route_selector(state: AgentState) -> Literal["researcher", "executor", "critic", "finalize"]:
    mapping = {
        "delegate_researcher": "researcher",
        "delegate_executor": "executor",
        "delegate_critic": "critic",
        "answer_directly": "finalize",
        "ask_user": "finalize",
        "finalize": "finalize",
    }
    return mapping.get(state["route"], "finalize")


def _after_subagent(state: AgentState) -> Literal["router", "critic"]:
    # Executor-результаты с высокой ценой ошибки уходят на критика;
    # остальные — обратно роутеру для следующего шага/финализации.
    if state.get("subagent_result", {}).get("agent") == "executor":
        return "critic"
    return "router"


def _after_critic(state: AgentState) -> Literal["router", "finalize"]:
    verdict = (state.get("critic_verdict") or {}).get("verdict", "approve")
    if verdict == "approve" or state["step_count"] >= MAX_STEPS:
        return "finalize"
    return "router"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("memory_retrieval", retrieve_memory)
    graph.add_node("router", route)
    graph.add_node("researcher", run_researcher)
    graph.add_node("executor", run_executor)
    graph.add_node("critic", run_critic)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("memory_retrieval")
    graph.add_edge("memory_retrieval", "router")
    graph.add_conditional_edges("router", _route_selector, {
        "researcher": "researcher", "executor": "executor",
        "critic": "critic", "finalize": "finalize",
    })
    graph.add_conditional_edges("researcher", _after_subagent, {"router": "router", "critic": "critic"})
    graph.add_conditional_edges("executor", _after_subagent, {"router": "router", "critic": "critic"})
    graph.add_conditional_edges("critic", _after_critic, {"router": "router", "finalize": "finalize"})
    graph.add_edge("finalize", END)

    # from_conn_string() возвращает контекст-менеджер (закрывает соединение
    # при выходе) — для процесса, живущего весь аптайм приложения, открываем
    # соединение напрямую и передаём его в SqliteSaver.
    conn = sqlite3.connect(os.environ.get("CHECKPOINT_DB", "./checkpoints.sqlite"), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return graph.compile(checkpointer=checkpointer)


def init_state(user_id: str, thread_id: str, user_message: str) -> AgentState:
    return {
        "user_id": user_id,
        "thread_id": thread_id,
        "messages": [{"role": "user", "content": user_message}],
        "memory_context": [],
        "route": "",
        "route_history": [],
        "subtask": "",
        "subagent_result": None,
        "critic_verdict": None,
        "critic_feedback": None,
        "step_count": 0,
        "final_answer": None,
    }


@observe(name="atlas_request")
def run_request(app, user_id: str, thread_id: str, user_message: str) -> AgentState:
    """Один запрос пользователя = один трейс Langfuse + один замер
    agent_graph_latency_seconds. Ключевой момент: состояние треда
    (checkpointer) хранит messages от прошлых ходов, но новый invoke
    передаёт свежий state с одним сообщением — LangGraph подменяет
    значения ключей, а не сливает списки, поэтому историю треда
    восстанавливаем из последнего чекпойнта вручную."""
    if langfuse_context is not None:
        try:
            langfuse_context.update_current_trace(user_id=user_id, session_id=thread_id,
                                                  input=user_message)
        except Exception:  # noqa: BLE001
            pass
    cfg = {"configurable": {"thread_id": thread_id}}
    state = init_state(user_id, thread_id, user_message)
    prev = app.get_state(cfg)
    if prev and prev.values.get("messages"):
        prev_msgs = prev.values["messages"]
        prev_answer = prev.values.get("final_answer")
        history = list(prev_msgs)
        if prev_answer and (not history or history[-1].get("role") != "assistant"):
            history.append({"role": "assistant", "content": prev_answer})
        state["messages"] = history[-20:] + state["messages"]  # окно короткосрочной памяти
    t0 = time.perf_counter()
    result = app.invoke(state, config=cfg)
    M.GRAPH_LATENCY.observe(time.perf_counter() - t0)
    if langfuse_context is not None:
        try:
            langfuse_context.update_current_trace(output=result.get("final_answer"),
                                                  metadata={"route_history": result.get("route_history"),
                                                            "step_count": result.get("step_count")})
        except Exception:  # noqa: BLE001
            pass
    return result


if __name__ == "__main__":
    # Требует запущенный Ollama с загруженными моделями — см. README.md.
    app = build_graph()
    result = run_request(app, "demo_user", "demo_thread", "Привет! Посчитай 15% от 340.")
    print(result["final_answer"])
