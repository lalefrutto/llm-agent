"""
compare_models.py — реальное сравнение кандидатов из config/models.yaml
на evals/dataset.jsonl через agent/llm_client.py (см. README.md, шаг 2).

Гоняет каждую не-guardrail задачу датасета на каждой модели напрямую
через OpenAI-совместимый клиент Ollama, меряет latency и считает
простые, но объективные метрики:
  - routing accuracy       (t01-t03, t09): совпадение action с ожиданием
  - JSON-валидность ответа (proxy для "качества tool-calling"/инструктивности)
  - hallucination guard    (t10): не выдумывает факт, которого нет в тексте
  - task_success           (t08): ответ содержит ожидаемые упоминания
  - critic_catch           (t07): критик возвращает ожидаемый verdict
  - memory_correctness     (t06): модель следует обновлению факта в диалоге
  - tool-calling           (TOOL_CASES): модель получает реальные tool
                           schemas из agent/tools/*, ожидается tool_call
                           с валидными JSON-аргументами под схему
  - инструктивность        рубрика 0..1 по каждому кейсу (см. rubric_*),
                           усреднённая по всем кейсам модели
  - hallucination rate     t10 гоняется HALLU_SAMPLES раз, считаем долю
                           ответов, где модель выдала "4 лапы" как факт
  - токены/сек             completion_tokens / wall-time по usage Ollama

Результат — evals/results/model_comparison.md.

Требует запущенный `ollama serve` с уже загруженными моделями из
config/models.yaml.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.llm_client import LLMClient, LLMConfig  # noqa: E402

MODELS = [
    "qwen2.5:7b-instruct-q4_K_M",
    "llama3.1:8b-instruct-q4_K_M",
    "hermes3:8b-llama3.1-q4_K_M",
    "phi3.5:3.8b-mini-instruct-q4_K_M",
]

from agent.tools.code_exec import TOOL_SCHEMA as CODE_EXEC_SCHEMA  # noqa: E402
from agent.tools.file_ops import TOOL_SCHEMAS as FILE_OPS_SCHEMAS  # noqa: E402

HALLU_SAMPLES = 5  # t10 — один кейс, поэтому семплируем несколько раз

# Реальные tool-calling кейсы: модель получает схемы из agent/tools и
# должна вернуть tool_call (а не текст) с аргументами, валидными по схеме.
TOOL_CASES = [
    {"id": "tc01", "input": "Посчитай через code_exec, сколько будет 18% от 4500. "
                            "Обязательно вызови инструмент, не считай в уме.",
     "tools": [CODE_EXEC_SCHEMA], "expect_tool": "code_exec", "required_args": ["code"]},
    {"id": "tc02", "input": "Прочитай файл data/report.csv из рабочей директории.",
     "tools": FILE_OPS_SCHEMAS, "expect_tool": "file_read", "required_args": ["path"]},
    {"id": "tc03", "input": "Сохрани текст 'hello atlas' в файл notes/hello.txt.",
     "tools": FILE_OPS_SCHEMAS, "expect_tool": "file_write", "required_args": ["path", "content"]},
]

ALLOWED_ACTIONS = {"answer_directly", "delegate_researcher", "delegate_executor",
                   "delegate_critic", "ask_user"}
ALLOWED_VERDICTS = {"approve", "revise", "reject"}

SOUL = (ROOT / "identity" / "SOUL.md").read_text(encoding="utf-8")
SYSTEM_PROMPT = (ROOT / "prompts" / "system_prompt.md").read_text(encoding="utf-8")
CRITIC_PROMPT = (ROOT / "prompts" / "subagent_critic.md").read_text(encoding="utf-8")

HEDGE_WORDS = ["не указано", "не сказано", "нет информации", "неизвестно",
               "не содержит", "не указан", "не упомянут", "не даёт", "не дано",
               "не приведено", "нельзя определить", "не хватает информации"]


def load_dataset(path: str) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def timed_chat(client: LLMClient, model: str, messages: list[dict], json_mode: bool = False):
    t0 = time.perf_counter()
    reply = client.chat(messages, model=model, json_mode=json_mode)
    dt = time.perf_counter() - t0
    return reply, dt


def extract_json(text: str) -> str:
    """Модели часто оборачивают JSON в ```json ... ``` или добавляют
    пояснительный текст вокруг — вырезаем первую сбалансированную
    {...} подстроку, чтобы не терять валидные ответы на этом фоне."""
    if text is None:
        return text
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def try_json(text: str):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return json.loads(extract_json(text))
    except (json.JSONDecodeError, TypeError):
        return None


def _is_mostly_russian(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    cyr = sum(1 for c in letters if "\u0400" <= c <= "\u04ff")
    return cyr / len(letters) >= 0.6


def rubric_routing(decision, case_cat: str) -> float:
    """Инструктивность для JSON-решений роутера (0..1), 4 критерия из
    prompts/system_prompt.md: (1) валидный JSON-объект, (2) есть все три
    ключа action/reasoning/subtask, (3) action из разрешённого списка,
    (4) при делегировании subtask непустой, при answer_directly — нет
    лишних полей вроде "result" (модель не должна отвечать в JSON)."""
    if not isinstance(decision, dict):
        return 0.0
    score = 1.0
    score += float(all(k in decision for k in ("action", "reasoning", "subtask")))
    action = decision.get("action")
    score += float(action in ALLOWED_ACTIONS)
    if action and action.startswith("delegate"):
        score += float(bool(decision.get("subtask")))
    else:
        extra = set(decision) - {"action", "reasoning", "subtask"}
        score += float(not extra)
    return score / 4


def rubric_critic(verdict) -> float:
    """(1) валидный JSON, (2) ключи verdict/issues/suggested_fix,
    (3) verdict из allowed-списка, (4) issues — непустой список строк."""
    if not isinstance(verdict, dict):
        return 0.0
    score = 1.0
    score += float(all(k in verdict for k in ("verdict", "issues", "suggested_fix")))
    score += float(verdict.get("verdict") in ALLOWED_VERDICTS)
    issues = verdict.get("issues")
    score += float(isinstance(issues, list) and len(issues) > 0
                   and all(isinstance(i, str) for i in issues))
    return score / 4


def rubric_text(text: str) -> float:
    """Для свободных ответов: (1) непустой, (2) на русском (SOUL.md),
    (3) без служебного JSON/action в тексте, (4) не оборван (заканчивается
    знаком препинания, не обрезан max_tokens)."""
    if not text or not text.strip():
        return 0.0
    t = text.strip()
    score = 1.0
    score += float(_is_mostly_russian(t))
    score += float('"action"' not in t and "reasoning:" not in t.lower())
    score += float(t[-1] in ".!?)\u00bb\"*`" or t.endswith("```"))
    return score / 4


def _tok_per_sec(client: LLMClient, dt: float) -> float | None:
    u = client.last_usage
    if u is None or not getattr(u, "completion_tokens", None) or dt <= 0:
        return None
    return u.completion_tokens / dt


def evaluate_model(model: str, cases: list[dict]) -> dict:
    client = LLMClient(LLMConfig(model=model))
    metrics = {
        "latencies": [], "json_valid": 0, "json_total": 0,
        "routing_ok": 0, "routing_total": 0,
        "step_eff_ok": 0, "step_eff_total": 0,
        "hallucination_ok": 0, "hallucination_total": 0,
        "task_success_ok": 0, "task_success_total": 0,
        "critic_ok": 0, "critic_total": 0,
        "memory_ok": 0, "memory_total": 0,
        "tool_ok": 0, "tool_total": 0,
        "hallu_hits": 0, "hallu_samples": 0,
        "rubric": [],           # инструктивность по кейсам, 0..1
        "tok_per_sec": [],      # токены/сек по вызовам, где есть usage
        "errors": 0,
    }
    detail = []

    def _record_speed(dt: float):
        tps = _tok_per_sec(client, dt)
        if tps is not None:
            metrics["tok_per_sec"].append(tps)

    for case in cases:
        cat = case["category"]
        if cat == "guardrail":
            continue
        try:
            if cat == "routing" or cat == "step_efficiency":
                messages = [
                    {"role": "system", "content": SOUL + "\n\n" + SYSTEM_PROMPT},
                    {"role": "system", "content": "Релевантный контекст из памяти:\n(память пуста)"},
                    {"role": "user", "content": case["input"]},
                ]
                reply, dt = timed_chat(client, model, messages, json_mode=True)
                metrics["latencies"].append(dt)
                _record_speed(dt)
                decision = try_json(reply.content)
                metrics["rubric"].append(rubric_routing(decision, cat))
                valid = isinstance(decision, dict) and "action" in decision
                metrics["json_total"] += 1
                metrics["json_valid"] += int(valid)
                if cat == "routing":
                    metrics["routing_total"] += 1
                    ok = valid and decision.get("action") == case["expected_route"]
                    metrics["routing_ok"] += int(ok)
                else:
                    metrics["step_eff_total"] += 1
                    ok = valid and decision.get("action") == "answer_directly"
                    metrics["step_eff_ok"] += int(ok)
                detail.append((case["id"], cat, ok, round(dt, 2),
                                decision if decision is not None else f"UNPARSED: {reply.content[:150]!r}"))

            elif cat == "hallucination":
                system = ("Отвечай СТРОГО на основе предоставленного текста. "
                          "Если ответа нет в тексте — явно скажи, что информация "
                          "не указана в тексте, не выдумывай факт из общих знаний.")
                messages = [{"role": "system", "content": system},
                            {"role": "user", "content": case["input"]}]
                # Один кейс — слишком шумно, семплируем HALLU_SAMPLES раз.
                # "Галлюцинация" = модель утверждает "4" как факт и при этом
                # НЕ оговаривает, что в тексте этого нет.
                hits, dt_sum, first_text = 0, 0.0, ""
                for _ in range(HALLU_SAMPLES):
                    reply, dt = timed_chat(client, model, messages)
                    dt_sum += dt
                    _record_speed(dt)
                    text = reply.content.lower()
                    hedged = any(w in text for w in HEDGE_WORDS)
                    asserts_four = ("4" in text or "четыре" in text or "четырёх" in text)
                    hallucinated = asserts_four and not hedged
                    hits += int(hallucinated)
                    metrics["hallu_samples"] += 1
                    metrics["rubric"].append(rubric_text(reply.content))
                    if not first_text:
                        first_text = reply.content
                metrics["hallu_hits"] += hits
                dt = dt_sum / HALLU_SAMPLES
                metrics["latencies"].append(dt)
                ok = hits == 0
                metrics["hallucination_total"] += 1
                metrics["hallucination_ok"] += int(ok)
                detail.append((case["id"], cat, ok, round(dt, 2),
                                f"галлюцинаций {hits}/{HALLU_SAMPLES}; пример: {first_text[:120]}"))

            elif cat == "task_success":
                messages = [{"role": "system", "content": SOUL},
                            {"role": "user", "content": case["input"]}]
                reply, dt = timed_chat(client, model, messages)
                metrics["latencies"].append(dt)
                _record_speed(dt)
                metrics["rubric"].append(rubric_text(reply.content))
                ok = all(m.lower() in reply.content.lower() for m in case["expected_answer_mentions"])
                metrics["task_success_total"] += 1
                metrics["task_success_ok"] += int(ok)
                detail.append((case["id"], cat, ok, round(dt, 2), reply.content[:150]))

            elif cat == "critic_catch":
                messages = [{"role": "system", "content": CRITIC_PROMPT},
                            {"role": "user", "content": case["input"]}]
                reply, dt = timed_chat(client, model, messages, json_mode=True)
                metrics["latencies"].append(dt)
                _record_speed(dt)
                verdict = try_json(reply.content)
                metrics["rubric"].append(rubric_critic(verdict))
                metrics["json_total"] += 1
                metrics["json_valid"] += int(verdict is not None)
                ok = isinstance(verdict, dict) and verdict.get("verdict") == case["expected_critic_verdict"]
                metrics["critic_total"] += 1
                metrics["critic_ok"] += int(ok)
                detail.append((case["id"], cat, ok, round(dt, 2),
                                verdict if verdict is not None else f"UNPARSED: {reply.content[:150]!r}"))

            elif cat == "memory_correctness":
                turns = case["turns"]
                messages = [{"role": "system", "content": SOUL}]
                reply = None
                dt_total = 0.0
                for turn in turns:
                    messages.append({"role": "user", "content": turn})
                    reply, dt = timed_chat(client, model, messages)
                    dt_total += dt
                    _record_speed(dt)
                    messages.append({"role": "assistant", "content": reply.content})
                metrics["latencies"].append(dt_total)
                metrics["rubric"].append(rubric_text(reply.content if reply else ""))
                final_text = reply.content.lower() if reply else ""
                ok = case["expected_answer_contains"].lower() in final_text
                metrics["memory_total"] += 1
                metrics["memory_ok"] += int(ok)
                detail.append((case["id"], cat, ok, round(dt_total, 2), reply.content[:150] if reply else ""))

        except Exception as e:  # noqa: BLE001
            metrics["errors"] += 1
            detail.append((case["id"], cat, False, None, f"ERROR: {e}"))

    # --- Реальный tool-calling: schemas из agent/tools, ожидаем tool_calls ---
    for tc in TOOL_CASES:
        try:
            messages = [
                {"role": "system", "content": SOUL + "\n\nТы — под-агент Executor. "
                 "Для вычислений и файловых операций ВСЕГДА используй доступные инструменты."},
                {"role": "user", "content": tc["input"]},
            ]
            t0 = time.perf_counter()
            reply = client.chat(messages, model=model, tools=tc["tools"])
            dt = time.perf_counter() - t0
            metrics["latencies"].append(dt)
            _record_speed(dt)
            calls = getattr(reply, "tool_calls", None) or []
            ok, info = False, f"нет tool_calls; text={str(reply.content)[:100]!r}"
            args_valid = False
            if calls:
                call = calls[0]
                name = call.function.name
                args = try_json(call.function.arguments)
                args_valid = isinstance(args, dict)
                schema_ok = args_valid and all(a in args for a in tc["required_args"])
                ok = name == tc["expect_tool"] and schema_ok
                info = f"{name}({json.dumps(args, ensure_ascii=False)[:120]})"
            metrics["tool_total"] += 1
            metrics["tool_ok"] += int(ok)
            metrics["json_total"] += 1
            metrics["json_valid"] += int(args_valid)
            metrics["rubric"].append(1.0 if ok else (0.5 if calls else 0.0))
            detail.append((tc["id"], "tool_calling", ok, round(dt, 2), info))
        except Exception as e:  # noqa: BLE001
            metrics["errors"] += 1
            metrics["tool_total"] += 1
            detail.append((tc["id"], "tool_calling", False, None, f"ERROR: {e}"))

    return metrics, detail


def fmt_pct(ok: int, total: int) -> str:
    return f"{ok}/{total} ({100 * ok / total:.0f}%)" if total else "n/a"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default=str(ROOT / "evals" / "dataset.jsonl"))
    parser.add_argument("--models", default=",".join(MODELS),
                        help="Подмножество моделей через запятую (для smoke-теста)")
    parser.add_argument("--out", default=str(ROOT / "evals" / "results" / "model_comparison.md"))
    args = parser.parse_args()
    dataset_path = args.dataset
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cases = load_dataset(dataset_path)

    all_results = {}
    all_detail = {}
    for model in models:
        print(f"\n=== {model} ===", flush=True)
        metrics, detail = evaluate_model(model, cases)
        all_results[model] = metrics
        all_detail[model] = detail
        for case_id, cat, ok, dt, extra in detail:
            print(f"  [{cat:20s}] {case_id}: {'OK' if ok else 'FAIL'} ({dt}s)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Сравнение моделей — evals/dataset.jsonl (реальный live-прогон)\n")
    lines.append(f"Датасет: `{Path(dataset_path).name}`, {len(cases)} кейсов "
                 f"(из них guardrail-кейсы пропущены — они не требуют модели, "
                 f"см. `run_evals.py`).\n")
    lines.append("Железо: NVIDIA RTX 4060 8GB VRAM, движок — Ollama "
                 "(`http://localhost:11434/v1`).\n")
    lines.append(f"Дата прогона: {time.strftime('%Y-%m-%d %H:%M')}. "
                 f"Дополнительно: {len(TOOL_CASES)} tool-calling кейса с реальными схемами "
                 f"из `agent/tools/`, t10 семплируется {HALLU_SAMPLES} раз.\n")
    lines.append("\n## Рубрика инструктивности (0..1 на кейс, среднее по модели)\n")
    lines.append("- JSON-решения роутера: валидный JSON / все ключи action+reasoning+subtask / "
                 "action из разрешённого списка / subtask заполнен при делегировании и нет лишних полей при answer_directly.")
    lines.append("- JSON критика: валидный JSON / ключи verdict+issues+suggested_fix / verdict из списка / issues — непустой список строк.")
    lines.append("- Свободный текст: непустой / на русском (SOUL.md) / без служебного JSON / не оборван.")
    lines.append("- Tool-calling: 1.0 — нужный инструмент + аргументы по схеме; 0.5 — вызов есть, но не тот/невалидный; 0 — вызова нет.\n")

    lines.append("\n## Сводная таблица\n")
    lines.append("| Модель | Инструктивность (рубрика) | Tool-calling (schemas) | JSON-валидность | "
                  "Routing acc. | Hallucination rate (t10×5) | Task success | Critic catch | "
                  "Memory correctness | Step efficiency | Avg latency, s | Tok/s | Errors |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for model in models:
        m = all_results[model]
        avg_lat = sum(m["latencies"]) / len(m["latencies"]) if m["latencies"] else 0
        rubric = sum(m["rubric"]) / len(m["rubric"]) if m["rubric"] else 0
        tps = sum(m["tok_per_sec"]) / len(m["tok_per_sec"]) if m["tok_per_sec"] else 0
        hallu = (f"{m['hallu_hits']}/{m['hallu_samples']} ({100 * m['hallu_hits'] / m['hallu_samples']:.0f}%)"
                 if m["hallu_samples"] else "n/a")
        lines.append(
            f"| `{model}` | {rubric:.2f} | "
            f"{fmt_pct(m['tool_ok'], m['tool_total'])} | "
            f"{fmt_pct(m['json_valid'], m['json_total'])} | "
            f"{fmt_pct(m['routing_ok'], m['routing_total'])} | "
            f"{hallu} | "
            f"{fmt_pct(m['task_success_ok'], m['task_success_total'])} | "
            f"{fmt_pct(m['critic_ok'], m['critic_total'])} | "
            f"{fmt_pct(m['memory_ok'], m['memory_total'])} | "
            f"{fmt_pct(m['step_eff_ok'], m['step_eff_total'])} | "
            f"{avg_lat:.2f} | {tps:.1f} | {m['errors']} |"
        )

    lines.append("\n## Детали по кейсам\n")
    for model in models:
        lines.append(f"\n### `{model}`\n")
        lines.append("| id | категория | результат | latency, s | вывод модели / решение |")
        lines.append("|---|---|---|---|---|")
        for case_id, cat, ok, dt, extra in all_detail[model]:
            extra_str = json.dumps(extra, ensure_ascii=False) if isinstance(extra, dict) else str(extra)
            extra_str = extra_str.replace("|", "\\|").replace("\n", " ")[:200]
            lines.append(f"| {case_id} | {cat} | {'✅' if ok else '❌'} | {dt} | {extra_str} |")

    lines.append("\n## Итоговая рекомендация\n")
    lines.append("_(заполняется вручную после анализа таблицы)_\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    raw_path = out_path.with_suffix(".json")
    raw_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nРезультаты записаны в {out_path} (+ {raw_path.name})")


if __name__ == "__main__":
    main()
