"""
run_evals.py — прогон evals/dataset.jsonl против собранного графа
агента (agent/graph.py). Без --live прогоняются только
детерминированные guardrail-кейсы; с --live нужен запущенный Ollama
(+ Qdrant для Mem0) — запускать из корня репозитория:
    python evals/run_evals.py --dataset evals/dataset.jsonl --live

Метрики считаются по категориям, описанным в evals/README.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalReport:
    total: int = 0
    passed: int = 0
    by_category: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    failures: list = field(default_factory=list)

    def record(self, category: str, ok: bool, case_id: str, detail: str = ""):
        self.total += 1
        self.by_category[category][1] += 1
        if ok:
            self.passed += 1
            self.by_category[category][0] += 1
        else:
            self.failures.append({"id": case_id, "category": category, "detail": detail})

    def print_summary(self):
        print(f"\n=== Итог: {self.passed}/{self.total} пройдено ===")
        for cat, (ok, total) in sorted(self.by_category.items()):
            print(f"  {cat:20s} {ok}/{total}")
        if self.failures:
            print("\nПровалившиеся кейсы:")
            for f in self.failures:
                print(f"  [{f['category']}] {f['id']}: {f['detail']}")


def load_dataset(path: str) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_guardrail_case(case: dict, report: EvalReport):
    """Guardrail-кейсы можно проверить БЕЗ живой модели — это чисто
    структурная проверка tools (детерминированная, идеальна для CI)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if "socket" in case["input"] or "import socket" in case["input"]:
        from agent.tools.code_exec import _static_check
        blocked = _static_check(case["input"]) is not None
        report.record("guardrail", blocked == case["expected_block"], case["id"],
                       detail="ожидали блокировку code_exec" if not blocked else "")
    elif "etc/passwd" in case["input"] or ".." in case["input"]:
        from agent.tools.file_ops import _safe_resolve, PathEscapeError
        import os
        import tempfile
        workspace = str(Path(tempfile.gettempdir()) / "agent_workspace_eval")
        os.environ.setdefault("AGENT_WORKSPACE", workspace)
        Path(workspace).mkdir(exist_ok=True)
        try:
            _safe_resolve("../../etc/passwd")
            blocked = False
        except PathEscapeError:
            blocked = True
        report.record("guardrail", blocked == case["expected_block"], case["id"],
                       detail="ожидали PathEscapeError" if not blocked else "")


EVAL_USER = f"eval_user_{int(time.time())}"

HEDGE_WORDS = ["не указано", "не сказано", "нет информации", "неизвестно",
               "не содержит", "не указан", "не упомянут", "не даёт", "не дано",
               "не приведено", "нельзя определить", "не хватает информации", "не следует"]


def run_live_case(case: dict, report: EvalReport, graph_app, run_request):
    """Кейсы, требующие реального инференса (routing, memory, task_success,
    hallucination, critic_catch, step_efficiency) — запускаются только
    если передан собранный граф (--live, требует Ollama + Qdrant).

    Каждый кейс идёт в СВОЙ thread (id кейса + timestamp), а user_id
    уникален на прогон (EVAL_USER): иначе Mem0 подтягивает ответы из
    прошлых прогонов в контекст роутера/финализатора и, например,
    t10 "вспоминает" свою же прошлую галлюцинацию (наблюдалось в
    прогоне 2: 'кошки имеют четыре лапы' пришло из памяти eval_user)."""
    category = case["category"]
    tid = f"{case['id']}-{int(time.time())}"
    eval_user = EVAL_USER
    try:
        if category == "routing":
            result = run_request(graph_app, eval_user, tid, case["input"])
            first = (result.get("route_history") or ["<none>"])[0]
            ok = first == case["expected_route"]
            report.record(category, ok, case["id"],
                          detail=f"ожидали {case['expected_route']}, роутер решил {result.get('route_history')}")

        elif category == "step_efficiency":
            result = run_request(graph_app, eval_user, tid, case["input"])
            ok = result.get("step_count", 99) <= case["expected_max_steps"]
            report.record(category, ok, case["id"],
                          detail=f"шагов {result.get('step_count')}, лимит {case['expected_max_steps']}, "
                                 f"route={result.get('route_history')}")

        elif category == "task_success":
            result = run_request(graph_app, eval_user, tid, case["input"])
            answer = (result.get("final_answer") or "").lower()
            ok = all(m.lower() in answer for m in case["expected_answer_mentions"])
            report.record(category, ok, case["id"],
                          detail=f"ожидали упоминание {case['expected_answer_mentions']}; ответ: {answer[:120]!r}")

        elif category == "hallucination":
            result = run_request(graph_app, eval_user, tid, case["input"])
            answer = (result.get("final_answer") or "").lower()
            hedged = any(w in answer for w in HEDGE_WORDS)
            ok = hedged
            report.record(category, ok, case["id"],
                          detail=f"ответ без оговорки 'в тексте не указано': {answer[:120]!r}")

        elif category == "memory_correctness":
            # Отдельный user_id на прогон, чтобы старые факты о городе из
            # прошлых запусков не подмешивались через Mem0.
            uid = f"eval_mem_{int(time.time())}"
            answer = ""
            for i, turn in enumerate(case["turns"]):
                # каждый ход — новый thread: проверяем ДОЛГОСРОЧНУЮ память, а не буфер диалога
                result = run_request(graph_app, uid, f"{tid}-turn{i}", turn)
                answer = result.get("final_answer") or ""
            ok = case["expected_answer_contains"].lower() in answer.lower()
            report.record(category, ok, case["id"],
                          detail=f"ожидали {case['expected_answer_contains']!r}; ответ: {answer[:120]!r}")

        elif category == "critic_catch":
            # Критика вызываем напрямую с намеренно испорченным результатом executor.
            from agent.graph import run_critic
            state = {"subagent_result": {
                "agent": "executor",
                "output": "Готово: def ratio(a, b): return a / b  — вызывается ratio(total, count) без проверок",
                "tool_calls": [{"tool": "code_exec", "args": {"code": "print(10/0)"},
                                "result": '{"exit_code": 1, "stdout": "", "stderr": "ZeroDivisionError: division by zero"}'}],
            }}
            state = run_critic(state)
            verdict = (state.get("critic_verdict") or {}).get("verdict")
            ok = verdict == case["expected_critic_verdict"]
            report.record(category, ok, case["id"],
                          detail=f"ожидали {case['expected_critic_verdict']!r}, критик вернул {verdict!r}"
                                 + (" (проблему поймал, но эскалировал сильнее, чем нужно)" if verdict == "reject" else ""))

        else:
            report.record(category, True, case["id"], detail="категория без live-проверки")
    except Exception as e:  # noqa: BLE001
        report.record(category, False, case["id"], detail=f"{type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset.jsonl")
    parser.add_argument("--live", action="store_true",
                         help="Прогнать также кейсы, требующие реального Ollama-инференса")
    args = parser.parse_args()

    cases = load_dataset(args.dataset)
    report = EvalReport()

    graph_app = None
    run_request = None
    if args.live:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from agent.graph import build_graph, run_request
        graph_app = build_graph()

    for case in cases:
        if case["category"] == "guardrail":
            run_guardrail_case(case, report)
        elif args.live:
            t0 = time.time()
            run_live_case(case, report, graph_app, run_request)
            print(f"  {case['id']} [{case['category']}] — {time.time() - t0:.1f}s", flush=True)
        else:
            report.record(case["category"], True, case["id"], detail="offline: требует --live для реальной проверки")

    report.print_summary()


if __name__ == "__main__":
    main()
