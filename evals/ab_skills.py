"""A/B: влияют ли skills на качество под-агентов.
    python evals/ab_skills.py --repeats 3
Каждая задача гоняется k раз с подгрузкой SKILL.md и k раз без неё (SKILLS=[]),
считаются успех (ожидаемая подстрока в ответе), число шагов графа и время.
Нужны запущенные Ollama, Qdrant и Docker (как для run_evals.py --live)."""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent.graph as g  # noqa: E402

TASKS = [
    {"id": "sales", "input": "Проанализируй продажи по месяцам: 120, 135, 90, 160, 155 — посчитай среднее и найди аномалию",
     "expect": "132"},
    {"id": "growth", "input": "Выручка по кварталам: 400, 420, 380, 510. Посчитай рост последнего квартала к предыдущему в процентах",
     "expect": "34"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    app = g.build_graph()
    all_skills = g.SKILLS
    rows = []
    for task in TASKS:
        for mode in ("skills", "no_skills"):
            g.SKILLS = all_skills if mode == "skills" else []
            for i in range(args.repeats):
                t0 = time.time()
                res = g.run_request(app, f"ab_{uuid.uuid4().hex[:6]}", uuid.uuid4().hex[:8], task["input"])
                ok = task["expect"] in (res.get("final_answer") or "")
                rows.append((task["id"], mode, ok, res.get("step_count"), round(time.time() - t0, 1)))
                print(f"{task['id']:7s} {mode:9s} #{i + 1} {'✓' if ok else '✗'} steps={res.get('step_count')} "
                      f"t={rows[-1][4]}s skills={res.get('skills_used')} | {(res.get('final_answer') or '')[:90]!r}",
                      flush=True)
    print("\n=== Итог ===")
    for task in TASKS:
        for mode in ("skills", "no_skills"):
            r = [x for x in rows if x[0] == task["id"] and x[1] == mode]
            print(f"{task['id']:7s} {mode:9s} успех {sum(x[2] for x in r)}/{len(r)}, "
                  f"шагов в среднем {sum(x[3] for x in r) / len(r):.1f}, время {sum(x[4] for x in r) / len(r):.0f} с")


if __name__ == "__main__":
    main()
