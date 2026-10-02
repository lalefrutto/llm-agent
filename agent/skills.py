"""
skills.py — загрузчик skills (skills/*/SKILL.md) в контекст под-агентов.

Skill — процедурное знание для класса задач (см. skills/README.md). Здесь
он превращается из документа в часть рантайма: перед вызовом под-агента
выбираются подходящие SKILL.md, и их тело дописывается в системный промпт.

Выбор детерминированный, без лишнего LLM-вызова (на 8 GB VRAM каждый
вызов — секунды): по полям frontmatter
  agents:          каким узлам графа skill вообще доступен;
  always: true     включать всегда для этих узлов (code-review для executor);
  triggers: [...]  подстроки (основы слов) в запросе/подзадаче;
  min_delegations: N  включать в finalize, если делегирований было >= N.
description остаётся человекочитаемым описанием «когда применять».
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
MAX_SKILLS_PER_CALL = 2  # бюджет контекста: skill ~400-700 токенов


@dataclass
class Skill:
    name: str
    description: str
    body: str
    agents: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)
    always: bool = False
    min_delegations: int | None = None


def parse_skill(text: str) -> Skill:
    if not text.startswith("---"):
        raise ValueError("SKILL.md должен начинаться с YAML frontmatter")
    _, front, body = text.split("---", 2)
    meta = yaml.safe_load(front) or {}
    return Skill(
        name=meta["name"],
        description=meta.get("description", ""),
        body=body.strip(),
        agents=list(meta.get("agents") or []),
        triggers=[str(t).lower() for t in (meta.get("triggers") or [])],
        always=bool(meta.get("always", False)),
        min_delegations=meta.get("min_delegations"),
    )


def load_skills(skills_dir: Path = SKILLS_DIR) -> list[Skill]:
    return [parse_skill(p.read_text(encoding="utf-8")) for p in sorted(skills_dir.glob("*/SKILL.md"))]


def select_skills(skills: list[Skill], agent: str, text: str, delegations: int = 0) -> list[Skill]:
    """Skills для узла `agent` по тексту запроса. Сначала always-skills,
    затем по числу совпавших триггеров; не больше MAX_SKILLS_PER_CALL."""
    lowered = text.lower()
    scored: list[tuple[int, Skill]] = []
    for s in skills:
        if agent not in s.agents:
            continue
        hits = sum(1 for t in s.triggers if t in lowered)
        if s.min_delegations is not None and delegations >= s.min_delegations:
            hits += 1
        if s.always:
            scored.append((1000, s))
        elif hits:
            scored.append((hits, s))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:MAX_SKILLS_PER_CALL]]


def render(selected: list[Skill]) -> str:
    """Блок для системного промпта под-агента."""
    if not selected:
        return ""
    parts = [f"## Skill: {s.name}\n{s.body}" for s in selected]
    return ("\n\n# Активные skills\nСледуй процедурам ниже — они описывают, как хорошо "
            "выполнять этот тип задачи.\n\n" + "\n\n".join(parts))
