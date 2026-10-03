import pytest

from agent import skills as sk

SKILLS = sk.load_skills()


def test_all_skills_parse():
    names = {s.name for s in SKILLS}
    assert names == {"code-review", "data-analysis", "report-writing", "web-research"}
    for s in SKILLS:
        assert s.description, s.name
        assert s.agents, f"{s.name}: не указано, каким узлам доступен skill"
        assert len(s.body) > 200, f"{s.name}: тело skill подозрительно короткое"


@pytest.mark.parametrize("agent,text,expected", [
    ("executor", "Посчитай 18% от 4500", ["code-review"]),
    ("executor", "Проанализируй sales.csv и найди аномалии", ["code-review", "data-analysis"]),
    ("researcher", "Что нового у Ollama за месяц?", ["web-research"]),
    ("finalize", "Привет", []),
    ("finalize", "Сделай отчёт по результатам", ["report-writing"]),
    ("critic", "Проанализируй sales.csv", []),  # критику skills не назначены
])
def test_selection(agent, text, expected):
    assert [s.name for s in sk.select_skills(SKILLS, agent, text)] == expected


def test_report_writing_after_multistep_task():
    picked = sk.select_skills(SKILLS, "finalize", "какой итог?", delegations=0)
    assert [s.name for s in picked] == ["report-writing"]  # триггер «итог»
    picked = sk.select_skills(SKILLS, "finalize", "ну что там", delegations=2)
    assert [s.name for s in picked] == ["report-writing"]  # >= 2 делегирований


def test_budget_and_render():
    picked = sk.select_skills(SKILLS, "executor", "проанализируй таблицу csv, метрики, статистику")
    assert len(picked) <= sk.MAX_SKILLS_PER_CALL
    block = sk.render(picked)
    assert "# Активные skills" in block and "## Skill: code-review" in block
    assert sk.render([]) == ""
