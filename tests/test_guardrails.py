import pytest

from agent.guardrails import destructive_pattern, irreversible_reason, is_confirmation
from agent.tools import code_exec, file_ops


@pytest.mark.parametrize("text", [
    "Удали все файлы в моей рабочей директории",
    "удалить папку data",
    "Сотри базу пользователей",
    "очисти workspace",
    "перезапиши файл report.csv",
    "выполни rm -rf /workspace",
    "DROP TABLE users",
    "Отправь письмо начальнику с отчётом",
    "Переведи деньги на карту 5000 руб",
])
def test_irreversible_detected(text):
    assert irreversible_reason(text) is not None


@pytest.mark.parametrize("text", [
    "Посчитай 18% от 4500",
    "удали дубликаты и посчитай среднее",      # работа с данными в памяти, не с файлами
    "очисти данные от выбросов",
    "Что нового у Ollama?",
    "Привет, как дела?",
    "Сравни Ollama, llama.cpp и vLLM",
])
def test_regular_requests_not_flagged(text):
    assert irreversible_reason(text) is None


@pytest.mark.parametrize("text,ok", [
    ("да, подтверждаю", True), ("Да", True), ("подтверждаю", True), ("yes", True),
    ("да, но сначала покажи список файлов", False), ("нет", False), ("Посчитай 2+2", False),
])
def test_confirmation(text, ok):
    assert is_confirmation(text) is ok


def test_network_code_blocked():
    res = code_exec.code_exec("import socket; socket.socket().connect(('example.com', 80))")
    assert res.exit_code == -1 and "заблокирован" in res.stderr


def test_destructive_code_blocked_without_confirmation():
    res = code_exec.code_exec("import shutil; shutil.rmtree('/workspace/data')")
    assert res.exit_code == -1 and "разрушающая" in res.stderr
    assert destructive_pattern("import os\nos.remove('a.txt')") == "os.remove"
    assert destructive_pattern("print(sum(range(10)))") is None


def test_path_traversal_blocked():
    with pytest.raises(file_ops.PathEscapeError):
        file_ops.file_read("../../etc/passwd")
    with pytest.raises(file_ops.PathEscapeError):
        file_ops.file_write("../outside.txt", "x")


def test_file_write_and_exists():
    assert not file_ops.exists("notes/a.txt")
    file_ops.file_write("notes/a.txt", "hello")
    assert file_ops.exists("notes/a.txt")
    assert file_ops.file_read("notes/a.txt") == "hello"


@pytest.mark.parametrize("text,ok", [
    ("Я живу в Москве", True),
    ("На самом деле я переехал в Санкт-Петербург", True),
    ("Меня зовут Тимур, я работаю аналитиком", True),
    ("У меня защита диплома 25 декабря", True),
    ("Запомни: отчёты присылать в PDF", True),
    ("На основе только этого текста: 'Кошки — млекопитающие' — сколько лап у кошки?", False),
    ("Посчитай, сколько будет 18% от 4500", False),
    ("Проанализируй мои продажи: 120, 135, 90", False),
    ("Как меня зовут и где я учусь?", False),
    ("Привет", False),
    ("моего кота зовут мурзум", True),
    ("мою собаку зовут Рекс", True),
    ("у моей мамы день рождения 5 мая", True),
    ("нам дали дедлайн 1 ноября", True),
    ("как зовут моего кота?", False),
])
def test_memory_gate(text, ok):
    from memory.memory_manager import is_memory_worthy
    assert is_memory_worthy(text) is ok


def test_memory_skip_reason_is_specific():
    from memory.memory_manager import memory_skip_reason
    assert memory_skip_reason("как зовут моего кота?") == "вопрос"
    assert memory_skip_reason("Посчитай 18% от 4500") == "поручение агенту"
    assert memory_skip_reason("Привет") == "нет утверждения о себе"
    assert memory_skip_reason("моего кота зовут Мурзум") is None


@pytest.mark.parametrize("text", [
    "что самого крупного произошло в мире за последнюю неделю можешь рассказать?",
    "Что нового вышло у Ollama за последний месяц?",
    "Какая последняя версия Python?",
    "какая сегодня погода в Казани",
    "курс доллара к рублю",
    "какие новости в мире ИИ",
    "Что сейчас происходит на рынке видеокарт?",
])
def test_freshness_detected(text):
    from agent.guardrails import freshness_reason
    assert freshness_reason(text) is not None


@pytest.mark.parametrize("text", [
    "Посчитай 18% от 4500", "Привет, как дела?", "Как меня зовут?",
    "Объясни, что такое RAG", "Сравни Ollama и vLLM для 8GB VRAM",
])
def test_freshness_not_triggered(text):
    from agent.guardrails import freshness_reason
    assert freshness_reason(text) is None
