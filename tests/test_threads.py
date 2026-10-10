import pytest
from fastapi.testclient import TestClient

from agent.threads import ThreadStore, clean_title, make_title


@pytest.mark.parametrize("raw,expected", [
    ("Имя кота Мурзум", "Имя кота Мурзум"),
    ("«Расчёт процентов».", "Расчёт процентов"),
    ("Название: Анализ продаж\nвторая строка", "Анализ продаж"),
    ("", "моего кота зовут Мурзум"),                      # пустой ответ -> начало сообщения
    ("x" * 200, "моего кота зовут Мурзум"),               # слишком длинное -> начало сообщения
])
def test_clean_title(raw, expected):
    assert clean_title(raw, "моего кота зовут Мурзум") == expected


def test_make_title_survives_model_error():
    def boom(_):
        raise RuntimeError("ollama down")
    assert make_title(boom, "Посчитай 18% от 4500") == "Посчитай 18% от 4500"


def test_store_crud(tmp_path):
    s = ThreadStore(str(tmp_path / "t.sqlite"))
    s.create("u1", "a", "Первый")
    s.create("u1", "b", "Второй")
    s.create("u2", "c", "Чужой")
    s.touch("a")                                           # «a» стал последним активным
    assert [t["thread_id"] for t in s.list("u1")] == ["a", "b"]
    assert s.rename("b", "  Новое имя  ")["title"] == "Новое имя"
    assert s.rename("b", "   ")["title"] == "Новое имя"    # пустое имя не затирает старое
    assert s.get("nope") is None


@pytest.fixture
def client(monkeypatch):
    import agent.main as main
    calls = []
    monkeypatch.setattr(main, "run_request", lambda app, u, t, m: calls.append(m) or {
        "final_answer": "ок", "route_history": ["answer_directly"], "step_count": 1, "trace": []})
    monkeypatch.setattr(main, "_title_chat", lambda messages: "Имя кота")
    with TestClient(main.app) as c:
        yield c, calls


def test_threads_api(client):
    c, _ = client
    r1 = c.post("/chat", json={"user_id": "web", "message": "моего кота зовут Мурзум"}).json()
    assert r1["title"] == "Имя кота"
    tid = r1["thread_id"]
    r2 = c.post("/chat", json={"user_id": "web", "thread_id": tid, "message": "а собаку Рекс"}).json()
    assert r2["thread_id"] == tid and r2["title"] == "Имя кота"   # название не перегенерируется
    assert [t["thread_id"] for t in c.get("/threads", params={"user_id": "web"}).json()] == [tid]
    assert c.patch(f"/threads/{tid}", json={"user_id": "web", "title": "Питомцы"}).json()["title"] == "Питомцы"
    assert c.patch(f"/threads/{tid}", json={"user_id": "other", "title": "x"}).status_code == 404
    assert c.get(f"/threads/{tid}").json()["title"] == "Питомцы"
    assert c.delete(f"/threads/{tid}", params={"user_id": "other"}).status_code == 404
    assert c.delete(f"/threads/{tid}", params={"user_id": "web"}).json()["deleted"] == tid
    assert c.get("/threads", params={"user_id": "web"}).json() == []


def test_local_memory_forget_thread(tmp_path):
    from memory.memory_manager import LocalJSONMemoryStore
    m = LocalJSONMemoryStore(str(tmp_path / "mem.json"))
    m.add("u", "кот Мурзум", metadata={"thread_id": "A"})
    m.add("u", "живу в Казани", metadata={"thread_id": "B"})
    m.add("x", "чужой факт", metadata={"thread_id": "A"})
    assert m.forget_thread("u", "A") == 1
    assert [i["text"] for i in m._load()] == ["живу в Казани", "чужой факт"]
