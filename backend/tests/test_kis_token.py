import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import kis


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(kis, "TOKEN_CACHE", tmp_path / "kis_token.json")
    monkeypatch.setattr(kis, "_memory", None)
    monkeypatch.setenv("KIS_APP_KEY", "k")
    monkeypatch.setenv("KIS_APP_SECRET", "s")


def fake_post(expires_in=86400):
    r = MagicMock(status_code=200)
    r.json.return_value = {"access_token": "tok", "expires_in": expires_in}
    return patch("kis.requests.post", return_value=r)


def test_reuses_cache_without_new_request():
    with fake_post() as p:
        assert kis.get_token()["access_token"] == "tok"
        kis.get_token()
        kis.get_token()
        assert p.call_count == 1


def test_reuses_file_cache_after_restart(monkeypatch):
    with fake_post() as p:
        kis.get_token()
        monkeypatch.setattr(kis, "_memory", None)  # 프로세스 재시작 시뮬레이션
        kis.get_token()
        assert p.call_count == 1


def test_reissues_within_5min_of_expiry():
    with fake_post(expires_in=4 * 60) as p:  # 만료까지 4분 -> 마진 안
        kis.get_token()
        kis.get_token()
        assert p.call_count == 2


def test_keeps_token_just_outside_margin():
    with fake_post(expires_in=6 * 60) as p:
        kis.get_token()
        kis.get_token()
        assert p.call_count == 1


def test_corrupted_cache_reissues():
    kis.TOKEN_CACHE.write_text('{"foo": 1}')
    with fake_post() as p:
        kis.get_token()
        assert p.call_count == 1


def test_status_no_cache():
    with fake_post() as p:
        st = kis.token_status()
        assert st["cached"] is False and p.call_count == 0


def test_status_after_issue():
    with fake_post():
        kis.get_token()
    st = kis.token_status()
    assert st["cached"] and st["valid"]
    assert 86400 - 5 <= st["remaining_seconds"] <= 86400
    assert "access_token" not in st
