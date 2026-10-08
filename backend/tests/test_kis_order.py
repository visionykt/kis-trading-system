import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import requests
from fastapi.testclient import TestClient

import auth
import kis
from main import app

OK = {"rt_cd": "0", "msg_cd": "APBK0013", "msg1": "주문 전송 완료 되었습니다.",
      "output": {"KRX_FWDG_ORD_ORGNO": "91252", "ODNO": "0000117057", "ORD_TMD": "121052"}}


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for k, v in {"KIS_APP_KEY": "k", "KIS_APP_SECRET": "s", "KIS_CANO": "12345678", "KIS_ACNT_PRDT_CD": "01"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("KIS_ENV", raising=False)
    monkeypatch.setenv("KIS_ORDER_ENABLED", "true")
    monkeypatch.setattr(kis, "get_token", lambda: {"access_token": "tok"})


@pytest.fixture
def client():
    app.dependency_overrides[auth.current_user] = lambda: {"username": "t"}
    yield TestClient(app)
    app.dependency_overrides.clear()


def fake_post(body=OK, status=200):
    r = MagicMock(status_code=status)
    r.json.return_value = body
    return patch("kis.requests.post", return_value=r)


def body_of(side="buy", order_type="limit", quantity=10, price=70000, symbol="005930"):
    return {"symbol": symbol, "side": side, "order_type": order_type, "quantity": quantity, "price": price}


@pytest.mark.parametrize("side,mock,tr", [
    ("buy", False, "TTTC0802U"), ("buy", True, "VTTC0802U"),
    ("sell", False, "TTTC0801U"), ("sell", True, "VTTC0801U"),
])
def test_tr_id_branching(monkeypatch, side, mock, tr):
    if mock:
        monkeypatch.setenv("KIS_ENV", "mock")
    with fake_post() as p:
        kis.place_order("005930", side, "limit", 1, 100)
    assert p.call_args.kwargs["headers"]["tr_id"] == tr
    assert ("openapivts" in p.call_args.args[0]) == mock


def test_limit_mapping():
    with fake_post() as p:
        out = kis.place_order("005930", "buy", "limit", 10, 70000)
    assert p.call_args.args[0].endswith("/uapi/domestic-stock/v1/trading/order-cash")
    assert p.call_args.kwargs["json"] == {
        "CANO": "12345678", "ACNT_PRDT_CD": "01", "PDNO": "005930",
        "ORD_DVSN": "00", "ORD_QTY": "10", "ORD_UNPR": "70000",
    }
    assert out["order_no"] == "0000117057" and out["krx_org_no"] == "91252" and out["order_time"] == "121052"


def test_market_mapping():
    with fake_post() as p:
        out = kis.place_order("005930", "sell", "market", 3, 99999)
    j = p.call_args.kwargs["json"]
    assert j["ORD_DVSN"] == "01" and j["ORD_UNPR"] == "0" and j["ORD_QTY"] == "3"
    assert out["price"] is None


def test_rejected_raises_normalized_error():
    bad = {"rt_cd": "1", "msg_cd": "APBK0952", "msg1": "주문가능금액을 초과 했습니다"}
    with fake_post(bad):
        with pytest.raises(kis.KISOrderRejected) as e:
            kis.place_order("005930", "buy", "limit", 1, 100)
    assert e.value.msg_cd == "APBK0952" and isinstance(e.value, kis.KISError)


def test_network_error_not_retried():
    with patch("kis.requests.post", side_effect=requests.Timeout("t")) as p:
        with pytest.raises(kis.KISError) as e:
            kis.place_order("005930", "buy", "limit", 1, 100)
    assert p.call_count == 1 and e.value.status == 504


def test_missing_odno():
    with fake_post({"rt_cd": "0", "output": {}}):
        with pytest.raises(kis.KISError):
            kis.place_order("005930", "buy", "limit", 1, 100)


def test_endpoint_requires_login():
    with fake_post() as p:
        r = TestClient(app).post("/kis/order", json=body_of())
    assert r.status_code in (401, 403) and p.call_count == 0


def test_endpoint_ok(client):
    with fake_post() as p:
        r = client.post("/kis/order", json=body_of())
    assert r.status_code == 200 and r.json()["order_no"] == "0000117057"
    assert p.call_count == 1


def test_market_ignores_price(client):
    with fake_post() as p:
        r = client.post("/kis/order", json=body_of(order_type="market", price=123))
    assert r.status_code == 200 and p.call_args.kwargs["json"]["ORD_UNPR"] == "0"


@pytest.mark.parametrize("bad", [
    body_of(quantity=0), body_of(quantity=-1), body_of(price=0), body_of(price=-5), body_of(price=None),
    body_of(symbol="abc"), body_of(side="hold"), body_of(order_type="stop"),
    {"symbol": "005930", "side": "buy"},
])
def test_validation_422_without_kis_call(client, bad):
    with fake_post() as p:
        r = client.post("/kis/order", json=bad)
    assert r.status_code == 422 and p.call_count == 0


def test_kis_rejection_400(client):
    with fake_post({"rt_cd": "1", "msg_cd": "APBK0918", "msg1": "장종료"}):
        r = client.post("/kis/order", json=body_of())
    assert r.status_code == 400 and r.json()["msg_cd"] == "APBK0918"


def test_kis_http_error_502(client):
    with fake_post({"msg1": "oops"}, status=500):
        r = client.post("/kis/order", json=body_of())
    assert r.status_code == 502


def test_kis_timeout_502(client):
    with patch("kis.requests.post", side_effect=requests.Timeout("t")):
        r = client.post("/kis/order", json=body_of())
    assert r.status_code == 502


@pytest.mark.parametrize("value", [None, "false", "", "0"])
def test_disabled_returns_403_without_kis_call(client, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("KIS_ORDER_ENABLED")
    else:
        monkeypatch.setenv("KIS_ORDER_ENABLED", value)
    with fake_post() as p:
        r = client.post("/kis/order", json=body_of())
    assert r.status_code == 403 and "KIS_ORDER_ENABLED=false" in r.json()["detail"]
    assert p.call_count == 0


def test_disabled_still_requires_login(monkeypatch):
    monkeypatch.setenv("KIS_ORDER_ENABLED", "false")
    assert TestClient(app).post("/kis/order", json=body_of()).status_code in (401, 403)
