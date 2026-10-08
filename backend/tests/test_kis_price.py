import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

import kis
from main import app

OUTPUT = {
    "stck_prpr": "70500", "prdy_vrss": "-500", "prdy_ctrt": "-0.70",
    "acml_vol": "12345678", "stck_oprc": "71000", "stck_hgpr": "71200", "stck_lwpr": "70300",
}


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("KIS_APP_KEY", "k")
    monkeypatch.setenv("KIS_APP_SECRET", "s")
    monkeypatch.setattr(kis, "get_token", lambda: {"access_token": "tok"})


def fake_get(status=200, body=None):
    r = MagicMock(status_code=status)
    r.json.return_value = body if body is not None else {"rt_cd": "0", "output": OUTPUT}
    return patch("kis.requests.get", return_value=r)


client = TestClient(app)


def test_request_headers_and_params():
    with fake_get() as g:
        kis.get_stock_price("005930")
    args, kw = g.call_args
    assert args[0].endswith("/uapi/domestic-stock/v1/quotations/inquire-price")
    h = kw["headers"]
    assert h["authorization"] == "Bearer tok" and h["appkey"] == "k" and h["appsecret"] == "s"
    assert h["tr_id"] == "FHKST01010100"
    assert kw["params"] == {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": "005930"}


def test_endpoint_ok():
    with fake_get():
        r = client.get("/kis/stock/005930/price")
    assert r.status_code == 200
    assert r.json() == {
        "code": "005930", "price": 70500, "change": -500, "change_rate": -0.70,
        "volume": 12345678, "open": 71000, "high": 71200, "low": 70300,
    }


@pytest.mark.parametrize("sym", ["abc", "12345", "1234567", "00593A"])
def test_invalid_symbol_422(sym):
    with fake_get() as g:
        r = client.get(f"/kis/stock/{sym}/price")
    assert r.status_code == 422 and g.call_count == 0


def test_kis_error_502():
    with fake_get(body={"rt_cd": "1", "msg1": "bad"}):
        r = client.get("/kis/stock/005930/price")
    assert r.status_code == 502


def test_malformed_output_502():
    with fake_get(body={"rt_cd": "0", "output": {}}):
        r = client.get("/kis/stock/005930/price")
    assert r.status_code == 502
