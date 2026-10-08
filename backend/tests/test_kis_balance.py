import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

import auth
import kis
from main import app

RAW = {
    "rt_cd": "0",
    "output1": [
        {"pdno": "005930", "prdt_name": "삼성전자", "hldg_qty": "10", "pchs_avg_pric": "70000.5000",
         "prpr": "75000", "evlu_pfls_amt": "49995", "evlu_pfls_rt": "7.14"},
        {"pdno": "000660", "prdt_name": "SK하이닉스", "hldg_qty": "0", "pchs_avg_pric": "0",
         "prpr": "1", "evlu_pfls_amt": "0", "evlu_pfls_rt": "0"},
    ],
    "output2": [
        {"dnca_tot_amt": "1000000", "prvs_rcdl_excc_amt": "900000", "scts_evlu_amt": "750000",
         "tot_evlu_amt": "1750000", "pchs_amt_smtl_amt": "700005", "evlu_pfls_smtl_amt": "49995"}
    ],
}


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for k, v in {"KIS_APP_KEY": "k", "KIS_APP_SECRET": "s", "KIS_CANO": "12345678", "KIS_ACNT_PRDT_CD": "01"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(kis, "get_token", lambda: {"access_token": "tok"})


def fake_get(body=RAW, status=200):
    r = MagicMock(status_code=status)
    r.json.return_value = body
    return patch("kis.requests.get", return_value=r)


def test_request_params_and_tr_id():
    with fake_get() as g:
        kis.get_balance()
    args, kw = g.call_args
    assert args[0].endswith("/uapi/domestic-stock/v1/trading/inquire-balance")
    assert kw["headers"]["tr_id"] == "TTTC8434R"
    p = kw["params"]
    assert p["CANO"] == "12345678" and p["ACNT_PRDT_CD"] == "01"
    for k in ("AFHR_FLPR_YN", "OFL_YN", "INQR_DVSN", "UNPR_DVSN", "FUND_STTL_ICLD_YN",
              "FNCG_AMT_AUTO_RDPT_YN", "PRCS_DVSN", "CTX_AREA_FK100", "CTX_AREA_NK100"):
        assert k in p


def test_mock_mode_uses_vts(monkeypatch):
    monkeypatch.setenv("KIS_ENV", "mock")
    with fake_get() as g:
        kis.get_balance()
    args, kw = g.call_args
    assert "openapivts" in args[0] and kw["headers"]["tr_id"] == "VTTC8434R"


def test_normalization():
    with fake_get():
        out = kis.get_balance()
    assert out["summary"] == {
        "total_evaluation": 1750000, "cash": 1000000, "orderable_cash": 900000,
        "stock_evaluation": 750000, "purchase_amount": 700005, "profit_loss": 49995, "profit_rate": 7.14,
    }
    assert out["holdings"] == [  # 수량 0 종목은 제외
        {"code": "005930", "name": "삼성전자", "quantity": 10, "avg_price": 70000.5, "price": 75000,
         "profit_loss": 49995, "profit_rate": 7.14}
    ]


def test_empty_account():
    empty = {"rt_cd": "0", "output1": [], "output2": [{"dnca_tot_amt": "0", "tot_evlu_amt": "0",
             "pchs_amt_smtl_amt": "0", "evlu_pfls_smtl_amt": "0"}]}
    with fake_get(empty):
        out = kis.get_balance()
    assert out["holdings"] == [] and out["summary"]["profit_rate"] == 0.0


def test_kis_error():
    with fake_get({"rt_cd": "1", "msg1": "bad"}):
        with pytest.raises(kis.KISError):
            kis.get_balance()


def test_malformed_holding():
    with fake_get({"rt_cd": "0", "output1": [{"pdno": "1", "hldg_qty": "5"}], "output2": [{}]}):
        with pytest.raises(kis.KISError):
            kis.get_balance()


def test_endpoint_requires_login():
    assert TestClient(app).get("/kis/balance").status_code in (401, 403)


def test_endpoint_ok_and_502():
    app.dependency_overrides[auth.current_user] = lambda: {"username": "t"}
    try:
        c = TestClient(app)
        with fake_get():
            r = c.get("/kis/balance")
        assert r.status_code == 200 and r.json()["summary"]["total_evaluation"] == 1750000
        with fake_get({"rt_cd": "1", "msg1": "bad"}):
            assert c.get("/kis/balance").status_code == 502
    finally:
        app.dependency_overrides.clear()
