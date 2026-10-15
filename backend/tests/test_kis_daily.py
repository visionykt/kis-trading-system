import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import kis


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("KIS_APP_KEY", "k")
    monkeypatch.setenv("KIS_APP_SECRET", "s")
    monkeypatch.setattr(kis, "get_token", lambda: {"access_token": "tok"})
    monkeypatch.setattr(kis.time, "sleep", lambda s: None)


def row(d, o=100, h=110, l=90, c=105, v=1000):
    return {"stck_bsop_date": d, "stck_oprc": str(o), "stck_hgpr": str(h), "stck_lwpr": str(l),
            "stck_clpr": str(c), "acml_vol": str(v)}


def resp(rows):
    r = MagicMock(status_code=200)
    r.json.return_value = {"rt_cd": "0", "output1": {}, "output2": rows}
    return r


def test_request_uses_tr_and_adjusted_price():
    with patch("kis.requests.get", return_value=resp([row("20261013"), {}])) as g:
        out = kis.inquire_daily_chart("005930", "20261001", "20261013")
    args, kw = g.call_args
    assert args[0].endswith("/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice")
    assert kw["headers"]["tr_id"] == "FHKST03010100"
    assert kw["params"] == {
        "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": "005930", "FID_INPUT_DATE_1": "20261001",
        "FID_INPUT_DATE_2": "20261013", "FID_PERIOD_DIV_CODE": "D", "FID_ORG_ADJ_PRC": "0",
    }
    assert len(out) == 1  # 빈 dict 제거


def test_paginates_and_sorts_ascending():
    # 1페이지: 100건(최신->과거), 2페이지: 나머지 2건
    days = [f"2026{m:02d}{d:02d}" for m in range(1, 13) for d in range(1, 29)][:102]
    page1 = [row(d) for d in reversed(days[2:])]
    page2 = [row(d) for d in reversed(days[:2])]
    with patch("kis.requests.get", side_effect=[resp(page1), resp(page2)]) as g:
        out = kis.get_daily_candles("005930", "20260101", "20261231")
    assert g.call_count == 2
    # 2번째 호출 끝 날짜 = 1페이지에서 가장 과거 날짜 - 1일
    assert g.call_args_list[1].kwargs["params"]["FID_INPUT_DATE_2"] == "20260102"
    assert [r["ts"] for r in out] == sorted(r["ts"] for r in out)
    assert len(out) == 102 and out[0]["ts"] == date(2026, 1, 1)
    assert out[0] == {"symbol": "005930", "ts": date(2026, 1, 1), "open": 100.0, "high": 110.0,
                      "low": 90.0, "close": 105.0, "volume": 1000}


def test_malformed_row_raises():
    with patch("kis.requests.get", return_value=resp([{"stck_bsop_date": "20261013", "stck_oprc": "x"}])):
        with pytest.raises(kis.KISError):
            kis.get_daily_candles("005930", "20261001", "20261013")
