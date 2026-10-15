import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

import portfolio as pf

IDX = pd.to_datetime(["2026-10-06", "2026-10-07", "2026-10-08", "2026-10-12"])


def make_panel(n_days=60, n_sym=12, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2026-01-01", periods=n_days)
    cols = [f"{i:06d}" for i in range(n_sym)]
    close = pd.DataFrame(10000 * np.exp(np.cumsum(rng.normal(0, 0.02, (n_days, n_sym)), axis=0)), idx, cols)
    open_ = close.shift(1).fillna(close) * np.exp(rng.normal(0, 0.01, (n_days, n_sym)))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    vol = pd.DataFrame(rng.integers(1_000, 100_000, (n_days, n_sym)), idx, cols).astype(float)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": vol}


def test_returns_and_missing_prev_day():
    p = {"open": pd.DataFrame({"a": [100, 110, 120, 130.0], "b": [np.nan, 50, 55, 60.0]}, IDX),
         "close": pd.DataFrame({"a": [105, 108, 125, 128.0], "b": [np.nan, 52, 54, 61.0]}, IDX)}
    r = pf.returns(p)
    assert r["night"].loc["2026-10-07", "a"] == pytest.approx(110 / 105 - 1)
    assert np.isnan(r["night"].loc["2026-10-07", "b"])  # 상장 첫날은 전일 종가 없음
    assert r["night"].loc["2026-10-08", "b"] == pytest.approx(55 / 52 - 1)


def test_equal_weight_uses_available_symbols():
    night = pd.DataFrame({"a": [np.nan, 0.02, 0.04], "b": [np.nan, np.nan, 0.0]}, IDX[:3])
    ew = pf.equal_weight({"night": night, "day": night * 0 + 0.01, "hold": night})
    assert ew["night"].tolist() == pytest.approx([0.02, 0.02])
    assert ew["n"].tolist() == [1, 2]


def test_signal_is_previous_day_top_k():
    score = pd.DataFrame({"a": [3, 1, 2.0], "b": [2, 3, 1.0], "c": [1, 2, np.nan]}, IDX[:3])
    sig = pf.signal(score, k=1)
    assert sig.iloc[0].sum() == 0  # 첫날은 전날 점수가 없으니 무포지션
    assert sig.loc["2026-10-07"].to_dict() == {"a": True, "b": False, "c": False}  # 10-06 점수 1위
    assert sig.loc["2026-10-08"].to_dict() == {"a": False, "b": True, "c": False}


def test_portfolio_returns_cash_when_no_pick():
    night = pd.DataFrame({"a": [0.01, 0.02, -0.01], "b": [0.03, 0.04, np.nan]}, IDX[:3])
    sig = pd.DataFrame({"a": [False, True, False], "b": [False, True, True]}, IDX[:3])
    r, inv = pf.portfolio_returns(sig, night)
    assert r.tolist() == pytest.approx([0.0, 0.03, 0.0])  # 3일차: b 거래정지 -> 현금
    assert inv.tolist() == [False, True, False]


def test_cost_only_on_invested_days():
    r = pd.Series([0.0, 0.01, 0.01])
    rows = pf.cost_table(r, pd.Series([False, True, True]), costs=[0.0, 0.002])
    assert rows[1]["cumulative_return"] == pytest.approx(1.008 ** 2 - 1)


@pytest.mark.parametrize("key", list(pf.STRATEGIES))
def test_no_lookahead(key):
    p = make_panel()
    fn = pf.STRATEGIES[key][1]
    for cut in p["open"].index[[25, 40, 59]]:
        assert pf.lookahead_check(p, fn, cut)


def test_lookahead_check_catches_unshifted_signal(monkeypatch):
    p = make_panel()
    monkeypatch.setattr(pf, "signal", lambda score, k=pf.TOP_K: pf.top_k_mask(score, k))  # shift 누락 버그
    assert not pf.lookahead_check(p, pf.score_momentum, p["open"].index[40])


def test_run_top5_aligned_period():
    res = pf.run_top5(make_panel(n_days=80))
    assert len(res["curves"]) == res["period"]["days"]
    assert [b["key"] for b in res["board"]][0] == "KOSPI100_EW"
    for b in res["board"]:
        assert [n["cost"] for n in b["net"]] == pf.COSTS
        assert b["net"][-1]["cumulative_return"] <= b["net"][0]["cumulative_return"]
