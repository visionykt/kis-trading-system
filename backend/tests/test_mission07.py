import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

import factors
import mission07 as m

SYMS = ["A", "B", "C", "D", "E"]


def synth_panel(n=160, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, (n, len(SYMS))), axis=0)), index=idx, columns=SYMS)
    open_ = close.shift(1).fillna(close.iloc[0]) * np.exp(rng.normal(0.001, 0.01, (n, len(SYMS))))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.02, (n, len(SYMS))))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.02, (n, len(SYMS))))
    vol = pd.DataFrame(rng.integers(1_000, 100_000, (n, len(SYMS))).astype(float), index=idx, columns=SYMS)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": vol}


def test_top_n_is_variable_and_skips_nan():
    score = pd.DataFrame({"A": [3.0, 3.0], "B": [2.0, np.nan], "C": [1.0, np.nan]}, index=[0, 1])
    sel = m.top_n(score, 2)
    assert sel.iloc[0].tolist() == [True, True, False]
    assert sel.iloc[1].tolist() == [True, False, False]  # 후보가 1개뿐이면 1종목만 (나머지는 현금)


def test_run_mask_uses_previous_day_signal_and_cash_when_empty():
    idx = pd.bdate_range("2024-01-01", periods=4)
    night = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04], "B": [0.0, -0.02, 0.01, 0.0]}, index=idx)
    mask = pd.DataFrame({"A": [True, False, False, False], "B": [True, False, False, True]}, index=idx)
    r, inv, sig = m.run_mask(mask, night)
    assert not sig.iloc[0].any()  # 첫날 신호 없음
    assert r.iloc[0] == 0 and not inv.iloc[0]
    assert r.iloc[1] == pytest.approx((0.02 + -0.02) / 2) and inv.iloc[1]  # t=0 에 고른 A,B 의 t=1 밤 수익, 1/2 씩
    assert r.iloc[2] == 0 and not inv.iloc[2]  # t=1 에 아무것도 안 골랐으면 현금
    assert r.iloc[3] == 0 and not inv.iloc[3]  # t=2 도 마찬가지
    # t=3 의 선택(B)은 t=4 에야 쓰이므로 이 구간에는 반영되지 않는다


def test_score_metrics_cost_only_on_invested_days_and_calmar():
    r = pd.Series([0.02, 0.0, -0.01, 0.03])
    inv = pd.Series([True, False, True, True])
    s = m.score_metrics(r, inv, 0.002)
    net = pd.Series([0.018, 0.0, -0.012, 0.028])
    assert s["cumulative_return"] == pytest.approx((1 + net).prod() - 1)
    assert s["invested_days"] == 3
    assert s["calmar"] == pytest.approx(s["cumulative_return"] / abs(s["mdd"]))
    assert s["win_rate_invested"] == pytest.approx(2 / 3)  # 보유일 중 승률은 현금일을 빼고 계산
    assert s["win_rate"] == pytest.approx(2 / 4)
    assert m.score_metrics(r, inv, 0.0)["cumulative_return"] > s["cumulative_return"]


def test_strategy_rules_match_their_description():
    p = synth_panel()
    F = factors.compute_factors(p)
    m2 = m.pick_yang_sharpe(p, F)
    assert m2.any().any() and not (m2 & ~(p["close"] > p["open"])).any().any()  # 선택된 건 전부 양봉
    assert m2.sum(axis=1).max() <= 3
    m1 = m.pick_breakout_edge(p, F)
    assert m1.any().any() and not (m1 & ~((F["ma20_gap"] > 0) & (F["vol_ratio"] > 1))).any().any()
    m3 = m.pick_regime_cash(p, F)
    assert (m3 & ~m2).sum().sum() == 0  # M3 는 M2 의 부분집합
    bear = ~m.market_regime(p, 120).reindex(m3.index).fillna(False)
    assert not m3.loc[bear].any().any()  # 하락 레짐에서는 전액 현금
    h = m.pick_lecture_hypothesis(p, F)
    assert not (h & ~(F["upper_shadow"] < 0.15)).any().any()


@pytest.mark.parametrize("fn", [m.pick_yang_sharpe, m.pick_breakout_edge, m.pick_regime_cash, m.pick_lecture_hypothesis,
                                m.pick_sample("C_volume_surge"), m.pick_sample("D_rolling_edge")])
def test_no_lookahead_for_every_strategy(fn):
    p = synth_panel(n=200)
    assert m.lookahead_check(p, fn, p["open"].index[150])


def test_lookahead_check_catches_a_cheating_strategy():
    """오늘 종가 대비 내일 시가를 보는(미래 참조) 전략은 검사에 걸려야 한다."""
    p = synth_panel(n=200)
    cheat = lambda panel, F: m.top_n((panel["open"].shift(-1) / panel["close"] - 1), 2)
    assert not m.lookahead_check(p, cheat, p["open"].index[150])


def test_regime_uses_only_past_prices():
    p = synth_panel(n=200)
    base = m.market_regime(p, 60)
    cut = p["close"].index[120]
    broken = {k: v.copy() for k, v in p.items()}
    broken["close"].loc[broken["close"].index > cut] *= 0.1
    pd.testing.assert_series_equal(base.loc[:cut], m.market_regime(broken, 60).loc[:cut])


# ---------------------------------------------------------------- API 페이로드 / 날짜별 선정 내역

@pytest.fixture
def res(monkeypatch):
    import universe

    panel = synth_panel(n=320, seed=3)
    monkeypatch.setattr(universe, "load_panel", lambda start=None, end=None, **kw: panel)
    monkeypatch.setattr(universe, "list_symbols", lambda *a, **k: [{"symbol": s, "name": f"종목{s}", "sector": "x"} for s in SYMS])
    return m.build(split="")


def test_api_payload_is_json_safe_and_complete(res):
    import json

    d = m.api_payload(res)
    json.dumps(d, allow_nan=False)  # NaN/inf 가 남아 있으면 ValueError
    assert set(d["strategies"]) >= {"KOSPI100_EW", *m.STRATEGIES}
    assert {k for k, v in d["strategies"].items() if v["group"] == "mine"} == set(m.MINE)
    assert d["costs"] == [0.0, 0.0005, 0.001, 0.002]
    row = d["curve"][0]
    assert "ts" in row and all(f"inv_{k}" in row for k in m.STRATEGIES) and "KOSPI100_EW" in row
    for b in d["board"]:
        assert [n["cost"] for n in b["net"]] == m.COSTS
        assert b["criteria"]


def test_picks_on_snaps_to_previous_trading_day_and_reports_next_night(res):
    idx = res["night"].index
    t = idx[idx.get_loc(pd.Timestamp(res["period"]["start"])) + 5]
    sat = t + pd.Timedelta(days=1) if t.weekday() == 4 else t  # 금요일이면 토요일로 조회
    d = m.picks_on(res, sat)
    assert d["date"] == t.date().isoformat()
    nxt = res["night"].shift(-1)
    for key, s in d["strategies"].items():
        picked = {p["symbol"] for p in s["picks"]}
        assert picked == set(res["masks"][key].columns[res["masks"][key].loc[t]])
        for p in s["picks"]:
            assert p["weight"] == pytest.approx(1 / len(s["picks"]))
            assert p["next_night_return"] == pytest.approx(nxt.at[t, p["symbol"]])  # 신호일 다음 거래일의 밤 수익률
            assert set(p["factors"]) == set(m.PICK_COLS)


def test_picks_on_last_day_is_pending_and_early_date_rejected(res):
    last = res["night"].index[-1]
    d = m.picks_on(res, last)
    assert d["next_date"] is None
    assert all(p["next_night_return"] is None for s in d["strategies"].values() for p in s["picks"])
    with pytest.raises(ValueError):
        m.picks_on(res, pd.Timestamp(res["period"]["start"]) - pd.Timedelta(days=30))
