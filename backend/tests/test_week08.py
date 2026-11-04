import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

import backtest
import week08 as w8
import week08_ml as ml

SYMS = [f"S{i}" for i in range(8)]


def synth_panel(n=420, seed=1) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    shape = (n, len(SYMS))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, shape), axis=0)), index=idx, columns=SYMS)
    open_ = close.shift(1).fillna(close.iloc[0]) * np.exp(rng.normal(0.001, 0.01, shape))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.02, shape))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.02, shape))
    vol = pd.DataFrame(rng.integers(1_000, 100_000, shape).astype(float), index=idx, columns=SYMS)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": vol}


@pytest.fixture(scope="module")
def panel():
    return synth_panel()


@pytest.fixture(scope="module")
def eng(panel):
    idx = panel["close"].index
    return w8.Engine(panel, start=idx[200], end=idx[-2])


def test_period_constants_match_lecture_rules():
    assert (w8.START, w8.END) == (pd.Timestamp("2020-01-02"), pd.Timestamp("2026-10-21"))
    assert w8.COST == 0.001


def test_extended_factors_have_no_lookahead(panel):
    """cut 일 이후 OHLCV 를 망가뜨려도 cut 전날까지의 팩터(18+개)와 레짐 게이트가 그대로여야 한다."""
    cut = panel["close"].index[300]
    base = w8.ext_factors(panel)
    rng = np.random.default_rng(0)
    broken = {k: v.copy() for k, v in panel.items()}
    for df in broken.values():
        df.loc[df.index >= cut] = rng.uniform(1, 1e6, size=df.loc[df.index >= cut].shape)
    with np.errstate(all="ignore"):
        other = w8.ext_factors(broken)
    for name, df in base.items():
        pd.testing.assert_frame_equal(df.loc[:cut - pd.Timedelta(days=1)], other[name].loc[:cut - pd.Timedelta(days=1)],
                                      check_exact=False, obj=name)


def test_regime_gate_is_cash_during_warmup(eng):
    full = w8.regimes(eng.F)["ma120"]
    assert not full[:100].any()  # 120일선이 아직 없는 초반은 NaN 비교 -> False -> 현금


def test_pick_earns_next_night_return_and_counts_cash_days(eng, panel):
    night = (panel["open"] / panel["close"].shift(1) - 1)
    score = np.full(eng.nxt.shape, np.nan)
    j = SYMS.index("S3")
    score[:, j] = 1.0  # 항상 S3 만 후보
    score[eng.t0 + 5, :] = np.nan  # 이 날은 후보 없음 -> 현금
    r, inv = eng.pick(score, 3)
    assert len(r) == eng.t1 - eng.t0 + 1
    assert not inv[5] and r[5] == 0.0
    t = eng.t0 + 7
    assert r[7] == pytest.approx(night["S3"].iloc[t + 1])  # t 일 신호 -> t+1 일 시가 수익 (shift(1) 와 같은 의미)
    assert inv[7]


def test_sharpe_matches_backtest_metrics(eng):
    rng = np.random.default_rng(3)
    r = rng.normal(0.002, 0.01, 300)
    inv = rng.random(300) > 0.3
    net = pd.Series(r - w8.COST * inv)
    assert w8.sharpe(r, inv) == pytest.approx(backtest.metrics(net)["sharpe"])
    m = w8.fast_metrics(r, inv)
    assert m["invested_days"] == int(inv.sum()) and m["days"] == 300


def test_mask_applies_gate_and_k(eng):
    score = eng.rank("vol_ratio")
    m = eng.mask(score, 3, "ma120")
    assert (m.sum(axis=1) <= 3).all()
    assert not m.iloc[:100].any().any()


def test_train_and_evaluate_are_separated_and_reusable_on_other_period(panel):
    idx = panel["close"].index
    train_eng = w8.Engine(panel, start=idx[200], end=idx[300])
    cube = ml.feature_cube(train_eng)
    y_raw, y_rank = ml.target_matrices(train_eng)
    X, y = ml.training_rows(cube, y_rank, train_eng.t0, train_eng.t1)
    assert np.isfinite(X).all() and len(X) == len(y) > 0
    model = ml.train_model("ridge", X, y)
    bundle = {"model": model, "k": 3, "filter": "none", "regime": "none"}
    coef_before = model[-1].coef_.copy()
    res = ml.evaluate_model(bundle, panel, idx[310], idx[-2])  # 학습 때 쓰지 않은 다른 기간에 그대로 적용
    assert len(res["days"]) == len(res["r"]) and res["days"][0] >= idx[310]
    np.testing.assert_array_equal(coef_before, model[-1].coef_)  # 재학습 없음
    assert res["metrics"]["days"] == len(res["days"])


def test_target_is_next_night_return_not_same_day(eng):
    y_raw, y_rank = ml.target_matrices(eng)
    t = eng.t0 + 10
    j = 2
    expected = eng.panel["open"].iloc[t + 1, j] / eng.panel["close"].iloc[t, j] - 1
    assert y_raw[t, j] == pytest.approx(np.clip(expected, -0.1, 0.1))
    assert -0.5 <= np.nanmin(y_rank) and np.nanmax(y_rank) <= 0.5
