import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

import week08 as w8
import week10_dataset as ds

SYMS = [f"S{i}" for i in range(12)]
START, END = pd.Timestamp("2022-06-01"), pd.Timestamp("2023-12-29")


def synth_panel(n=520, seed=3) -> dict:
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
def data(panel):
    return ds.build(panel, start=START, end=END)


def test_features_are_cross_sectional_rank_in_range(data):
    c = data.cube
    assert np.nanmin(c) >= -0.5 and np.nanmax(c) <= 0.5
    t = data.eng.t0 + 100
    # 날짜별 순위: 같은 날 종목 평균이 ~0 (-0.5~+0.5 로 중심화)
    assert abs(np.nanmean(c[t, :, 0])) < 0.1


def test_no_market_wide_features_in_input():
    assert not any(f.startswith("mkt_") for f in ds.FEATURES)


def test_target_is_market_neutral_and_aligned(data, panel):
    eng = data.eng
    t = eng.t0 + 50
    o, c = panel["open"].to_numpy(), panel["close"].to_numpy()
    raw = o[t + 1] / c[t] - 1  # t 일 종가 매수 -> t+1 일 시가 매도
    assert np.allclose(data.y_excess[t], raw - raw.mean())
    assert abs(np.nanmean(data.y_excess[t])) < 1e-12


def test_features_do_not_see_the_future(panel, data):
    """t 일 이후 데이터를 전부 잘라 다시 만들어도 t 일 이하의 입력이 똑같아야 한다 (미래 참조/스케일러 누수 없음)."""
    cut = 380
    short = {k: v.iloc[:cut + 1] for k, v in panel.items()}
    d2 = ds.build(short, start=START, end=short["close"].index[-1])
    n = min(len(d2.cube), cut + 1)
    a, b = data.cube[:n - 1], d2.cube[:n - 1]
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b), atol=1e-6)


def test_sequence_window_layout(data):
    s = data.samples("2023-03-01", "2023-06-30", "3D")
    assert s.x_batch(np.arange(1)).shape[1:] == (ds.SEQ_LEN, len(ds.FEATURES))
    for k in (0, len(s) // 2, len(s) - 1):
        t, i = s.t[k], s.i[k]
        x = s.x_batch(np.array([k]))[0]
        assert np.array_equal(x, data.cube[t - ds.SEQ_LEN + 1:t + 1, i])  # 마지막 행 = 당일 t
        assert np.isfinite(x).all() and np.isfinite(s.y[k])
    # 2D 는 3D 의 마지막 시점과 같다
    s2 = data.samples("2023-03-01", "2023-06-30", "2D")
    assert s2.x_batch(np.arange(1)).shape[1:] == (len(ds.FEATURES),)


def test_periods_do_not_overlap_and_are_ordered(data):
    parts = {k: data.samples(a, b, "2D") for k, (a, b) in
             {"train": ("2022-06-01", "2022-12-31"), "val": ("2023-01-01", "2023-06-30"), "test": ("2023-07-01", "2023-12-29")}.items()}
    day = lambda s: s.t - data.eng.t0
    assert day(parts["train"]).max() < day(parts["val"]).min() and day(parts["val"]).max() < day(parts["test"]).min()
    # 학습 샘플의 실현일(t+1)이 검증 시작일 이전
    assert data.days[day(parts["train"]).max()] < pd.Timestamp("2023-01-01")


def test_walk_forward_uses_only_past():
    d = ds.Week10Data.__new__(ds.Week10Data)
    d.eng = type("E", (), {"days": pd.bdate_range("2020-01-02", "2026-10-21")})()
    plan = ds.Week10Data.walk_forward_plan(d)
    assert [p["year"] for p in plan] == [2022, 2023, 2024, 2025, 2026]
    for p in plan:
        assert pd.Timestamp(p["train"][1]) < pd.Timestamp(p["val"][0]) <= pd.Timestamp(p["val"][1]) < pd.Timestamp(p["test"][0])
    assert plan[0]["val"] == ("2021-07-01", "2021-12-31") and plan[-1]["test"][1] == "2026-10-21"


def test_backtest_matches_week08_engine(data):
    score = data.cube[:, :, ds.FEATURES.index("night_sharpe_20")].copy()
    full = data.backtest(score)
    r, inv = data.eng.pick(score, ds.TOP_K)
    assert np.allclose(full["r"], r) and full["metrics"]["sharpe"] == pytest.approx(w8.fast_metrics(r, inv)["sharpe"])
    half = data.backtest(score, "2023-01-01", "2023-06-30")
    assert half["days"] == len(half["r"]) < full["days"]


def test_torch_loader_and_tensors(data):
    s = data.samples("2023-03-01", "2023-04-30", "3D")
    X, y = s.tensors()
    assert X.shape == (len(s), ds.SEQ_LEN, len(ds.FEATURES)) and y.shape == (len(s),)
    xb, yb = next(iter(s.loader(batch_size=64, shuffle=True)))
    assert xb.shape == (64, ds.SEQ_LEN, len(ds.FEATURES)) and xb.dtype.is_floating_point
