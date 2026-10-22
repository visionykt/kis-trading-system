import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

import factors
import portfolio as pf

# 강의 슬라이드 3 의 005930 예시 5행
SLIDE = pd.DataFrame(
    {
        "open": [59200, 61100, 62400, 61500, 60900],
        "high": [60800, 62000, 62600, 61900, 63000],
        "low": [58900, 60700, 60800, 59800, 60600],
        "close": [60500, 61800, 61100, 60200, 62700],
        "volume": [100, 200, 300, 400, 500],
    },
    index=pd.to_datetime(["2026-10-06", "2026-10-07", "2026-10-08", "2026-10-12", "2026-10-13"]),
).astype(float)


def random_ohlcv(n=120, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = close * np.exp(rng.normal(0, 0.01, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.02, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.02, n))
    vol = rng.integers(1_000, 100_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol},
                        index=pd.bdate_range("2024-01-01", periods=n))


def test_candle_shape_matches_slide_row():
    f = factors.compute_factors(SLIDE)
    # 10-08: O62400 H62600 L60800 C61100  (범위 1800)
    d = pd.Timestamp("2026-10-08")
    assert f["close_pos"][d] == pytest.approx((61100 - 60800) / 1800)
    assert f["upper_shadow"][d] == pytest.approx((62600 - 62400) / 1800)
    assert f["body_ratio"][d] == pytest.approx(1300 / 1800)
    # 윗꼬리 + 몸통 + 아랫꼬리 = 1 (전체 캔들 길이)
    lower = (np.minimum(SLIDE["open"], SLIDE["close"]) - SLIDE["low"]) / (SLIDE["high"] - SLIDE["low"])
    total = f["upper_shadow"] + f["body_ratio"] + lower
    assert total.to_numpy() == pytest.approx(1.0, abs=1e-6)
    assert ((f["close_pos"] >= 0) & (f["close_pos"] <= 1)).all()


def test_turnover_and_flat_candle_has_no_division_by_zero():
    f = factors.compute_factors(SLIDE)
    assert f["turnover"].tolist() == [60500 * 100, 61800 * 200, 61100 * 300, 60200 * 400, 62700 * 500]
    flat = pd.DataFrame({"open": [10.0], "high": [10.0], "low": [10.0], "close": [10.0], "volume": [5.0]},
                        index=pd.to_datetime(["2026-01-02"]))
    g = factors.compute_factors(flat)
    for k in ("close_pos", "upper_shadow", "body_ratio"):
        assert np.isfinite(g[k]).all() and g[k].iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_rolling_and_trend_factors_match_manual():
    df = random_ohlcv()
    f = factors.compute_factors(df)
    t = 60
    assert f["vol_ratio"].iloc[t] == pytest.approx(df["volume"].iloc[t] / df["volume"].iloc[t - 19: t + 1].mean())
    assert f["ma20_gap"].iloc[t] == pytest.approx(df["close"].iloc[t] / df["close"].iloc[t - 19: t + 1].mean() - 1)
    assert f["ret_5d"].iloc[t] == pytest.approx(df["close"].iloc[t] / df["close"].iloc[t - 5] - 1)


def test_night_factors_match_manual():
    df = random_ohlcv()
    f = factors.compute_factors(df)
    rn = (df["open"] / df["close"].shift(1) - 1)
    t = 60
    win = rn.iloc[t - 19: t + 1]
    assert f["night_win_20"].iloc[t] == pytest.approx((win > 0).mean())
    assert f["night_sharpe_20"].iloc[t] == pytest.approx(win.mean() / win.std())


def test_warmup_is_nan_not_partial_window():
    f = factors.compute_factors(random_ohlcv())
    assert f["vol_ratio"].iloc[:19].isna().all() and f["vol_ratio"].iloc[19:].notna().all()
    assert f["ma20_gap"].iloc[:19].isna().all()
    assert f["ret_5d"].iloc[:5].isna().all() and f["ret_5d"].iloc[5:].notna().all()
    # R_night 는 첫 행이 NaN 이라 20일 창이 차는 건 21번째 행부터
    assert f["night_win_20"].iloc[:20].isna().all() and f["night_win_20"].iloc[20:].notna().all()
    assert f["night_sharpe_20"].iloc[:20].isna().all()


def test_no_lookahead_future_data_cannot_change_past_factors():
    """t 이후 OHLCV 를 전부 망가뜨려도 t 까지의 팩터 값은 비트 단위로 같아야 한다."""
    df = random_ohlcv()
    base = factors.compute_factors(df)
    cut = df.index[70]
    broken = df.copy()
    broken.loc[broken.index > cut] = np.random.default_rng(1).uniform(1, 1e6, size=broken.loc[broken.index > cut].shape)
    after = factors.compute_factors(broken)
    for k in factors.FACTORS:
        pd.testing.assert_series_equal(base[k].loc[:cut], after[k].loc[:cut], check_exact=True)


def test_factor_at_t_depends_on_day_t_close_so_trade_signal_needs_shift():
    """팩터(t) 는 t 일 종가를 쓰므로 R_night(t) 와 곱하면 안 된다 -> portfolio.signal 의 shift(1) 로만 사용."""
    p = {k: pd.DataFrame({"A": random_ohlcv(seed=1)[k], "B": random_ohlcv(seed=2)[k]}) for k in factors.OHLCV}
    f = factors.factor_panel(p)
    score = f["close_pos"]
    sig = pf.signal(score, k=1)
    assert not sig.iloc[0].any()  # 첫날은 전일 신호가 없다
    pd.testing.assert_frame_equal(sig.iloc[1:].reset_index(drop=True),
                                  pf.top_k_mask(score, 1).iloc[:-1].reset_index(drop=True))
    # 점수(t)를 올려 t 일 신호를 바꿔도 t 일 수익에 쓰이는 신호(sig[t])는 변하지 않는다
    t = 50
    p2 = {k: v.copy() for k, v in p.items()}
    p2["close"].iloc[t] *= 1.2
    sig2 = pf.signal(factors.factor_panel(p2)["close_pos"], k=1)
    pd.testing.assert_series_equal(sig.iloc[t], sig2.iloc[t])


def test_panel_equals_per_symbol_and_add_columns_keeps_ohlcv():
    a, b = random_ohlcv(seed=1), random_ohlcv(seed=2)
    panel = {k: pd.DataFrame({"A": a[k], "B": b[k]}) for k in factors.OHLCV}
    fp = factors.factor_panel(panel)
    for name in factors.FACTORS:
        pd.testing.assert_series_equal(fp[name]["A"], factors.compute_factors(a)[name], check_names=False)
        pd.testing.assert_series_equal(fp[name]["B"], factors.compute_factors(b)[name], check_names=False)
    out = factors.add_factor_columns(a)
    assert list(out.columns) == list(factors.OHLCV) + list(factors.FACTORS)
    pd.testing.assert_frame_equal(out[list(factors.OHLCV)], a)


def test_rejects_missing_columns_and_unsorted_index():
    with pytest.raises(ValueError, match="누락"):
        factors.compute_factors(SLIDE.drop(columns="volume"))
    with pytest.raises(ValueError, match="오름차순"):
        factors.compute_factors(SLIDE.iloc[::-1])


def test_api_start_end_are_distinct_query_params():
    """start/end 가 같은 Query 객체를 공유하면 start 가 end 값을 읽는 버그의 회귀 테스트."""
    from fastapi.testclient import TestClient

    import main

    seen = {}
    orig = main._date_range
    main._date_range = lambda s, e: seen.update(s=s, e=e) or orig(s, e)
    try:
        r = TestClient(main.app, raise_server_exceptions=False).get("/backtest/top5?start=20250101&end=20240101")
    finally:
        main._date_range = orig
    assert seen == {"s": "20250101", "e": "20240101"}
    assert r.status_code == 422
