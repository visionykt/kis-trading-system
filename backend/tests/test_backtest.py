import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

import backtest

# 강의 슬라이드 3 의 005930 예시 5행
SLIDE = pd.DataFrame(
    {
        "open": [59200, 61100, 62400, 61500, 60900],
        "high": [60800, 62000, 62600, 61900, 63000],
        "low": [58900, 60700, 60800, 59800, 60600],
        "close": [60500, 61800, 61100, 60200, 62700],
        "volume": [1, 1, 1, 1, 1],
    },
    index=pd.to_datetime(["2026-10-06", "2026-10-07", "2026-10-08", "2026-10-12", "2026-10-13"]),
).astype({"open": float, "high": float, "low": float, "close": float})


def test_day_night_matches_slide():
    r = backtest.day_night_returns(SLIDE)
    assert list(r.index.strftime("%m-%d")) == ["10-07", "10-08", "10-12", "10-13"]  # 첫날은 밤 수익률 없음
    assert r["night"].round(3).tolist() == [0.010, 0.010, 0.007, 0.012]
    assert r["day"].round(3).tolist() == [0.011, -0.021, -0.021, 0.030]
    assert r.loc["2026-10-08", "night"] == pytest.approx(62400 / 61800 - 1)


def test_unsorted_input_is_sorted():
    a = backtest.day_night_returns(SLIDE)
    b = backtest.day_night_returns(SLIDE.iloc[::-1])
    pd.testing.assert_frame_equal(a, b)


def test_metrics_and_identity():
    result, rets = backtest.run_backtest(SLIDE)
    s = result["strategies"]
    assert s["hold"]["cumulative_return"] == pytest.approx(62700 / 60500 - 1)
    assert s["night"]["cumulative_return"] == pytest.approx(
        (62400 / 61800 * 61500 / 61100 * 60900 / 60200 * 61100 / 60500) - 1)
    assert s["day"]["win_rate"] == 0.5 and s["night"]["win_rate"] == 1.0
    assert s["night"]["mdd"] == 0.0
    assert s["day"]["mdd"] == pytest.approx((61100 / 62400) * (60200 / 61500) - 1)
    assert result["identity_max_abs_error"] < 1e-12
    assert rets["cum_hold"].iloc[-1] == pytest.approx(62700 / 60500)


def test_mdd_counts_loss_from_initial_capital():
    assert backtest.max_drawdown(pd.Series([-0.1, 0.05])) == pytest.approx(-0.1)


def test_needs_two_rows():
    with pytest.raises(ValueError):
        backtest.run_backtest(SLIDE.iloc[:1])


def test_save_results(tmp_path):
    result, rets = backtest.run_backtest(SLIDE)
    files = backtest.save_results("005930", result, rets, tmp_path)
    assert {"summary", "daily", "report"} <= files.keys()
    assert "밤(종배)" in Path(files["report"]).read_text(encoding="utf-8")


def test_collect_skips_unsettled_today(monkeypatch):
    from datetime import date, datetime as real_dt

    import candles

    class FakeDT(real_dt):
        @classmethod
        def now(cls, tz=None):
            return real_dt(2026, 10, 15, 13, 25, tzinfo=tz)

    rows = [{"symbol": "005930", "ts": date(2026, 10, d), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
            for d in (14, 15)]
    monkeypatch.setattr(candles, "datetime", FakeDT)
    monkeypatch.setattr(candles.kis, "get_daily_candles", lambda *a: rows)
    stored = []

    class Conn:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def cursor(self): return self
        def executemany(self, sql, rs): stored.extend(rs)

    monkeypatch.setattr(candles.db, "connect", Conn)
    info = candles.collect_daily_candles("005930", "20261001")
    assert info["last"] == "2026-10-14" and [r["ts"].day for r in stored] == [14]
