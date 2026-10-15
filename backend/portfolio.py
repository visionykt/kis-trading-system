"""유니버스(날짜 x 종목 패널) 낮/밤 수익률, 동일가중 종배, Top-5 선택 전략 백테스트.

모든 입력 패널은 index=ts 오름차순, columns=symbol 인 DataFrame 이다.

미래 참조 방지 규칙: t 일 장 마감(15:20) 기준 점수 score(t) 로 고른 종목은 t 종가에 사서 t+1 시가에 판다.
그 수익은 R_night(t+1) 이므로 수익률과 곱하는 신호는 항상 top_k_mask(score).shift(1) 이다.
"""
import numpy as np
import pandas as pd

from backtest import metrics

COSTS = [0.0, 0.0005, 0.001, 0.0015, 0.002]  # 왕복 거래비용 (수수료+세금+슬리피지)
TOP_K = 5


def returns(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    o, c = panel["open"], panel["close"]
    prev_c = c.shift(1)  # 전 거래일에 해당 종목 행이 없으면(상장 전/거래정지) NaN -> 자동 제외
    return {"day": c / o - 1, "night": o / prev_c - 1, "hold": c / prev_c - 1}


# ---------------------------------------------------------------- MISSION 02

def per_symbol_table(rets: dict[str, pd.DataFrame], names: dict[str, dict] | None = None) -> pd.DataFrame:
    """종목별 낮/밤/보유 성과표 (각 종목이 밤 수익률을 가진 날만 사용)."""
    rows = []
    for sym in rets["night"].columns:
        valid = rets["night"][sym].notna() & rets["day"][sym].notna()
        if valid.sum() < 2:
            continue
        m = {k: metrics(rets[k][sym][valid]) for k in ("hold", "day", "night")}
        info = (names or {}).get(sym, {})
        rows.append({
            "symbol": sym, "name": info.get("name"), "sector": info.get("sector"), "days": int(valid.sum()),
            "start": rets["night"].index[valid][0].date().isoformat(),
            **{f"{k}_{f}": m[k][f] for k in ("hold", "day", "night") for f in ("cumulative_return", "win_rate", "mdd", "sharpe")},
        })
    df = pd.DataFrame(rows).set_index("symbol")
    df["night_beats_day"] = df["night_cumulative_return"] > df["day_cumulative_return"]
    return df.sort_values("night_win_rate", ascending=False)


def equal_weight(rets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """KOSPI 100 동일가중(그날 거래 가능한 종목 1/N) 일별 hold/day/night 수익률."""
    valid = rets["night"].notna() & rets["day"].notna()
    out = pd.DataFrame({k: rets[k].where(valid).mean(axis=1) for k in ("hold", "day", "night")})
    out["n"] = valid.sum(axis=1)
    return out[out["n"] > 0]


def cost_table(r: pd.Series, invested: pd.Series | None = None, costs=COSTS) -> list[dict]:
    """왕복 비용 c 를 포지션을 가진 날마다 차감한 r_net = r - c 의 성과."""
    invested = invested if invested is not None else pd.Series(True, index=r.index)
    return [{"cost": c, **metrics(r - c * invested)} for c in costs]


# ---------------------------------------------------------------- MISSION 03

def _rank_pct(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)


def score_momentum(p, r):
    """A. 장대양봉·고가마감: (C - L) / (H - L)."""
    rng = (p["high"] - p["low"]).replace(0, np.nan)
    return (p["close"] - p["low"]) / rng


def score_mean_reversion(p, r):
    """B. 장중 낙폭과대: -R_day(t)."""
    return -r["day"]


def score_volume_surge(p, r):
    """C. 거래량 폭발 + 양봉: V(t) / MA20_V * I(R_day > 0). 음봉은 후보에서 제외."""
    ratio = p["volume"] / p["volume"].rolling(20, min_periods=20).mean()
    return ratio.where(r["day"] > 0)


def score_rolling_edge(p, r):
    """D. 최근 20일 종배 샤프: Mean(R_night, 20) / Std(R_night, 20)."""
    n = r["night"]
    return n.rolling(20, min_periods=20).mean() / n.rolling(20, min_periods=20).std()


def score_edge_close(p, r):
    """E. (나만의 공식) 종배 우위 x 강한 마감: D 와 A 의 횡단면 백분위 평균.

    밤에 꾸준히 오른 종목(D) 중 오늘 고가 근처에서 끝난(A) 종목을 고른다.
    """
    return (_rank_pct(score_rolling_edge(p, r)) + _rank_pct(score_momentum(p, r))) / 2


STRATEGIES = {
    "A_momentum": ("A. 장대양봉·고가마감", score_momentum),
    "B_mean_reversion": ("B. 낙폭과대 역추세", score_mean_reversion),
    "C_volume_surge": ("C. 거래량 폭발+양봉", score_volume_surge),
    "D_rolling_edge": ("D. 20일 종배 샤프", score_rolling_edge),
    "E_edge_close": ("E. 종배우위x고가마감 (my)", score_edge_close),
}


def top_k_mask(score: pd.DataFrame, k: int = TOP_K) -> pd.DataFrame:
    """각 날짜에서 점수 상위 k 종목 True (NaN 점수는 후보 아님)."""
    return score.rank(axis=1, ascending=False, method="first") <= k


def signal(score: pd.DataFrame, k: int = TOP_K) -> pd.DataFrame:
    """t 일 수익률(R_night(t))에 곱할 보유 신호 = t-1 일 점수로 고른 종목."""
    return top_k_mask(score, k).shift(1, fill_value=False)


def portfolio_returns(sig: pd.DataFrame, night: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """동일비중(1/선택종목수) 종배 수익률. 고른 종목이 없거나 전부 거래정지면 현금(0)."""
    held = sig & night.notna()
    r = night.where(held).mean(axis=1)
    invested = held.any(axis=1)
    return r.fillna(0.0), invested


def lookahead_check(panel: dict[str, pd.DataFrame], score_fn, cut: pd.Timestamp, k: int = TOP_K) -> bool:
    """cut 이후 데이터를 모두 망가뜨려도 cut 일까지의 신호가 그대로인지 확인 (미래 참조 0 증명).

    cut 일 신호는 cut-1 일까지의 데이터로만 정해져야 하므로 cut 일 자체의 OHLCV 도 망가뜨린다.
    """
    def sig_of(p):
        return signal(score_fn(p, returns(p)), k)

    base = sig_of(panel)
    rng = np.random.default_rng(0)
    broken = {f: df.copy() for f, df in panel.items()}
    future = broken["open"].index >= cut
    for f, df in broken.items():
        df.loc[future] = rng.uniform(1, 1e6, size=df.loc[future].shape)
    return bool(base.loc[:cut].equals(sig_of(broken).loc[:cut]))


def run_top5(panel: dict[str, pd.DataFrame], k: int = TOP_K, costs=COSTS) -> dict:
    """전략별 Top-k 종배 vs KOSPI 100 동일가중 종배. 모든 전략을 같은 기간(전 전략 신호가 생긴 날부터)으로 맞춘다."""
    rets = returns(panel)
    ew = equal_weight(rets)
    series, invested, picks, cheat = {}, {}, {}, {}
    for key, (_, fn) in STRATEGIES.items():
        score = fn(panel, rets)
        sig = signal(score, k)
        series[key], invested[key] = portfolio_returns(sig, rets["night"])
        picks[key] = top_k_mask(score, k)
        # 함정 01 시연: shift 없이 같은 날 점수로 같은 날 밤 수익률을 고르면 (O(t)를 미리 아는 셈)
        cheat[key] = portfolio_returns(top_k_mask(score, k), rets["night"])[0]
    start = max(inv[inv].index[0] for inv in invested.values())
    idx = ew.index[ew.index >= start]
    base = ew["night"].reindex(idx)
    base_m = metrics(base)

    board = [{"key": "KOSPI100_EW", "name": "KOSPI100 동일가중 종배", **base_m, "alpha": 0.0,
              "net": cost_table(base, costs=costs)}]
    for key, (name, _) in STRATEGIES.items():
        r, inv = series[key].reindex(idx), invested[key].reindex(idx)
        m = metrics(r)
        board.append({
            "key": key, "name": name, **m,
            "alpha": m["cumulative_return"] - base_m["cumulative_return"],
            "alpha_cagr": m["cagr"] - base_m["cagr"],
            "invested_days": int(inv.sum()),
            "net": cost_table(r, inv, costs),
            "cheat_no_shift_cumulative_return": metrics(cheat[key].reindex(idx))["cumulative_return"],
        })
    curves = pd.DataFrame({"KOSPI100_EW": base, **{k_: series[k_].reindex(idx) for k_ in STRATEGIES}})
    return {"period": {"start": idx[0].date().isoformat(), "end": idx[-1].date().isoformat(), "days": len(idx)},
            "board": board, "curves": curves, "picks": picks, "invested": invested}
