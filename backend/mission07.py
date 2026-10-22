"""Week 7 MISSION 02: 종가 배팅 최고 점수 챌린지 — 샘플 A~D / 강의 예시 가설 / 나만의 전략 3종 리더보드.

점수 3대 지표(왕복비용 0.20% 차감 기준): 누적수익률, 칼마(= 누적수익률 / |MDD|), 샤프(= 일평균/표준편차 x sqrt(252)).

모든 전략은 "t 일 15:20 까지의 데이터로 고른 종목 마스크(t)" 를 돌려주는 함수이고,
수익률과 곱할 때는 항상 mask.shift(1) 이다 (t 일 종가 매수 -> t+1 일 시가 매도 = R_night(t+1)).
종목 수는 가변이다: 선택된 종목끼리 동일비중, 하나도 없으면 그날은 현금(수익 0, 비용 0).
비용은 포지션을 가진 날마다 왕복 c 를 차감한다 (r_net = r - c).

사용법 (컨테이너 안):  python mission07.py [--start 20200101] [--end ...] [--split 20240101]
"""
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

import candles
import charts
import factors
import portfolio as pf
import universe
from backtest import RESULTS_DIR, metrics

COSTS = [0.0, 0.0005, 0.001, 0.002]
SCORE_COST = 0.002  # 점수 기준 비용
SPLIT = "20240101"  # 앞 구간(탐색·설계) / 뒤 구간(검증) 경계


# ------------------------------------------------------------------ 공통 도구

def rank_pct(df: pd.DataFrame) -> pd.DataFrame:
    """날짜별 횡단면 백분위 (0~1). 팩터 단위가 달라도 평균낼 수 있게 한다."""
    return df.rank(axis=1, pct=True)


def top_n(score: pd.DataFrame, n: int) -> pd.DataFrame:
    """점수 상위 n 종목 (NaN 점수는 후보 아님 -> 후보가 n 개보다 적으면 그만큼만 선택)."""
    return pf.top_k_mask(score, n) & score.notna()


def market_regime(panel, window: int) -> pd.Series:
    """KOSPI 100 동일가중 지수가 window 일선 위인가 (t 일 종가까지만 사용)."""
    idx = (panel["close"].pct_change(fill_method=None).mean(axis=1).fillna(0) + 1).cumprod()
    return idx > idx.rolling(window, min_periods=window).mean()


def broadcast(s: pd.Series, like: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.repeat(s.reindex(like.index).fillna(False).to_numpy()[:, None], like.shape[1], axis=1),
                        index=like.index, columns=like.columns)


# ------------------------------------------------------------------ 전략 정의
# 각 함수: (panel, F) -> bool DataFrame (t 일 종가에 사고 싶은 종목). panel/F 는 날짜 x 종목.

def pick_sample(key: str, k: int = 5):
    fn = pf.STRATEGIES[key][1]
    return lambda p, F: top_n(fn(p, pf.returns(p)), k)


def pick_lecture_hypothesis(p, F):
    """H. 강의 예시 그대로: 20일선 위 + 윗꼬리 <0.15, 거래대금·거래량배수 상위 3."""
    ok = (F["ma20_gap"] > 0) & (F["upper_shadow"] < 0.15)
    return top_n((rank_pct(F["turnover"]) + rank_pct(F["vol_ratio"])).where(ok), 3)


def pick_breakout_edge(p, F):
    """M1. 수급돌파 + 야간관성: 20일선 위 & 거래량 평소 이상 후보 중 3요소 평균순위 상위 3."""
    ok = (F["ma20_gap"] > 0) & (F["vol_ratio"] > 1)
    score = (rank_pct(F["night_sharpe_20"]) + rank_pct(F["turnover"]) + rank_pct(F["vol_ratio"])) / 3
    return top_n(score.where(ok), 3)


def pick_yang_sharpe(p, F):
    """M2. 고샤프 야간관성 x 양봉: 오늘 양봉 마감(close>open) 후보 중 20일 야간샤프 상위 3."""
    return top_n(F["night_sharpe_20"].where(p["close"] > p["open"]), 3)


def pick_regime_cash(p, F):
    """M3. M2 + 시장 레짐 필터: 동일가중 지수가 120일선 아래면 전액 현금."""
    return pick_yang_sharpe(p, F) & broadcast(market_regime(p, 120), p["close"])


STRATEGIES = {
    "A_momentum": ("샘플 A 장대양봉·고가마감", "(C-L)/(H-L) 상위 5", pick_sample("A_momentum")),
    "B_mean_reversion": ("샘플 B 낙폭과대 역추세", "-R_day 상위 5", pick_sample("B_mean_reversion")),
    "C_volume_surge": ("샘플 C 거래량폭발+양봉", "vol/MA20 x 양봉 상위 5", pick_sample("C_volume_surge")),
    "D_rolling_edge": ("샘플 D 20일 종배샤프", "night_sharpe_20 상위 5", pick_sample("D_rolling_edge")),
    "H_lecture_hypothesis": (
        "[참고] 강의 가설: 윗꼬리제거 수급돌파",
        "ma20_gap>0 & upper_shadow<0.15 후보 중 거래대금·거래량배수 평균순위 상위 3종목", pick_lecture_hypothesis),
    "M1_breakout_edge": (
        "[나만의 1] 수급돌파 야간관성",
        "20일선 위(ma20_gap>0) & 거래량 평소 이상(vol_ratio>1) 후보 중 night_sharpe_20·turnover·vol_ratio "
        "평균순위 상위 3종목", pick_breakout_edge),
    "M2_yang_sharpe": (
        "[나만의 2] 양봉 마감 고샤프 야간집중",
        "오늘 양봉 마감(close>open) 종목 중 night_sharpe_20(20일 야간 평균/표준편차) 상위 3종목", pick_yang_sharpe),
    "M3_regime_cash": (
        "[나만의 3] 레짐필터 현금보유",
        "M2 규칙 + KOSPI100 동일가중 지수가 120일선 아래(하락 추세장)면 매수하지 않고 전액 현금 (MDD 방어)", pick_regime_cash),
}
MINE = ["M1_breakout_edge", "M2_yang_sharpe", "M3_regime_cash"]


# ------------------------------------------------------------------ 실행·평가

def calmar(m: dict) -> float | None:
    return m["cumulative_return"] / abs(m["mdd"]) if m["mdd"] else None


def score_metrics(r: pd.Series, inv: pd.Series, cost: float) -> dict:
    """비용 차감 후 3대 점수 + 승률(전체일 / 보유일)."""
    net = r - cost * inv
    m = metrics(net)
    return {**m, "calmar": calmar(m), "win_rate_invested": float((net[inv] > 0).mean()) if inv.any() else None,
            "invested_days": int(inv.sum())}


def run_mask(mask: pd.DataFrame, night: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """선택 마스크(t) -> (일 수익률, 보유 여부, 실제 적용된 신호). 신호 = mask.shift(1) — 미래 참조 차단."""
    sig = mask.shift(1, fill_value=False)
    r, inv = pf.portfolio_returns(sig, night)
    return r, inv, sig


def lookahead_check(panel, fn, cut: pd.Timestamp) -> bool:
    """cut 일 이후 OHLCV 를 망가뜨려도 cut 일까지의 선택 마스크가 같은지 (cut 일 자체 포함해 망가뜨림 -> 신호는 cut-1 까지만 의존)."""
    def sig(p):
        with np.errstate(all="ignore"):  # 망가뜨린 가격으로 만든 누적지수는 float 범위를 넘을 수 있다 (과거 신호 비교에는 무관)
            return fn(p, factors.compute_factors(p)).shift(1, fill_value=False)

    base = sig(panel)
    rng = np.random.default_rng(0)
    broken = {f: df.copy() for f, df in panel.items()}
    fut = broken["open"].index >= cut
    for df in broken.values():
        df.loc[fut] = rng.uniform(1, 1e6, size=df.loc[fut].shape)
    return bool(base.loc[:cut].equals(sig(broken).loc[:cut]))


def build(start=None, end=None, split: str = SPLIT, costs=COSTS) -> dict:
    panel = universe.load_panel(start=start, end=end)
    names = {s["symbol"]: s["name"] for s in universe.list_symbols()}
    rets = pf.returns(panel)
    night = rets["night"]
    F = factors.compute_factors(panel)

    masks = {k: fn(panel, F) for k, (_, _, fn) in STRATEGIES.items()}
    runs = {k: run_mask(m, night) for k, m in masks.items()}
    cheat = {k: pf.portfolio_returns(m, night)[0] for k, m in masks.items()}  # shift 없는 부정행위 시연

    # 모든 전략이 같은 기간을 보도록 마지막으로 시작하는 전략에 맞춘다 (팩터 워밍업 차이 제거)
    first = max(inv[inv].index[0] for _, inv, _ in runs.values())
    ew = pf.equal_weight(rets)["night"]
    idx = ew.index[ew.index >= first]
    cut = pd.Timestamp(universe.parse_ymd(split)) if split else None

    def board_row(key, name, criteria, r, inv):
        r, inv = r.reindex(idx).fillna(0.0), inv.reindex(idx).fillna(False)
        row = {"key": key, "name": name, "criteria": criteria,
               "net": [{"cost": c, **score_metrics(r, inv, c)} for c in costs]}
        row.update(next(n for n in row["net"] if abs(n["cost"] - SCORE_COST) < 1e-12))
        if cut is not None:  # 설계 구간(앞) / 검증 구간(뒤) 분리 성과
            for tag, sl in (("early", r.index < cut), ("late", r.index >= cut)):
                m = score_metrics(r[sl], inv[sl], SCORE_COST)
                row[tag] = {k: m[k] for k in ("cumulative_return", "mdd", "calmar", "sharpe", "win_rate")}
        return row, r, inv

    board, curves, invested = [], {}, {}
    row, r, inv = board_row("KOSPI100_EW", "KOSPI100 동일가중 종배 (벤치마크)", "100종목 전체 동일비중 매일 종배",
                            ew, pd.Series(True, index=ew.index))
    board.append(row)
    curves["KOSPI100_EW"] = r
    for key, (name, criteria, _) in STRATEGIES.items():
        r0, inv0, _ = runs[key]
        row, r, inv = board_row(key, name, criteria, r0, inv0)
        with np.errstate(over="ignore"):  # 부정행위 시연값은 천문학적이라 누적곱이 float 범위를 넘을 수 있다
            row["cheat_no_shift_cumulative_return"] = score_metrics(
                cheat[key].reindex(idx).fillna(0.0), inv, SCORE_COST)["cumulative_return"]
        board.append(row)
        curves[key], invested[key] = r, inv

    cuts = [panel["open"].index[int(len(panel["open"].index) * q)] for q in (0.3, 0.6, 0.9)]
    look = {k: all(lookahead_check(panel, fn, c) for c in cuts) for k, (_, _, fn) in STRATEGIES.items()}
    best = {m: max(board[1:], key=lambda b: b[m] if b[m] is not None else -1e9)["key"]
            for m in ("cumulative_return", "calmar", "sharpe")}
    return {"period": {"start": idx[0].date().isoformat(), "end": idx[-1].date().isoformat(), "days": len(idx)},
            "split": split, "score_cost": SCORE_COST, "board": board, "best": best, "lookahead": look,
            "lookahead_cuts": [c.date().isoformat() for c in cuts], "curves": pd.DataFrame(curves),
            "invested": invested, "masks": masks, "panel": panel, "factors": F, "night": night, "names": names}


# ------------------------------------------------------------------ 선정 내역 (설명 가능성)

PICK_COLS = ["close_pos", "upper_shadow", "ma20_gap", "vol_ratio", "ret_5d", "night_win_20", "night_sharpe_20"]


def picks_log(res: dict, key: str) -> pd.DataFrame:
    """전략 key 가 매일 고른 종목: 신호일(t) 팩터값 + 실제로 얻은 다음 날 밤 수익률 R_night(t+1)."""
    mask, night, F, names = res["masks"][key], res["night"], res["factors"], res["names"]
    rows = []
    nxt = night.shift(-1)
    for ts, row in mask.loc[res["period"]["start"]:].iterrows():
        for sym in row.index[row.to_numpy()]:
            rows.append({"signal_date": ts.date().isoformat(), "symbol": sym, "name": names.get(sym),
                         **{c: F[c].at[ts, sym] for c in PICK_COLS}, "next_night_return": nxt.at[ts, sym]})
    return pd.DataFrame(rows)


def concentration(res: dict, key: str, top: int = 5) -> list[dict]:
    """한 두 종목 운으로 점수가 나온 건 아닌지: 종목별 선택 일수와 누적기여(로그수익 합)."""
    sig = res["masks"][key].shift(1, fill_value=False).loc[res["period"]["start"]:]
    night = res["night"].reindex(sig.index)
    held = sig & night.notna()
    contrib = np.log1p(night.where(held)).sum() / held.sum(axis=1).clip(lower=1).mean()
    days = held.sum()
    df = pd.DataFrame({"days": days, "contrib": contrib}).sort_values("days", ascending=False).head(top)
    return [{"symbol": s, "name": res["names"].get(s), "days": int(r.days), "log_contrib": float(r.contrib)}
            for s, r in df.iterrows()]


# ------------------------------------------------------------------ 리포트

def _fmt_row(b: dict, m: dict) -> str:
    sh = f"{m['sharpe']:.2f}" if m["sharpe"] is not None else "-"
    cal = f"{m['calmar']:.1f}" if m["calmar"] is not None else "-"
    wr = m["win_rate_invested"]
    return (f"{b['name']:<30}{m['cumulative_return']:>+12.0%}{m['mdd']:>+8.1%}{cal:>8}{sh:>7}"
            f"{m['win_rate']:>8.1%}{(f'{wr:.1%}' if wr is not None else '-'):>9}{m['invested_days']:>7}")


def format_report(res: dict) -> str:
    p = res["period"]
    L = [f"[MISSION 02 · Week 7] 종가배팅 최고 점수 챌린지  {p['start']} ~ {p['end']} ({p['days']}거래일)",
         f"점수 기준 왕복비용 {res['score_cost']:.2%} 차감 · 신호=mask.shift(1) · 선택 종목 없으면 현금\n"]
    hdr = f"{'전략':<30}{'누적수익률':>12}{'MDD':>8}{'Calmar':>8}{'Sharpe':>7}{'승률(전체)':>8}{'승률(보유)':>9}{'보유일':>7}"
    L += [hdr, "-" * len(hdr)]
    for b in res["board"]:
        L.append(_fmt_row(b, b))
    L.append("\n[왕복비용 민감도] 누적수익률 / 샤프")
    L.append(f"{'전략':<30}" + "".join(f"{c:>16.2%}" for c in COSTS))
    for b in res["board"]:
        L.append(f"{b['name']:<30}" + "".join(
            f"{n['cumulative_return']:>+10.0%} /{(n['sharpe'] or 0):>4.1f}" for n in b["net"]))
    if res["split"]:
        s = res["split"]
        L.append(f"\n[설계 구간 vs 검증 구간] {s} 이전 / 이후 (비용 {res['score_cost']:.2%}) — 누적 · Calmar · Sharpe")
        for b in res["board"]:
            e, l = b["early"], b["late"]
            L.append(f"{b['name']:<30}  이전 {e['cumulative_return']:>+7.0%} {e['calmar']:>6.1f} {e['sharpe']:>5.2f}"
                     f"   이후 {l['cumulative_return']:>+7.0%} {l['calmar']:>6.1f} {l['sharpe']:>5.2f}")
    L.append("\n[함정 01 · 미래참조] 신호 = mask.shift(1)")
    L.append(f"  cut({', '.join(res['lookahead_cuts'])}) 이후 데이터를 망가뜨려도 신호 불변: "
             + ", ".join(f"{k.split('_')[0]}={'OK' if v else 'FAIL'}" for k, v in res["lookahead"].items()))
    L.append("  shift(1) 을 빼면 부풀려지는 누적수익률(부정행위 시연): "
             + ", ".join(f"{b['key'].split('_')[0]} {b['cumulative_return']:+.0%}->{b['cheat_no_shift_cumulative_return']:+.0%}"
                         for b in res["board"][1:]))
    L.append("\n[최고 점수] " + " · ".join(
        f"{lbl} {next(b['name'] for b in res['board'] if b['key'] == res['best'][m])}"
        for m, lbl in (("cumulative_return", "수익률"), ("calmar", "Calmar"), ("sharpe", "Sharpe"))))
    L.append("\n[종목 선택 기준 · 설명]")
    for b in res["board"][1:]:
        L.append(f"  {b['name']}\n      {b['criteria']}")
    L.append("\n[집중도 점검] 가장 자주 뽑힌 종목 (한두 종목 운이 아닌지)")
    for k in MINE:
        c = res["concentration"][k]
        L.append(f"  {STRATEGIES[k][0]}: " + ", ".join(f"{x['name']}({x['days']}일)" for x in c))
    L.append("\n[주의] 유니버스는 '현재' KOSPI 100 구성종목이라 생존편향이 있고, 설계 구간·검증 구간 모두 전략 설계 중에 "
             "관찰했으므로 진정한 표본 외(out-of-sample) 결과가 아니다. 점수는 상한선으로 해석할 것.")
    return "\n".join(L)


def save(res: dict) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    f = {k: RESULTS_DIR / f"mission07_{k}" for k in ("leaderboard.json", "daily.csv", "report.txt", "picks.csv", "chart.png")}
    f["leaderboard.json"].write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"), "period": res["period"], "split": res["split"],
        "score_cost": res["score_cost"], "board": res["board"], "best": res["best"], "lookahead_check": res["lookahead"],
        "concentration": res["concentration"],
        "factors": {k: dict(zip(("group", "formula", "hypothesis"), v)) for k, v in factors.FACTORS.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    cur = res["curves"]
    cur.add_prefix("r_").join(cur.apply(charts.cum).add_prefix("cum_")).rename_axis("ts").to_csv(f["daily.csv"], float_format="%.6f")
    pd.concat({k: picks_log(res, k) for k in MINE}, names=["strategy"]).reset_index(level=0).to_csv(
        f["picks.csv"], index=False, float_format="%.6f")
    f["report.txt"].write_text(format_report(res) + "\n", encoding="utf-8")
    c = res["score_cost"]
    series = [{"label": "KOSPI100 EW", "y": charts.cum(cur["KOSPI100_EW"] - c), "color": charts.BASELINE, "ls": "--"}]
    for i, k in enumerate(STRATEGIES):
        r = cur[k] - c * res["invested"][k]
        series.append({"label": k.split("_")[0], "y": charts.cum(r),
                       "color": charts.CATEGORICAL[i % len(charts.CATEGORICAL)], "ls": "-" if k in MINE else ":"})
    if not charts.line_chart(f["chart.png"], [{"title": f"Net of {c:.2%} round-trip cost", "series": series}],
                             "Week 7 MISSION 02  samples A-D / lecture hypothesis / my strategies M1-M3", log=True):
        f.pop("chart.png")
    return {k: str(v) for k, v in f.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default=candles.DEFAULT_START, help="YYYYMMDD")
    ap.add_argument("--end", default=None, help="YYYYMMDD")
    ap.add_argument("--split", default=SPLIT, help="설계/검증 구간 경계 YYYYMMDD (빈 값이면 분리 안 함)")
    args = ap.parse_args()
    res = build(universe.parse_ymd(args.start), universe.parse_ymd(args.end), args.split)
    res["concentration"] = {k: concentration(res, k) for k in MINE}
    print(format_report(res))
    for k, v in save(res).items():
        print(f"  {k:<16}-> {v}")


if __name__ == "__main__":
    main()
