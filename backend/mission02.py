"""MISSION 02: KOSPI 100 일봉 백필 & 유니버스 낮 vs 밤 백테스트 (+ 수수료 민감도, 수정주가 점검).

사용법 (컨테이너 안):  python mission02.py [--start 20200101] [--no-collect]
"""
import argparse
import json
from datetime import datetime

import pandas as pd

import candles
import charts
import kis
import portfolio as pf
import universe
from backtest import RESULTS_DIR, STRATEGIES, metrics

# 액면분할 검증 예시: 카카오 2021-04-15 5:1 분할
SPLIT_DEMO = ("035720", "20210401", "20210430", "2021-04-15")
# 하루 가격제한폭 ±30% 를 넘는 밤 수익률은 (장기 거래정지 직후가 아니면) 원주가 착시 의심
SUSPICIOUS = -0.30


def pct(v: float | None) -> str:
    return "-" if v is None else f"{v:+.2%}"


def adjusted_price_check(rets: dict[str, pd.DataFrame], demo: bool) -> dict:
    night = rets["night"].stack()
    worst = night.nsmallest(5)
    out = {
        "threshold": SUSPICIOUS,
        "suspicious_nights": int((night < SUSPICIOUS).sum()),
        "worst_nights": [{"ts": ts.date().isoformat(), "symbol": s, "night": float(v)} for (ts, s), v in worst.items()],
    }
    if demo:
        sym, start, end, day = SPLIT_DEMO
        try:
            cmp = universe.raw_vs_adjusted(sym, start, end)
            row = cmp.loc[day]
            out["split_demo"] = {
                "symbol": sym, "date": day,
                "raw": {"prev_close": float(cmp["raw_close"].shift(1).loc[day]), "open": float(row["raw_open"]),
                        "night": float(row["raw_night"])},
                "adjusted": {"prev_close": float(cmp["adjusted_close"].shift(1).loc[day]),
                             "open": float(row["adjusted_open"]), "night": float(row["adjusted_night"])},
            }
        except (kis.KISError, KeyError) as e:
            out["split_demo"] = {"error": str(e)[:200]}
    return out


def build(start=None, end=None, demo=True) -> dict:
    panel = universe.load_panel(start=start, end=end)
    names = {s["symbol"]: s for s in universe.list_symbols()}
    rets = pf.returns(panel)
    table = pf.per_symbol_table(rets, names)
    ew = pf.equal_weight(rets)
    return {
        "panel": panel, "rets": rets, "table": table, "ew": ew,
        "ew_metrics": {k: metrics(ew[k]) for k in STRATEGIES},
        "ew_costs": pf.cost_table(ew["night"]),
        "adjusted_check": adjusted_price_check(rets, demo),
    }


def format_report(res: dict) -> str:
    t, ew = res["table"], res["ew"]
    L = [f"[KOSPI 100] {len(t)}종목  {ew.index[0].date()} ~ {ew.index[-1].date()}  ({len(ew)}거래일, 동일가중 1/N)"]
    L.append(f"{'전략':<12}{'누적수익률':>14}{'CAGR':>10}{'승률':>9}{'MDD':>10}{'Sharpe':>9}")
    for k, name in STRATEGIES.items():
        m = res["ew_metrics"][k]
        L.append(f"{name:<12}{m['cumulative_return']:>+14.2%}{m['cagr']:>+10.2%}{m['win_rate']:>9.1%}"
                 f"{m['mdd']:>+10.2%}{m['sharpe']:>9.2f}")
    L.append("\n[수수료 민감도] 동일가중 종배, 매일 왕복비용 차감 r_net = r_night - cost")
    for c in res["ew_costs"]:
        L.append(f"  cost {c['cost']:.2%}: 누적 {c['cumulative_return']:>+10.2%}  CAGR {c['cagr']:>+7.2%}  MDD {c['mdd']:>+7.2%}")
    nb = int(t["night_beats_day"].sum())
    L.append(f"\n밤 > 낮 종목: {nb}/{len(t)}   밤 누적 > 0: {int((t['night_cumulative_return'] > 0).sum())}/{len(t)}"
             f"   낮 누적 > 0: {int((t['day_cumulative_return'] > 0).sum())}/{len(t)}")
    L.append("\n[종배 승률 Top-5]")
    for sym, r in t.head(5).iterrows():
        L.append(f"  {sym} {r['name']:<12} 승률 {r['night_win_rate']:.1%}  밤 {pct(r['night_cumulative_return'])}"
                 f"  낮 {pct(r['day_cumulative_return'])}  ({r['days']}일)")
    a = res["adjusted_check"]
    L.append(f"\n[수정주가 점검] 밤 수익률 < {a['threshold']:.0%} : {a['suspicious_nights']}건 / 최악 "
             + ", ".join(f"{w['symbol']} {w['ts']} {w['night']:+.1%}" for w in a["worst_nights"][:3]))
    d = a.get("split_demo")
    if d and "error" not in d:
        L.append(f"  카카오 {d['date']} 액면분할(5:1): 원주가 밤 {d['raw']['night']:+.1%}"
                 f" ({d['raw']['prev_close']:,.0f} -> {d['raw']['open']:,.0f})  vs  수정주가 밤 {d['adjusted']['night']:+.1%}"
                 f" ({d['adjusted']['prev_close']:,.0f} -> {d['adjusted']['open']:,.0f})")
    elif d:
        L.append(f"  액면분할 비교 실패: {d['error']}")
    return "\n".join(L)


def summary_json(res: dict) -> dict:
    t, ew = res["table"], res["ew"]
    return {
        "universe": universe.UNIVERSE, "symbols": len(t),
        "period": {"start": ew.index[0].date().isoformat(), "end": ew.index[-1].date().isoformat(), "days": len(ew)},
        "equal_weight": res["ew_metrics"], "equal_weight_night_costs": res["ew_costs"],
        "night_beats_day": int(t["night_beats_day"].sum()),
        "top5_night_win_rate": t.head(5).reset_index()[
            ["symbol", "name", "night_win_rate", "night_cumulative_return", "day_cumulative_return", "days"]
        ].to_dict("records"),
        "adjusted_check": res["adjusted_check"],
    }


def save(res: dict) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    f = {k: RESULTS_DIR / f"mission02_{k}" for k in ("summary.json", "symbols.csv", "daily.csv", "report.txt", "chart.png")}
    f["summary.json"].write_text(json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"),
                                             **summary_json(res)}, ensure_ascii=False, indent=2), encoding="utf-8")
    res["table"].to_csv(f["symbols.csv"], float_format="%.6f", encoding="utf-8-sig")
    ew = res["ew"].copy()
    for k in STRATEGIES:
        ew[f"cum_{k}"] = charts.cum(ew[k])
    ew.rename_axis("ts").to_csv(f["daily.csv"], float_format="%.6f")
    f["report.txt"].write_text(format_report(res) + "\n", encoding="utf-8")
    left = [
        {"label": "Buy & Hold", "y": ew["cum_hold"], "color": charts.CATEGORICAL[0], "ls": "--"},
        {"label": "Day", "y": ew["cum_day"], "color": charts.CATEGORICAL[1]},
        {"label": "Night", "y": ew["cum_night"], "color": charts.CATEGORICAL[2]},
    ]
    right = [{"label": f"Night - {c:.2%}", "y": charts.cum(res["ew"]["night"] - c), "color": col}
             for c, col in zip(pf.COSTS, charts.SEQUENTIAL_BLUE)]
    if not charts.line_chart(f["chart.png"], [
        {"title": "KOSPI 100 equal-weight: Buy & Hold vs Day vs Night", "series": left},
        {"title": "Night (overnight) after round-trip cost per day", "series": right},
    ], "MISSION 02  KOSPI 100 universe backtest (2020~)"):
        f.pop("chart.png")
    return {k: str(v) for k, v in f.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default=candles.DEFAULT_START, help="YYYYMMDD")
    ap.add_argument("--end", default=None, help="YYYYMMDD")
    ap.add_argument("--no-collect", action="store_true", help="KIS 호출 없이 DB 데이터만 사용")
    args = ap.parse_args()
    if not args.no_collect:
        print(f"symbols 등록: {universe.register_symbols()}종목")
        r = universe.backfill(args.start, args.end)
        print(f"백필: {r['symbols']}종목 {r['rows']}행 적재, 실패 {len(r['failed'])}건 {r['failed'] or ''}")
    res = build(universe.parse_ymd(args.start), universe.parse_ymd(args.end), demo=not args.no_collect)
    print(format_report(res))
    for k, v in save(res).items():
        print(f"  {k:<13}-> {v}")


if __name__ == "__main__":
    main()
