"""MISSION 03: 매일 15:20 KOSPI 100 중 종배 Top-5 선택 알고리즘 리더보드.

사용법 (컨테이너 안):  python mission03.py [--start 20200101] [--cost 0.002]
(데이터는 mission02.py 로 먼저 백필한다)
"""
import argparse
import json
from datetime import datetime

import pandas as pd

import candles
import charts
import portfolio as pf
import universe
from backtest import RESULTS_DIR

REPORT_COST = 0.002  # 리더보드에 함께 보여줄 순수익 기준 왕복비용 (보수적 상단)


def build(start=None, end=None, n_checks: int = 3) -> dict:
    panel = universe.load_panel(start=start, end=end)
    names = {s["symbol"]: s["name"] for s in universe.list_symbols()}
    res = pf.run_top5(panel)
    # 함정 01 증명: 실제 데이터에서 cut 일 이후를 망가뜨려도 cut 까지의 신호가 동일한지
    idx = panel["open"].index
    cuts = [idx[int(len(idx) * q)] for q in (0.25, 0.5, 0.9)][:n_checks]
    res["lookahead"] = {k: all(pf.lookahead_check(panel, fn, c) for c in cuts) for k, (_, fn) in pf.STRATEGIES.items()}
    res["lookahead_cuts"] = [c.date().isoformat() for c in cuts]
    # 오늘 밤(마지막 거래일 종가) 종배 후보
    last = idx[-1]
    res["latest_picks"] = {
        k: [{"symbol": s, "name": names.get(s)} for s in m.columns[m.loc[last]]] for k, m in res["picks"].items()
    }
    res["latest_date"] = last.date().isoformat()
    return res


def _net(b: dict, cost: float) -> dict:
    return next(n for n in b["net"] if abs(n["cost"] - cost) < 1e-12)


def format_report(res: dict, cost: float = REPORT_COST) -> str:
    p = res["period"]
    L = [f"[MISSION 03] KOSPI 100 Top-{pf.TOP_K} 종배  {p['start']} ~ {p['end']} ({p['days']}거래일), 종목당 1/{pf.TOP_K}"]
    L.append(f"{'전략':<26}{'누적(gross)':>14}{'알파':>14}{'CAGR':>9}{'승률':>8}{'MDD':>9}{'Sharpe':>8}"
             f"{f'누적(-{cost:.2%})':>15}{'MDD(net)':>10}")
    for b in res["board"]:
        n = _net(b, cost)
        L.append(f"{b['name']:<26}{b['cumulative_return']:>+14.1%}{b['alpha']:>+14.1%}{b['cagr']:>+9.1%}"
                 f"{b['win_rate']:>8.1%}{b['mdd']:>+9.1%}{b['sharpe']:>8.2f}{n['cumulative_return']:>+15.1%}{n['mdd']:>+10.1%}")
    L.append("\n[함정 01 · 미래참조] 신호 = top5_mask(score).shift(1)")
    L.append(f"  cut 이후 데이터를 무작위로 바꿔도 cut 까지 신호 불변 ({', '.join(res['lookahead_cuts'])}): "
             + ", ".join(f"{k.split('_')[0]}={'OK' if ok else 'FAIL'}" for k, ok in res["lookahead"].items()))
    L.append("  shift(1) 을 빼면 (같은 날 점수로 같은 날 밤 수익률 선택) 누적수익률이 이렇게 부풀려진다:")
    for b in res["board"][1:]:
        L.append(f"    {b['name']:<26} 정상 {b['cumulative_return']:>+12.1%}  ->  미래참조 {b['cheat_no_shift_cumulative_return']:>+16.1%}")
    L.append(f"\n[함정 02 · 거래비용] 왕복비용별 누적수익률")
    L.append(f"  {'전략':<26}" + "".join(f"{c:>12.2%}" for c in pf.COSTS))
    for b in res["board"]:
        L.append(f"  {b['name']:<26}" + "".join(f"{n['cumulative_return']:>+12.1%}" for n in b["net"]))
    L.append(f"\n[{res['latest_date']} 종가 기준 오늘 밤 종배 후보]")
    for k, picks in res["latest_picks"].items():
        L.append(f"  {pf.STRATEGIES[k][0]:<26} " + ", ".join(f"{x['name']}({x['symbol']})" for x in picks))
    return "\n".join(L)


def board_json(res: dict) -> list[dict]:
    return [{k: v for k, v in b.items()} for b in res["board"]]


def save(res: dict, cost: float = REPORT_COST) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    f = {k: RESULTS_DIR / f"mission03_{k}" for k in ("leaderboard.json", "daily.csv", "report.txt", "chart.png")}
    f["leaderboard.json"].write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"), "period": res["period"], "top_k": pf.TOP_K,
        "strategies": {k: v[0] for k, v in pf.STRATEGIES.items()}, "board": board_json(res),
        "lookahead_check": res["lookahead"], "lookahead_cuts": res["lookahead_cuts"],
        "latest_date": res["latest_date"], "latest_picks": res["latest_picks"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    curves = res["curves"]
    out = curves.add_prefix("r_").join(curves.apply(charts.cum).add_prefix("cum_"))
    out.rename_axis("ts").to_csv(f["daily.csv"], float_format="%.6f")
    f["report.txt"].write_text(format_report(res, cost) + "\n", encoding="utf-8")

    def series(c: float) -> list[dict]:
        inv = res["invested"]
        s = [{"label": "KOSPI100 EW", "y": charts.cum(curves["KOSPI100_EW"] - c), "color": charts.BASELINE, "ls": "--"}]
        for i, k in enumerate(pf.STRATEGIES):
            r = curves[k] - c * inv[k].reindex(curves.index).fillna(False)
            s.append({"label": k.split("_")[0], "y": charts.cum(r), "color": charts.CATEGORICAL[i]})
        return s

    if not charts.line_chart(f["chart.png"], [
        {"title": "Gross (no cost)", "series": series(0.0)},
        {"title": f"Net of {cost:.2%} round-trip cost per day", "series": series(cost)},
    ], f"MISSION 03  KOSPI 100 Top-{pf.TOP_K} overnight strategies vs equal-weight (A momentum, B mean-rev, "
       f"C volume, D 20d edge, E edge x close)", log=True):
        f.pop("chart.png")
    return {k: str(v) for k, v in f.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default=candles.DEFAULT_START, help="YYYYMMDD")
    ap.add_argument("--end", default=None, help="YYYYMMDD")
    ap.add_argument("--cost", type=float, default=REPORT_COST, help=f"리더보드 순수익 기준 왕복비용 {pf.COSTS} 중 하나")
    args = ap.parse_args()
    if args.cost not in pf.COSTS:
        ap.error(f"--cost 는 {pf.COSTS} 중 하나")
    res = build(universe.parse_ymd(args.start), universe.parse_ymd(args.end))
    print(format_report(res, args.cost))
    for k, v in save(res, args.cost).items():
        print(f"  {k:<16}-> {v}")


if __name__ == "__main__":
    main()
