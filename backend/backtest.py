"""낮(장중) vs 밤(종가배팅) 수익률 백테스트.

    R_day   = Close(t) / Open(t) - 1          09:00 매수 -> 15:30 매도
    R_night = Open(t) / Close(t-1) - 1        전일 15:30 매수 -> 당일 09:00 매도
    R_hold  = Close(t) / Close(t-1) - 1       단순보유 = (1 + R_night) * (1 + R_day) - 1
    누적    = (1 + r).cumprod()               (수수료/세금 제외)

사용법 (컨테이너 안):  python backtest.py 005930 [--start 20200101] [--end 20261015] [--no-collect]
"""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

STRATEGIES = {"hold": "단순보유", "day": "낮(장중)", "night": "밤(종배)"}
RESULTS_DIR = Path(__file__).resolve().parent / "results"
TRADING_DAYS = 252


def day_night_returns(df: pd.DataFrame) -> pd.DataFrame:
    """ts 오름차순 OHLC -> 일별 hold/day/night 수익률.

    첫 행은 전일 종가가 없어 밤/보유 수익률을 만들 수 없으므로 세 전략 모두 버린다
    (같은 기간을 비교해야 (1+hold) = (1+night)(1+day) 항등식이 성립한다).
    """
    df = df.sort_index()
    prev_close = df["close"].shift(1)
    out = pd.DataFrame(
        {
            "open": df["open"],
            "close": df["close"],
            "hold": df["close"] / prev_close - 1,
            "day": df["close"] / df["open"] - 1,
            "night": df["open"] / prev_close - 1,
        },
        index=df.index,
    )
    return out.iloc[1:]


def max_drawdown(r: pd.Series) -> float:
    equity = pd.concat([pd.Series([1.0]), (1 + r).cumprod().reset_index(drop=True)])  # 원금 1.0 부터
    return float((equity / equity.cummax() - 1).min())


def metrics(r: pd.Series) -> dict:
    n = len(r)
    total = float((1 + r).prod() - 1)
    std = r.std()
    return {
        "cumulative_return": total,
        "win_rate": float((r > 0).mean()),
        "mdd": max_drawdown(r),
        "sharpe": float(r.mean() / std * math.sqrt(TRADING_DAYS)) if std > 0 else None,
        "cagr": float((1 + total) ** (TRADING_DAYS / n) - 1) if n else None,
        "days": n,
    }


def run_backtest(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """일봉 DataFrame -> (전략별 성과지표, 일별 수익률/누적곡선)."""
    rets = day_night_returns(df)
    if rets.empty:
        raise ValueError("백테스트에는 최소 2일치 일봉이 필요합니다")
    for k in STRATEGIES:
        rets[f"cum_{k}"] = (1 + rets[k]).cumprod()
    summary = {k: metrics(rets[k]) for k in STRATEGIES}
    # 검증: 낮 x 밤 복리 = 단순보유
    identity_err = float((rets["cum_night"] * rets["cum_day"] - rets["cum_hold"]).abs().max())
    return {
        "period": {"start": rets.index[0].date().isoformat(), "end": rets.index[-1].date().isoformat()},
        "strategies": summary,
        "identity_max_abs_error": identity_err,
    }, rets


def format_report(symbol: str, result: dict) -> str:
    p = result["period"]
    lines = [
        f"[{symbol}] 낮 vs 밤 백테스트  {p['start']} ~ {p['end']}  ({result['strategies']['hold']['days']}거래일, 수수료 제외)",
        f"{'전략':<10}{'누적수익률':>12}{'CAGR':>10}{'승률':>9}{'MDD':>10}{'Sharpe':>9}",
    ]
    for k, name in STRATEGIES.items():
        m = result["strategies"][k]
        sharpe = f"{m['sharpe']:.2f}" if m["sharpe"] is not None else "-"
        lines.append(
            f"{name:<10}{m['cumulative_return']:>+12.2%}{m['cagr']:>+10.2%}{m['win_rate']:>9.1%}"
            f"{m['mdd']:>+10.2%}{sharpe:>9}"
        )
    lines.append(f"검증 (1+밤)x(1+낮) = 단순보유 : 최대 오차 {result['identity_max_abs_error']:.2e}")
    return "\n".join(lines)


def save_chart(symbol: str, rets: pd.DataFrame, path: Path) -> bool:
    import charts

    series = [
        {"label": "Buy & Hold", "y": rets["cum_hold"], "color": charts.CATEGORICAL[0], "ls": "--"},
        {"label": "Day 09:00->15:30", "y": rets["cum_day"], "color": charts.CATEGORICAL[1]},
        {"label": "Night 15:30->09:00", "y": rets["cum_night"], "color": charts.CATEGORICAL[2]},
    ]
    return charts.line_chart(path, [{"series": series}], f"{symbol} Buy & Hold vs Day vs Night (fees excluded)")


def save_results(symbol: str, result: dict, rets: pd.DataFrame, out_dir: Path = RESULTS_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"mission01_{symbol}"
    files = {"summary": base.with_name(base.name + "_summary.json"), "daily": base.with_name(base.name + "_daily.csv")}
    payload = {"symbol": symbol, "generated_at": datetime.now().isoformat(timespec="seconds"), **result}
    files["summary"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rets.rename_axis("ts").to_csv(files["daily"], float_format="%.6f")
    files["report"] = base.with_name(base.name + "_report.txt")
    files["report"].write_text(format_report(symbol, result) + "\n", encoding="utf-8")
    chart = base.with_name(base.name + "_chart.png")
    if save_chart(symbol, rets, chart):
        files["chart"] = chart
    return {k: str(v) for k, v in files.items()}


def main() -> None:
    import candles

    ap = argparse.ArgumentParser(description="MISSION 01: 낮(장중) vs 밤(종배) 백테스트")
    ap.add_argument("symbol", nargs="?", default="005930")
    ap.add_argument("--start", default=candles.DEFAULT_START, help="YYYYMMDD")
    ap.add_argument("--end", default=datetime.now(candles.KST).strftime("%Y%m%d"), help="YYYYMMDD")
    ap.add_argument("--no-collect", action="store_true", help="KIS 호출 없이 DB 데이터만 사용")
    args = ap.parse_args()

    if not args.no_collect:
        info = candles.collect_daily_candles(args.symbol, args.start, args.end)
        print(f"KIS 수집/적재: {info['rows']}행 ({info['first']} ~ {info['last']}), 수정주가 FID_ORG_ADJ_PRC=0")
    df = candles.load_daily_candles(
        args.symbol,
        datetime.strptime(args.start, "%Y%m%d").date(),
        datetime.strptime(args.end, "%Y%m%d").date(),
    )
    result, rets = run_backtest(df)
    print(format_report(args.symbol, result))
    for k, v in save_results(args.symbol, result, rets).items():
        print(f"  {k:<8}-> {v}")


if __name__ == "__main__":
    main()
