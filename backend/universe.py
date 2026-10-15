"""KOSPI 100 유니버스 등록 · 일봉 백필 · 패널(날짜 x 종목) 로딩."""
import csv
from datetime import date, datetime
from pathlib import Path

import pandas as pd

import candles
import db
import kis

KOSPI100_CSV = Path(__file__).resolve().parent / "data" / "kospi100.csv"
UNIVERSE = "KOSPI100"


def register_symbols(path: Path = KOSPI100_CSV, universe: str = UNIVERSE) -> int:
    """csv(symbol,name,sector)를 symbols 테이블에 upsert 한다."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [{**r, "universe": universe} for r in csv.DictReader(f)]
    with db.connect() as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO symbols (symbol, name, sector, universe) VALUES (%(symbol)s, %(name)s, %(sector)s, %(universe)s) "
            "ON CONFLICT (symbol) DO UPDATE SET name = EXCLUDED.name, sector = EXCLUDED.sector, universe = EXCLUDED.universe",
            rows,
        )
    return len(rows)


def list_symbols(universe: str = UNIVERSE) -> list[dict]:
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT symbol, name, sector FROM symbols WHERE universe = %s ORDER BY symbol", (universe,)
        ).fetchall()
    return [{"symbol": s, "name": n, "sector": sec} for s, n, sec in rows]


def last_dates(universe: str = UNIVERSE) -> dict[str, date]:
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT c.symbol, max(c.ts) FROM daily_candles c JOIN symbols s USING (symbol) "
            "WHERE s.universe = %s GROUP BY c.symbol",
            (universe,),
        ).fetchall()
    return dict(rows)


def backfill(start: str = candles.DEFAULT_START, end: str | None = None, universe: str = UNIVERSE,
             full: bool = False, log=print) -> dict:
    """유니버스 전 종목 일봉을 적재한다. 이미 받은 종목은 마지막 날짜부터만 이어 받는다 (full=True 면 전체 재수집).

    한 종목이 실패해도 나머지는 계속 진행하고 실패 목록을 돌려준다 (다시 실행하면 이어서 채움).
    """
    syms = list_symbols(universe)
    done = {} if full else last_dates(universe)
    total, failed = 0, []
    for i, s in enumerate(syms, 1):
        sym_start = start
        if s["symbol"] in done:
            # 마지막 날짜부터 다시 받아 직전 적재분(장중 미확정 등)을 덮어쓴다
            sym_start = max(start, done[s["symbol"]].strftime("%Y%m%d"))
        try:
            info = candles.collect_daily_candles(s["symbol"], sym_start, end)
        except kis.KISError as e:
            failed.append({"symbol": s["symbol"], "name": s["name"], "error": str(e)[:200]})
            log(f"[{i:3}/{len(syms)}] {s['symbol']} {s['name']}: 실패 {str(e)[:120]}")
            continue
        total += info["rows"]
        log(f"[{i:3}/{len(syms)}] {s['symbol']} {s['name']}: {info['rows']}행 ({info['first']} ~ {info['last']})")
    return {"symbols": len(syms), "rows": total, "failed": failed}


def load_panel(universe: str = UNIVERSE, start: date | None = None, end: date | None = None) -> dict[str, pd.DataFrame]:
    """{'open','high','low','close','volume'} -> DataFrame(index=ts 오름차순, columns=symbol)."""
    sql = ("SELECT c.symbol, c.ts, c.open, c.high, c.low, c.close, c.volume FROM daily_candles c "
           "JOIN symbols s USING (symbol) WHERE s.universe = %s")
    params: list = [universe]
    if start:
        sql += " AND c.ts >= %s"
        params.append(start)
    if end:
        sql += " AND c.ts <= %s"
        params.append(end)
    sql += " ORDER BY c.ts ASC"
    with db.connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    df = pd.DataFrame(rows, columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"])
    return {
        f: df.pivot(index="ts", columns="symbol", values=f).astype(float).sort_index()
        for f in ("open", "high", "low", "close", "volume")
    }


def raw_vs_adjusted(symbol: str, start: str, end: str) -> pd.DataFrame:
    """같은 구간을 수정주가/원주가로 받아 밤 수익률을 나란히 비교한다 (액면분할 착시 검증용)."""
    out = {}
    for label, adjusted in (("adjusted", True), ("raw", False)):
        rows = kis.inquire_daily_chart(symbol, start, end, adjusted=adjusted)
        s = pd.DataFrame(
            {"open": float(r["stck_oprc"]), "close": float(r["stck_clpr"])}
            for r in rows
        ).set_axis(pd.to_datetime([r["stck_bsop_date"] for r in rows])).sort_index()
        out[f"{label}_open"] = s["open"]
        out[f"{label}_close"] = s["close"]
        out[f"{label}_night"] = s["open"] / s["close"].shift(1) - 1
    return pd.DataFrame(out)


def parse_ymd(v: str | None) -> date | None:
    return datetime.strptime(v, "%Y%m%d").date() if v else None

