"""KIS 일봉을 daily_candles 테이블에 적재/조회한다."""
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd

import db
import kis

DEFAULT_START = "20200101"
KST = timezone(timedelta(hours=9))
MARKET_SETTLED = time(15, 40)  # 15:30 종가 동시호가 체결 후 여유

UPSERT = """
INSERT INTO daily_candles (symbol, ts, open, high, low, close, volume)
VALUES (%(symbol)s, %(ts)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s)
ON CONFLICT (symbol, ts) DO UPDATE SET
    open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
    close = EXCLUDED.close, volume = EXCLUDED.volume
"""


def collect_daily_candles(symbol: str, start: str = DEFAULT_START, end: str | None = None) -> dict:
    """KIS 에서 start~end 수정주가 일봉을 받아 upsert 한다 (재실행해도 중복 없음)."""
    now = datetime.now(KST)
    end = end or now.strftime("%Y%m%d")
    rows = kis.get_daily_candles(symbol, start, end)
    if now.time() < MARKET_SETTLED:
        # 장중 당일 캔들은 종가가 미확정이라 백테스트를 오염시키므로 적재하지 않는다
        rows = [r for r in rows if r["ts"] != now.date()]
    if rows:
        with db.connect() as conn, conn.cursor() as cur:
            cur.executemany(UPSERT, rows)
    return {
        "symbol": symbol,
        "rows": len(rows),
        "first": rows[0]["ts"].isoformat() if rows else None,
        "last": rows[-1]["ts"].isoformat() if rows else None,
    }


def load_daily_candles(symbol: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
    """daily_candles 를 날짜 오름차순(ORDER BY ts ASC) DataFrame 으로 읽는다."""
    sql = "SELECT ts, open, high, low, close, volume FROM daily_candles WHERE symbol = %s"
    params: list = [symbol]
    if start:
        sql += " AND ts >= %s"
        params.append(start)
    if end:
        sql += " AND ts <= %s"
        params.append(end)
    sql += " ORDER BY ts ASC"
    with db.connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts")
