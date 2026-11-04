import json
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

import pandas as pd
import psycopg
from fastapi import Depends, FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

import auth
import backtest
import candles
import db
import kis
import mission02
import mission03
import mission07
import universe


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="KIS Data Mining API", lifespan=lifespan)
app.include_router(auth.router)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(kis.KISError)
def kis_error_handler(request, exc: kis.KISError):
    return JSONResponse(status_code=502, content={"error": "KIS API error", "kis_status": exc.status, "kis_response": exc.body})


@app.exception_handler(kis.KISOrderRejected)
def kis_order_rejected_handler(request, exc: kis.KISOrderRejected):
    return JSONResponse(
        status_code=400,
        content={"error": "KIS order rejected", "msg_cd": exc.msg_cd, "message": exc.msg, "kis_response": exc.body},
    )


@app.get("/health")
def health():
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        version = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
        ).fetchone()
    return {
        "status": "ok",
        "timescaledb": version[0] if version else None,
        "kis_account": f"{os.environ.get('KIS_CANO')}-{os.environ.get('KIS_ACNT_PRDT_CD')}",
        "kis_app_key_loaded": bool(os.environ.get("KIS_APP_KEY")),
    }


@app.post("/kis/token")
def kis_token():
    """접근 토큰 발급 여부만 확인한다 (토큰 값 자체는 노출하지 않음)."""
    token = kis.get_token()
    return {"issued": True, "expires_at": token["expires_at"]}


@app.get("/kis/token/status")
def kis_token_status():
    """캐시된 토큰의 존재 여부/만료 시각/남은 시간 (신규 발급 없음)."""
    return kis.token_status()


@app.get("/kis/stock/{symbol}/price")
def kis_stock_price(symbol: str):
    if not re.fullmatch(r"\d{6}", symbol):
        raise HTTPException(status_code=422, detail="symbol must be a 6-digit stock code (e.g. 005930)")
    return kis.get_stock_price(symbol)


@app.get("/kis/stock/price", include_in_schema=False)
def kis_stock_price_legacy(code: str = Query("005930", pattern=r"^\d{6}$")):
    return kis.get_stock_price(code)


@app.get("/kis/balance")
def kis_balance(user: dict = Depends(auth.current_user)):
    """환경변수(KIS_CANO/KIS_ACNT_PRDT_CD)에 설정된 계좌의 정규화된 잔고. 로그인 필요."""
    return kis.get_balance()


class OrderRequest(BaseModel):
    symbol: str = Field(pattern=r"^\d{6}$", description="종목코드 6자리")
    side: Literal["buy", "sell"]
    order_type: Literal["limit", "market"]
    quantity: int = Field(gt=0)
    price: int | None = None

    @model_validator(mode="after")
    def check_price(self):
        if self.order_type == "limit" and (self.price is None or self.price <= 0):
            raise ValueError("price must be > 0 for limit orders")
        if self.order_type == "market":
            self.price = None  # 시장가는 단가를 무시한다
        return self


@app.post("/kis/order")
def kis_order(req: OrderRequest, user: dict = Depends(auth.current_user)):
    """국내주식 현금 매수/매도 주문 (로그인 필요). 실전 모드에서는 실제 주문이 나간다.

    KIS_ORDER_ENABLED=true 일 때만 동작한다 (기본 false, 호출 시점에 읽음).
    """
    if os.environ.get("KIS_ORDER_ENABLED", "false").strip().lower() not in ("1", "true", "yes", "on"):
        raise HTTPException(status_code=403, detail="주문 기능이 비활성화되어 있습니다 (KIS_ORDER_ENABLED=false)")
    return kis.place_order(req.symbol, req.side, req.order_type, req.quantity, req.price)


SYMBOL = Path(pattern=r"^\d{6}$", description="종목코드 6자리")
def YMD():
    """start/end 가 같은 Query 객체를 공유하면 FastAPI 가 마지막 파라미터명으로 덮어써 start 가 end 값을 읽는다 -> 호출마다 새로 만든다."""
    return Query(None, pattern=r"^\d{8}$", description="YYYYMMDD")


@app.post("/candles/{symbol}/collect")
def collect_candles(symbol: str = SYMBOL, start: str = Query(candles.DEFAULT_START, pattern=r"^\d{8}$"),
                    end: str | None = YMD(), user: dict = Depends(auth.current_user)):
    """KIS 일봉(FHKST03010100, 수정주가)을 받아 daily_candles 에 upsert 한다. 로그인 필요."""
    return candles.collect_daily_candles(symbol, start, end)


@app.get("/backtest/{symbol}/day-night")
def backtest_day_night(symbol: str = SYMBOL, start: str | None = YMD(), end: str | None = YMD()):
    """DB 일봉으로 단순보유 / 낮(장중) / 밤(종배) 누적수익률·승률·MDD 를 계산한다 (KIS 호출 없음)."""
    parse = lambda v: datetime.strptime(v, "%Y%m%d").date() if v else None
    df = candles.load_daily_candles(symbol, parse(start), parse(end))
    if len(df) < 2:
        raise HTTPException(status_code=404, detail="daily_candles 에 데이터가 부족합니다. 먼저 수집하세요.")
    result, rets = backtest.run_backtest(df)
    curve = rets[["cum_hold", "cum_day", "cum_night"]].round(6)
    return {"symbol": symbol, **result,
            "curve": [{"ts": ts.date().isoformat(), **row} for ts, row in curve.to_dict("index").items()]}


_cache: dict = {}


def _cached(name: str, fn):
    """daily_candles 가 바뀌지 않았으면 직전 계산 결과를 재사용한다 (유니버스 백테스트는 수 초 걸림).

    name 에 기간(start/end)이 들어 있으므로 기간별로 따로 캐시된다."""
    with db.connect() as conn:
        version = conn.execute("SELECT count(*), max(ts) FROM daily_candles").fetchone()
    hit = _cache.get(name)
    if hit and hit[0] == version:
        return hit[1]
    value = fn()
    _cache[name] = (version, value)
    return value


def _date_range(start: str | None, end: str | None):
    """YYYYMMDD 쿼리 -> (start, end) date. 시작이 종료보다 늦으면 422."""
    s, e = universe.parse_ymd(start), universe.parse_ymd(end)
    if s and e and s > e:
        raise HTTPException(status_code=422, detail="start 는 end 보다 늦을 수 없습니다")
    return s, e


def _curve(df) -> list[dict]:
    return [{"ts": ts.date().isoformat(), **row} for ts, row in df.round(6).to_dict("index").items()]


@app.get("/backtest/universe")
def backtest_universe(start: str | None = YMD(), end: str | None = YMD()):
    """MISSION 02: KOSPI 100 동일가중 낮/밤/보유 + 수수료 민감도 + 종목별 성과표 (DB 데이터만 사용). start/end 로 기간 지정."""
    s, e = _date_range(start, end)

    def compute():
        res = mission02.build(s, e, demo=False)
        ew = res["ew"][["hold", "day", "night"]]
        curve = (1 + ew).cumprod().add_prefix("cum_")
        table = res["table"].reset_index().astype(object).where(res["table"].reset_index().notna(), None)
        return {**mission02.summary_json(res), "curve": _curve(curve), "table": table.to_dict("records")}

    if not universe.list_symbols():
        raise HTTPException(status_code=404, detail="symbols 가 비어 있습니다. mission02.py 로 먼저 백필하세요.")
    return _cached(f"universe:{start}:{end}", compute)


@app.get("/backtest/top5")
def backtest_top5(start: str | None = YMD(), end: str | None = YMD()):
    """MISSION 03: Top-5 종배 선택 전략 리더보드 · 누적곡선 · 미래참조 점검 · 최신 후보. start/end 로 기간 지정."""
    s, e = _date_range(start, end)

    def compute():
        res = mission03.build(s, e)
        curve = (1 + res["curves"]).cumprod()
        for k, inv in res["invested"].items():  # 비용 차감용 포지션 보유 여부 (0/1)
            curve[f"inv_{k}"] = inv.reindex(curve.index).fillna(False).astype(int)
        return {
            "period": res["period"], "top_k": mission03.pf.TOP_K, "costs": mission03.pf.COSTS,
            "strategies": {k: v[0] for k, v in mission03.pf.STRATEGIES.items()},
            "board": mission03.board_json(res), "lookahead_check": res["lookahead"],
            "lookahead_cuts": res["lookahead_cuts"], "latest_date": res["latest_date"],
            "latest_picks": res["latest_picks"], "curve": _curve(curve),
        }

    if not universe.list_symbols():
        raise HTTPException(status_code=404, detail="symbols 가 비어 있습니다. mission02.py 로 먼저 백필하세요.")
    return _cached(f"top5:{start}:{end}", compute)


@app.get("/backtest/mission07")
def backtest_mission07(start: str | None = YMD(), end: str | None = YMD()):
    """Week 7 MISSION 02/03: 샘플 A~D · 강의가설 · 나만의 전략 1~3 리더보드 + 비용 0 기준 일별 수익률 (프론트가 비용 차감). start/end 로 기간 지정."""
    s, e = _date_range(start, end)
    if not universe.list_symbols():
        raise HTTPException(status_code=404, detail="symbols 가 비어 있습니다. mission02.py 로 먼저 백필하세요.")
    res = _cached(f"m07res:{start}:{end}", lambda: mission07.build(s, e))
    return _cached(f"m07:{start}:{end}", lambda: mission07.api_payload(res))


@app.get("/backtest/mission07/picks")
def backtest_mission07_picks(date: str = Query(pattern=r"^\d{8}$", description="신호일 YYYYMMDD (비거래일이면 직전 거래일)"),
                             start: str | None = YMD(), end: str | None = YMD()):
    """날짜별 종목 선정 내역: 각 전략이 그날 15:20 데이터로 뽑은 종목 · 팩터값 · 다음 거래일 밤 수익률."""
    s, e = _date_range(start, end)
    if not universe.list_symbols():
        raise HTTPException(status_code=404, detail="symbols 가 비어 있습니다. mission02.py 로 먼저 백필하세요.")
    res = _cached(f"m07res:{start}:{end}", lambda: mission07.build(s, e))
    try:
        return mission07.picks_on(res, pd.Timestamp(universe.parse_ymd(date)))
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))


@app.get("/backtest/week08")
def backtest_week08():
    """Week 8: 비ML 최고 vs ML 최고 vs 벤치마크·샘플 A~D (2020-01-02 ~ 2026-10-21 고정, 왕복 0.10%, 분할 없음).

    week08_report.py 가 만든 결과 파일을 그대로 돌려준다 (기간이 고정이라 매 요청 재계산하지 않는다)."""
    path = backtest.RESULTS_DIR / "week08_leaderboard.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="week08_leaderboard.json 이 없습니다. week08.py -> week08_ml.py -> week08_report.py 순서로 실행하세요.")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
