import os
import re
from contextlib import asynccontextmanager

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import auth
import db
import kis


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
