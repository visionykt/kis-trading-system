import os

import psycopg
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import kis

app = FastAPI(title="KIS Data Mining API")
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


@app.get("/kis/stock/price")
def kis_stock_price(code: str = Query("005930", pattern=r"^\d{6}$", description="종목코드 6자리")):
    out = kis.inquire_price(code)
    return {
        "code": code,
        "price": int(out["stck_prpr"]),
        "change": int(out["prdy_vrss"]),
        "change_rate": float(out["prdy_ctrt"]),
        "open": int(out["stck_oprc"]),
        "high": int(out["stck_hgpr"]),
        "low": int(out["stck_lwpr"]),
        "volume": int(out["acml_vol"]),
        "raw": out,
    }


@app.get("/kis/account/balance")
def kis_account_balance():
    return kis.inquire_balance()
