"""한국투자증권(KIS) Open API 클라이언트 (실전투자)."""
import json
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path

import requests

BASE_URL_REAL = "https://openapi.koreainvestment.com:9443"
BASE_URL_MOCK = "https://openapivts.koreainvestment.com:29443"


def is_mock() -> bool:
    """KIS_ENV=mock 이면 모의투자 서버/tr_id 를 사용한다 (기본: 실전). 모의투자는 별도 앱키가 필요하다."""
    return os.environ.get("KIS_ENV", "real").lower() in ("mock", "paper", "vts")


def base_url() -> str:
    return BASE_URL_MOCK if is_mock() else BASE_URL_REAL


def balance_tr_id() -> str:
    return "VTTC8434R" if is_mock() else "TTTC8434R"


# 토큰 발급은 1분당 1회로 제한되고 유효기간이 24시간이라 파일로 캐시한다.
# (uvicorn --reload 재시작에도 유지되도록 컨테이너 /tmp 사용, 호스트 폴더에는 남기지 않음)
TOKEN_CACHE = Path("/tmp/kis_token.json")

# 만료 REFRESH_MARGIN 전까지는 캐시 토큰을 그대로 사용하고, 그 이후에만 재발급한다.
REFRESH_MARGIN = timedelta(minutes=5)

_lock = threading.Lock()
_memory: dict | None = None  # 프로세스 내 1차 캐시 (파일 읽기 생략)


class KISError(Exception):
    def __init__(self, status: int, body):
        super().__init__(f"KIS API error {status}: {body}")
        self.status = status
        self.body = body


def _app_key() -> str:
    return os.environ["KIS_APP_KEY"]


def _app_secret() -> str:
    return os.environ["KIS_APP_SECRET"]


def _is_valid(data: dict | None) -> bool:
    """만료 5분 전 이전이면 재사용 가능."""
    if not data:
        return False
    try:
        return datetime.fromisoformat(data["expires_at"]) - REFRESH_MARGIN > datetime.now()
    except (KeyError, ValueError, TypeError):
        return False


def _read_cache() -> dict | None:
    """메모리 -> 파일 순으로 캐시를 읽는다 (만료 여부와 무관, 손상 시 None)."""
    global _memory
    if _memory:
        return _memory
    try:
        data = json.loads(TOKEN_CACHE.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or "access_token" not in data or "expires_at" not in data:
        return None
    _memory = data
    return data


def _load_cached_token() -> dict | None:
    data = _read_cache()
    return data if _is_valid(data) else None


def _save_cache(data: dict) -> None:
    global _memory
    tmp = TOKEN_CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.chmod(0o600)
    tmp.replace(TOKEN_CACHE)  # 원자적 교체
    _memory = data


def clear_cache() -> None:
    global _memory
    with _lock:
        _memory = None
        TOKEN_CACHE.unlink(missing_ok=True)


def issue_token() -> dict:
    """POST /oauth2/tokenP 로 접근 토큰을 새로 발급받아 캐시한다."""
    resp = requests.post(
        f"{base_url()}/oauth2/tokenP",
        json={"grant_type": "client_credentials", "appkey": _app_key(), "appsecret": _app_secret()},
        timeout=10,
    )
    body = resp.json()
    if resp.status_code != 200 or "access_token" not in body:
        raise KISError(resp.status_code, body)
    data = {
        "access_token": body["access_token"],
        "expires_at": (datetime.now() + timedelta(seconds=int(body["expires_in"]))).isoformat(),
    }
    _save_cache(data)
    return data


def get_token() -> dict:
    with _lock:
        return _load_cached_token() or issue_token()


def token_status() -> dict:
    """캐시 상태만 조회한다. 발급 요청은 보내지 않고 토큰 값도 노출하지 않는다."""
    with _lock:
        data = _read_cache()
    if not data:
        return {"cached": False, "valid": False, "expires_at": None, "remaining_seconds": None, "remaining": None}
    expires_at = datetime.fromisoformat(data["expires_at"])
    remaining = max(0, int((expires_at - datetime.now()).total_seconds()))
    return {
        "cached": True,
        "valid": _is_valid(data),  # False 면 다음 호출 때 재발급됨
        "expires_at": data["expires_at"],
        "refresh_at": (expires_at - REFRESH_MARGIN).isoformat(),
        "remaining_seconds": remaining,
        "remaining": str(timedelta(seconds=remaining)),
    }


def _get(path: str, tr_id: str, params: dict) -> dict:
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {get_token()['access_token']}",
        "appkey": _app_key(),
        "appsecret": _app_secret(),
        "tr_id": tr_id,
        "custtype": "P",
    }
    resp = requests.get(f"{base_url()}{path}", headers=headers, params=params, timeout=10)
    body = resp.json()
    if resp.status_code != 200 or body.get("rt_cd") != "0":
        raise KISError(resp.status_code, body)
    return body


def inquire_price(code: str) -> dict:
    """국내주식 현재가 시세 (tr_id FHKST01010100) 원본 output."""
    body = _get(
        "/uapi/domestic-stock/v1/quotations/inquire-price",
        "FHKST01010100",
        {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code},
    )
    return body["output"]


def get_stock_price(code: str) -> dict:
    """현재가 시세를 핵심 항목만 정규화해서 반환한다."""
    out = inquire_price(code)
    try:
        return {
            "code": code,
            "price": int(out["stck_prpr"]),
            "change": int(out["prdy_vrss"]),
            "change_rate": float(out["prdy_ctrt"]),
            "volume": int(out["acml_vol"]),
            "open": int(out["stck_oprc"]),
            "high": int(out["stck_hgpr"]),
            "low": int(out["stck_lwpr"]),
        }
    except (KeyError, ValueError, TypeError) as e:
        raise KISError(200, {"msg1": f"unexpected price response: {e!r}", "output": out})


def inquire_balance() -> dict:
    """국내주식 잔고 조회 원본 (실전 TTTC8434R / 모의 VTTC8434R)."""
    body = _get(
        "/uapi/domestic-stock/v1/trading/inquire-balance",
        balance_tr_id(),
        {
            "CANO": os.environ["KIS_CANO"],
            "ACNT_PRDT_CD": os.environ["KIS_ACNT_PRDT_CD"],
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        },
    )
    return {"holdings": body["output1"], "summary": body["output2"]}


def _num(v, cast=int):
    """KIS 는 모든 값을 문자열로 준다. 빈 값은 0 으로 취급."""
    return cast(v) if v not in (None, "") else cast(0)


def get_balance() -> dict:
    """잔고 조회 결과를 summary / holdings 로 정규화한다."""
    raw = inquire_balance()
    try:
        s = (raw["summary"] or [{}])[0]
        purchase = _num(s.get("pchs_amt_smtl_amt"))
        profit = _num(s.get("evlu_pfls_smtl_amt"))
        summary = {
            "total_evaluation": _num(s.get("tot_evlu_amt")),  # 총평가금액
            "cash": _num(s.get("dnca_tot_amt")),  # 예수금
            "orderable_cash": _num(s.get("prvs_rcdl_excc_amt")),  # 가수도정산금액(D+2 예수금)
            "stock_evaluation": _num(s.get("scts_evlu_amt")),  # 유가평가금액
            "purchase_amount": purchase,  # 매입금액 합계
            "profit_loss": profit,  # 총손익금액
            "profit_rate": round(profit / purchase * 100, 2) if purchase else 0.0,  # 총수익률(%)
        }
        holdings = [
            {
                "code": h["pdno"],
                "name": h["prdt_name"],
                "quantity": _num(h["hldg_qty"]),
                "avg_price": _num(h["pchs_avg_pric"], float),
                "price": _num(h["prpr"]),
                "profit_loss": _num(h["evlu_pfls_amt"]),
                "profit_rate": _num(h["evlu_pfls_rt"], float),
            }
            for h in raw["holdings"]
            if _num(h.get("hldg_qty")) > 0
        ]
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        raise KISError(200, {"msg1": f"unexpected balance response: {e!r}"})
    return {"summary": summary, "holdings": holdings}
