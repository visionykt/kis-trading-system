"""로그인 및 권한 검증 의존성, 계좌 조회 API.

  401 Unauthorized  : 아이디/비밀번호 불일치, 토큰 없음/위조
  401 Token Expired : 토큰 유효시간 만료
  403 Forbidden     : 권한 부족 (타인 계좌 접근 등)
"""
import jwt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

import db
import security

router = APIRouter()
bearer = HTTPBearer(auto_error=False)


def _unauthorized(message: str, code: str) -> HTTPException:
    return HTTPException(status_code=401, detail={"error": code, "message": message}, headers={"WWW-Authenticate": "Bearer"})


def _forbidden(message: str) -> HTTPException:
    return HTTPException(status_code=403, detail={"error": "Forbidden", "message": message})


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=72)


@router.post("/login")
def login(body: LoginRequest):
    with db.connect() as conn:
        row = conn.execute("SELECT id, password_hash, role FROM users WHERE username = %s", (body.username,)).fetchone()
    # 아이디가 없어도 동일하게 해시 비교를 수행해 응답 시간/메시지로 존재 여부가 드러나지 않게 한다.
    ok = security.verify_password(body.password, row[1] if row else security.DUMMY_HASH)
    if not row or not ok:
        raise _unauthorized("아이디 또는 비밀번호가 올바르지 않습니다.", "Unauthorized")
    token, expires_in = security.create_access_token(row[0], body.username, row[2])
    return {"access_token": token, "token_type": "bearer", "expires_in": expires_in, "role": row[2]}


def current_user(cred: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
    if cred is None:
        raise _unauthorized("인증 토큰이 필요합니다.", "Unauthorized")
    try:
        payload = security.decode_access_token(cred.credentials)
    except jwt.ExpiredSignatureError:
        raise _unauthorized("토큰이 만료되었습니다. 다시 로그인하세요.", "Token Expired")
    except jwt.InvalidTokenError:
        raise _unauthorized("유효하지 않은 토큰입니다.", "Unauthorized")
    return {"id": int(payload["sub"]), "username": payload["username"], "role": payload["role"]}


def require_admin(user: dict = Depends(current_user)) -> dict:
    if user["role"] != "admin":
        raise _forbidden("관리자 권한이 필요합니다.")
    return user


def _fmt(r) -> dict:
    return {"id": r[0], "account_number": r[1], "account_name": r[2], "created_at": r[3]}


@router.get("/accounts")
def list_accounts(user: dict = Depends(current_user)):
    """admin은 전체, user는 본인에게 매핑된 계좌만 반환한다."""
    with db.connect() as conn:
        if user["role"] == "admin":
            rows = conn.execute("SELECT id, account_number, account_name, created_at FROM accounts ORDER BY id").fetchall()
        else:
            rows = conn.execute(
                "SELECT a.id, a.account_number, a.account_name, a.created_at FROM accounts a "
                "JOIN user_accounts ua ON ua.account_id = a.id WHERE ua.user_id = %s ORDER BY a.id",
                (user["id"],),
            ).fetchall()
    return [_fmt(r) for r in rows]


@router.get("/accounts/{account_id}")
def get_account(account_id: int, user: dict = Depends(current_user)):
    """user가 매핑되지 않은 계좌를 지정하면 403 (계좌 존재 여부도 노출하지 않는다)."""
    with db.connect() as conn:
        if user["role"] != "admin":
            mapped = conn.execute(
                "SELECT 1 FROM user_accounts WHERE user_id = %s AND account_id = %s", (user["id"], account_id)
            ).fetchone()
            if not mapped:
                raise _forbidden("해당 계좌에 대한 접근 권한이 없습니다.")
        row = conn.execute(
            "SELECT id, account_number, account_name, created_at FROM accounts WHERE id = %s", (account_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail={"error": "Not Found", "message": "계좌가 없습니다."})
    return _fmt(row)
