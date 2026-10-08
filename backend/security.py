"""비밀번호 해시(bcrypt)와 JWT 발급/검증."""
import os
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "60"))
MAX_PASSWORD_BYTES = 72  # bcrypt 입력 한도


def _secret() -> str:
    secret = os.environ.get("JWT_SECRET", "")
    if len(secret) < 32:
        raise RuntimeError("JWT_SECRET 환경변수가 없거나 32자 미만입니다 (.env에 설정하세요).")
    return secret


def hash_password(password: str) -> str:
    raw = password.encode()
    if len(raw) > MAX_PASSWORD_BYTES:
        raise ValueError("비밀번호는 72바이트를 넘을 수 없습니다.")
    return bcrypt.hashpw(raw, bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    raw = password.encode()
    if len(raw) > MAX_PASSWORD_BYTES:
        return False
    try:
        return bcrypt.checkpw(raw, password_hash.encode())
    except ValueError:
        return False


# 존재하지 않는 아이디로 로그인할 때도 해시 비교 시간을 맞추기 위한 더미 해시
DUMMY_HASH = hash_password("dummy-password-for-timing")


def create_access_token(user_id: int, username: str, role: str) -> tuple[str, int]:
    expires = timedelta(minutes=JWT_EXPIRE_MINUTES)
    now = datetime.now(timezone.utc)
    payload = {"sub": str(user_id), "username": username, "role": role, "iat": now, "exp": now + expires}
    return jwt.encode(payload, _secret(), algorithm=JWT_ALGORITHM), int(expires.total_seconds())


def decode_access_token(token: str) -> dict:
    """만료면 jwt.ExpiredSignatureError, 그 외 불량 토큰이면 jwt.InvalidTokenError."""
    return jwt.decode(token, _secret(), algorithms=[JWT_ALGORITHM], options={"require": ["exp", "sub"]})
