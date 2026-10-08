"""사용자/계좌 스키마 생성과 초기 관리자 시드."""
import os

import psycopg

import security

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      VARCHAR(50)  NOT NULL UNIQUE,
    password_hash VARCHAR(100) NOT NULL,
    role          VARCHAR(10)  NOT NULL DEFAULT 'user' CHECK (role IN ('admin', 'user')),
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS accounts (
    id             SERIAL PRIMARY KEY,
    account_number VARCHAR(20)  NOT NULL UNIQUE,
    account_name   VARCHAR(100) NOT NULL,
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS user_accounts (
    user_id    INTEGER NOT NULL REFERENCES users(id)    ON DELETE CASCADE,
    account_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, account_id)
);
CREATE INDEX IF NOT EXISTS idx_user_accounts_account ON user_accounts(account_id);
"""


def connect() -> psycopg.Connection:
    return psycopg.connect(os.environ["DATABASE_URL"])


def init_db() -> None:
    admin_password = os.environ.get("ADMIN_INITIAL_PASSWORD", "admin1234")
    with connect() as conn:
        conn.execute(SCHEMA)
        # 이미 admin이 있으면 건드리지 않는다 (재시작 시 비밀번호가 덮어써지지 않도록)
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, 'admin') "
            "ON CONFLICT (username) DO NOTHING",
            ("admin", security.hash_password(admin_password)),
        )
