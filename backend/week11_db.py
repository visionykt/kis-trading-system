"""Week 11: week11_analysis.py 가 DB(w11_*)에 저장한 결과를 대시보드용 JSON 으로 읽는다.

저장은 week11_analysis.py (save_db) 가 한다.  여기서는 읽기만 한다 (torch 를 import 하지 않아 API 컨테이너 시작이 느려지지 않는다).
metrics 는 용량을 줄이려고 {"cols": [...], "rows": [[...], ...]} 형태로 돌려준다 (소수 5자리 반올림).
"""
import db

M_COLS = ["section", "variant", "model", "protocol", "segment", "gate", "k", "cost_bp", "fill", "seed", "sharpe", "cumulative_return",
          "mdd", "calmar", "win_rate", "invested_days", "days", "ic", "ic_ir", "ic_hit"]


def _r(v):
    return round(v, 5) if isinstance(v, float) else v


def payload() -> dict:
    with db.connect() as conn:
        mets = conn.execute(f"SELECT {', '.join(M_COLS)} FROM w11_metrics ORDER BY id").fetchall()
        plc = conn.execute("SELECT model, protocol, gate, kind, actual_sharpe, n, p_mean, p_std, p95, p99, p_value, z FROM w11_placebo ORDER BY model, protocol, gate, kind").fetchall()
        vif = conn.execute("SELECT scope, feature, vif_daily, vif_pooled FROM w11_feature_vif ORDER BY scope, vif_daily DESC").fetchall()
        meta = dict(conn.execute("SELECT key, value FROM w11_meta").fetchall())
    P = ["model", "protocol", "gate", "kind", "actual_sharpe", "n", "p_mean", "p_std", "p95", "p99", "p_value", "z"]
    return {"metrics": {"cols": M_COLS, "rows": [[_r(v) for v in r] for r in mets]},
            "placebo": [dict(zip(P, [_r(v) for v in r])) for r in plc],
            "vif": [dict(zip(["scope", "feature", "vif_daily", "vif_pooled"], [_r(v) for v in r])) for r in vif],
            "config": meta.get("config"), "fill_stats": meta.get("fill_stats"), "feature_summary": meta.get("feature_summary"),
            "repro_check": meta.get("repro_check"), "train_log": meta.get("train_log")}
