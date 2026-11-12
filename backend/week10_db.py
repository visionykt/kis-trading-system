"""Week 10 (3단계): 3대 세대 모델(룰베이스 / 전통 ML / 뉴럴 네트워크)의 백테스트 결과·메타데이터를 DB 에 저장하고 읽는다.

테이블 (재실행해도 같은 결과: 한 트랜잭션에서 비우고 다시 채운다)
  w10_models   모델 메타데이터 (세대, 아키텍처, 입력 shape, 총 파라미터 수, 주요 하이퍼파라미터 JSON)
  w10_metrics  지표 (샤프·누적수익률·CAGR·MDD·칼마·승률·매매일수).  protocol = three_way(구간 train/val/test) | walk_forward(구간 overall/연도)
               seed = -1 은 '대표값' (NN 은 시드 평균, 기준선은 단일 값), 0.. 은 시드별 값
  w10_history  에폭별 Train/Val Huber 손실 (3단 분할은 전 시드, Walk-Forward 는 시드 0 의 연도별 폴드)
  w10_daily    일별 순수익률 (누적곡선용: 3단 분할 = 전 구간 1,668일, Walk-Forward = 2022~2026.10, 시드 0, 게이트 없음)

사용법 (컨테이너 안):  python week10_db.py            # results/week10_nn_results.json (+ week10_curves.json) -> DB
"""
import json
from statistics import mean

import db
import week08 as w8
import week10_dataset as ds  # torch 를 import 하지 않는다 (API 컨테이너 시작이 느려지지 않게)

SCHEMA = """
CREATE TABLE IF NOT EXISTS w10_models (
    model       VARCHAR(30) PRIMARY KEY,
    generation  VARCHAR(10) NOT NULL CHECK (generation IN ('rule', 'trad_ml', 'nn')),
    title       VARCHAR(80) NOT NULL,
    arch        TEXT        NOT NULL,
    input_shape VARCHAR(30),
    params      INTEGER,
    hyper       JSONB       NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS w10_metrics (
    model             VARCHAR(30) NOT NULL REFERENCES w10_models(model) ON DELETE CASCADE,
    protocol          VARCHAR(12) NOT NULL CHECK (protocol IN ('three_way', 'walk_forward')),
    segment           VARCHAR(10) NOT NULL,
    gate              VARCHAR(20) NOT NULL,
    seed              SMALLINT    NOT NULL,
    sharpe            DOUBLE PRECISION,
    cumulative_return DOUBLE PRECISION,
    cagr              DOUBLE PRECISION,
    mdd               DOUBLE PRECISION,
    calmar            DOUBLE PRECISION,
    win_rate          DOUBLE PRECISION,
    invested_days     INTEGER,
    days              INTEGER,
    PRIMARY KEY (model, protocol, segment, gate, seed)
);
CREATE TABLE IF NOT EXISTS w10_history (
    model      VARCHAR(30) NOT NULL REFERENCES w10_models(model) ON DELETE CASCADE,
    protocol   VARCHAR(12) NOT NULL,
    fold       VARCHAR(10) NOT NULL,
    seed       SMALLINT    NOT NULL,
    epoch      SMALLINT    NOT NULL,
    train_loss DOUBLE PRECISION NOT NULL,
    val_loss   DOUBLE PRECISION NOT NULL,
    best       BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (model, protocol, fold, seed, epoch)
);
CREATE TABLE IF NOT EXISTS w10_daily (
    model      VARCHAR(30) NOT NULL REFERENCES w10_models(model) ON DELETE CASCADE,
    protocol   VARCHAR(12) NOT NULL,
    ts         DATE        NOT NULL,
    net_return DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (model, protocol, ts)
);
"""

METRIC_KEYS = ["sharpe", "cumulative_return", "cagr", "mdd", "calmar", "win_rate", "invested_days", "days"]
HEADLINE = "ma120"  # week10_nn.HEADLINE_GATE 와 같아야 한다 (save() 가 결과 파일 meta 와 대조)
RESULT_PATH = w8.RESULTS_DIR / "week10_nn_results.json"

MODELS = {  # key -> (세대, 표시 이름)
    "rule_best": ("rule", "비ML 룰베이스 최고"), "ridge": ("trad_ml", "Ridge (L2 선형회귀)"),
    "hgb": ("trad_ml", "HistGradientBoosting"), "mlp": ("nn", "MLP (2D)"), "gru": ("nn", "GRU (3D, L=20)"),
    "overfit_mlp": ("nn", "Overfit_MLP (비규제 대조군)"),
}


def model_rows(res: dict) -> list[dict]:
    import week10_nn as nn_  # SPECS(에폭·weight decay 등) 를 읽을 때만 torch 가 필요하다
    common = res["meta"]["train"]
    nn_hyper = {"optimizer": common["optimizer"], "lr": common["lr"], "weight_decay": common["weight_decay"], "batch": common["batch"],
                "loss": common["loss"], "early_stopping": common["early_stopping"], "seeds": res["meta"]["seeds"]}
    out = []
    rb = res["three_way"]["baselines"]["rule_best"]
    cfg = json.loads((w8.RESULTS_DIR / "week08_nonml_search.json").read_text(encoding="utf-8"))[0]
    out.append({"model": "rule_best", "arch": rb["desc"], "input_shape": "-", "params": 0,
                "hyper": {"factors": cfg["factors"], "weights": cfg["weights"], "filter": cfg["filter"], "k": cfg["k"], "regime": cfg["regime"],
                          "note": "학습 없음 (8주차 전 구간 탐색 1위). 표의 '게이트 없음' 값은 레짐 없이 같은 점수만 적용한 것"}})
    out.append({"model": "ridge", "arch": "Ridge(alpha=100) · 17개 순위 피처 -> y_excess", "input_shape": f"(N,{len(res['meta']['features'])})",
                "params": len(res["meta"]["features"]) + 1, "hyper": {"alpha": 100.0}})
    out.append({"model": "hgb", "arch": "HistGradientBoostingRegressor (얕은 부스팅)", "input_shape": f"(N,{len(res['meta']['features'])})", "params": None,
                "hyper": {"max_iter": 200, "learning_rate": 0.03, "max_depth": 3, "max_leaf_nodes": 7, "min_samples_leaf": 100, "l2_regularization": 10.0}})
    for k, v in res["three_way"]["models"].items():
        spec = nn_.SPECS[k]
        out.append({"model": k, "arch": v["arch"], "input_shape": v["input"], "params": v["params"],
                    "hyper": {**nn_hyper, "weight_decay": spec["wd"], "early_stopping_on": spec["early_stop"], "max_epochs": spec["epochs"],
                              "mode": spec["mode"]}})
    return out


def metric_rows(res: dict) -> list[tuple]:
    rows = []

    def add(model, protocol, segment, gate, seed, m):
        rows.append((model, protocol, segment, gate, seed, *[m.get(k) for k in METRIC_KEYS]))

    def add_mean(model, protocol, segment, gate, ms):
        add(model, protocol, segment, gate, -1, {k: (mean(x[k] for x in ms) if ms[0].get(k) is not None else None) for k in METRIC_KEYS})

    tw = res["three_way"]
    for name, v in tw["models"].items():
        for g in ("none", HEADLINE):
            for seg in ("train", "val", "test"):
                ms = [r["report"][g][seg] for r in v["runs"]]
                for r in v["runs"]:
                    add(name, "three_way", seg, g, r["seed"], r["report"][g][seg])
                add_mean(name, "three_way", seg, g, ms)
    for name, v in tw["baselines"].items():
        for g in ("none", HEADLINE):
            for seg in ("train", "val", "test"):
                add(name, "three_way", seg, g, -1, v[g][seg])
    if "own_gate" in tw["baselines"]["rule_best"]:  # 룰베이스 본래 레짐(8주차 최적)도 참고로
        og = tw["baselines"]["rule_best"]["own_gate"]
        for seg in ("train", "val", "test"):
            add("rule_best", "three_way", seg, "own:" + og["gate"], -1, og[seg])

    wf = res["walk_forward"]
    for name, v in wf["models"].items():
        seeds = sorted(v["seeds"])
        for g in v["seeds"][seeds[0]]:
            for seg in ["overall"] + list(v["seeds"][seeds[0]][g]["per_year"]):
                get = lambda s: (v["seeds"][s][g]["overall"] if seg == "overall" else v["seeds"][s][g]["per_year"][seg])
                for s in seeds:
                    add(name, "walk_forward", seg, g, int(s.replace("seed", "")), get(s))
                add_mean(name, "walk_forward", seg, g, [get(s) for s in seeds])
    for name, v in wf["baselines"].items():
        for g in v:
            add(name, "walk_forward", "overall", g, -1, v[g]["overall"])
            for y, m in v[g]["per_year"].items():
                add(name, "walk_forward", y, g, -1, m)
    return rows


def history_rows(res: dict) -> list[tuple]:
    rows = []

    def add(model, protocol, fold, seed, h):
        for e, (tl, vl) in enumerate(zip(h["train_loss"], h["val_loss"]), 1):
            rows.append((model, protocol, fold, seed, e, tl, vl, e == h["best_epoch"]))

    for name, v in res["three_way"]["models"].items():
        for r in v["runs"]:
            add(name, "three_way", "all", r["seed"], r["history"])
    for name, v in res["walk_forward"]["models"].items():
        for y, h in v["history_seed0"].items():
            add(name, "walk_forward", str(y), 0, h)
    return rows


def daily_rows() -> list[tuple]:
    path = w8.RESULTS_DIR / "week10_curves.json"
    if not path.exists():
        return []
    c = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for protocol in ("three_way", "walk_forward"):
        for model, rets in c["returns"].get(protocol, {}).items():
            rows += [(model, protocol, d, r) for d, r in zip(c["dates"][protocol], rets)]
    return rows


def save(res: dict | None = None) -> dict:
    res = res or json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert res["meta"]["headline_gate"] == HEADLINE, "헤드라인 게이트가 결과 파일과 다르다"
    models, metrics, hist, daily = model_rows(res), metric_rows(res), history_rows(res), daily_rows()
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA)
            cur.execute("TRUNCATE w10_daily, w10_history, w10_metrics, w10_models CASCADE")
            cur.executemany("INSERT INTO w10_models (model, generation, title, arch, input_shape, params, hyper) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                            [(m["model"], MODELS[m["model"]][0], MODELS[m["model"]][1], m["arch"], m["input_shape"], m["params"], json.dumps(m["hyper"], ensure_ascii=False))
                             for m in models])
            cur.executemany(f"INSERT INTO w10_metrics (model, protocol, segment, gate, seed, {', '.join(METRIC_KEYS)}) VALUES ({', '.join(['%s'] * (5 + len(METRIC_KEYS)))})", metrics)
            cur.executemany("INSERT INTO w10_history (model, protocol, fold, seed, epoch, train_loss, val_loss, best) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)", hist)
            cur.executemany("INSERT INTO w10_daily (model, protocol, ts, net_return) VALUES (%s,%s,%s,%s)", daily)
    return {"models": len(models), "metrics": len(metrics), "history": len(hist), "daily": len(daily)}


# ------------------------------------------------------------------ 읽기 (API / 대시보드)

def payload() -> dict:
    """대시보드용: 모델 메타 + 지표표 + 에폭 손실 + 누적수익률 곡선(일별 수익률로 복리 누적)."""
    import numpy as np
    with db.connect() as conn:
        mods = conn.execute("SELECT model, generation, title, arch, input_shape, params, hyper FROM w10_models ORDER BY model").fetchall()
        mets = conn.execute(f"SELECT model, protocol, segment, gate, seed, {', '.join(METRIC_KEYS)} FROM w10_metrics").fetchall()
        hist = conn.execute("SELECT model, protocol, fold, seed, epoch, train_loss, val_loss, best FROM w10_history ORDER BY model, protocol, fold, seed, epoch").fetchall()
        daily = conn.execute("SELECT model, protocol, ts, net_return FROM w10_daily ORDER BY protocol, model, ts").fetchall()
    metrics = [dict(zip(["model", "protocol", "segment", "gate", "seed"] + METRIC_KEYS, r)) for r in mets]
    history: dict = {}
    for m, p, f, s, e, tl, vl, b in hist:
        h = history.setdefault(f"{m}|{p}|{f}|{s}", {"model": m, "protocol": p, "fold": f, "seed": s, "train": [], "val": [], "best_epoch": None})
        h["train"].append(tl)
        h["val"].append(vl)
        if b:
            h["best_epoch"] = e
    curves: dict = {}
    for m, p, ts, r in daily:
        c = curves.setdefault(p, {}).setdefault(m, {"dates": [], "net": []})
        c["dates"].append(str(ts))
        c["net"].append(r)
    for p in curves:
        for m, c in curves[p].items():
            c["cum"] = (np.cumprod(1 + np.array(c["net"])) - 1).round(5).tolist()
            del c["net"]
    return {"models": [dict(zip(["model", "generation", "title", "arch", "input_shape", "params", "hyper"], r)) for r in mods],
            "metrics": metrics, "history": list(history.values()), "curves": curves,
            "headline_gate": HEADLINE, "ranges": ds.THREE_WAY}


if __name__ == "__main__":
    print("DB 저장:", save())
