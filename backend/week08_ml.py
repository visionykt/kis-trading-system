"""Week 8 MISSION 02: 머신러닝 종가 배팅 — 2020-01-02 ~ 2026-10-21 전 구간 학습(No Split) 후 같은 구간 샤프 극대화.

입력 X_t = t 일 장 마감까지 알 수 있는 파생 팩터(미래 참조 없음), 정답 y_t = 그날 밤 수익률 O(t+1)/C(t) - 1.
매일 모델이 종목별 예측 점수를 내면 상위 K 종목을 종가에 사서 다음 날 시가에 판다 (신호는 t 일, 수익은 t+1 일 = shift(1)).

9주차(다른 기간 적용)를 위해 학습과 평가를 분리했다.
  train_model(name, X, y)           -> 학습된 모델
  evaluate_model(bundle, panel, ..) -> 저장된 모델을 임의의 기간에 그대로 적용 (재학습 없음)
저장: backend/models/week08_best_model.joblib (모델 + 피처 목록 + K/레짐/필터 + 중요 팩터 상위 5개)

사용법 (컨테이너 안):  python week08_ml.py
"""
import json
import math
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import universe
import week08 as w8
from backtest import RESULTS_DIR

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "week08_best_model.joblib"

BASE = ["close_pos", "upper_shadow", "body_ratio", "vol_ratio", "vol_surge_up", "turnover_rank", "ma20_gap", "ret_5d",
        "ret_20d", "range_pct", "r_day", "gap_today", "rsi_14", "night_win_20", "night_mean_20", "night_sharpe_20",
        "night_sharpe_60", "mkt_night_20", "mkt_ma20_gap", "mkt_ma120_gap"]
RANKED = ["vol_ratio", "night_sharpe_20", "night_sharpe_60", "night_mean_20", "r_day", "ret_5d"]  # 같은 날 횡단면 순위 버전
FEATURES = BASE + [f"rk_{f}" for f in RANKED]
KS = [3, 5, 7, 10]
FILTERS = ["none", "양봉"]


# ------------------------------------------------------------------ 데이터셋

def feature_cube(eng: w8.Engine) -> np.ndarray:
    """(날짜, 종목, 피처) float32. 모든 값은 t 일 장 마감까지의 정보."""
    cols = [eng.F[f].to_numpy() for f in BASE] + [eng.rank(f) for f in RANKED]
    return np.stack(cols, axis=2).astype(np.float32)


def target_matrices(eng: w8.Engine) -> tuple[np.ndarray, np.ndarray]:
    """y_raw(±10% 로 자름), y_rank(같은 날 횡단면 백분위 - 0.5)."""
    y = eng.nxt
    return np.clip(y, -0.1, 0.1), pd.DataFrame(y).rank(axis=1, pct=True).to_numpy() - 0.5


def training_rows(cube: np.ndarray, y: np.ndarray, t_lo: int, t_hi: int) -> tuple[np.ndarray, np.ndarray]:
    """t_lo..t_hi(포함) 의 (X, y) 샘플 쌍. 피처/정답이 하나라도 NaN 이면 제외."""
    X = cube[t_lo:t_hi + 1].reshape(-1, cube.shape[2])
    yy = y[t_lo:t_hi + 1].reshape(-1)
    ok = np.isfinite(X).all(1) & np.isfinite(yy)
    return X[ok], yy[ok]


# ------------------------------------------------------------------ 모델 (강하게 정규화: 전 구간 학습이라 과적합 방지)

def make_model(name: str):
    if name == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=300, subsample=0.7,
                             subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0, n_jobs=8, verbose=-1, random_state=0)
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=300,
                                             l2_regularization=10.0, random_state=0)
    if name == "rf":
        return RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=300, max_features=0.5, n_jobs=8,
                                     random_state=0)
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1000.0))
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-2, early_stopping=True,
                                                            validation_fraction=0.1, max_iter=60, random_state=0))
    raise ValueError(name)


MODELS = ["lgbm", "hgb", "rf", "ridge", "mlp"]
TARGETS = ["rank", "raw"]


def train_model(name: str, X: np.ndarray, y: np.ndarray):
    """학습(fit) 단계: 전 구간 (X_t, y_t) 쌍을 통째로 학습."""
    model = make_model(name)
    model.fit(X, y)
    return model


def predict_scores(model, cube: np.ndarray, eng: w8.Engine) -> np.ndarray:
    """예측(predict) 단계: 모든 (날짜, 종목)의 예측 점수. 피처에 NaN 이 있으면 NaN(후보 아님)."""
    T, N, F = cube.shape
    flat = cube.reshape(-1, F)
    ok = np.isfinite(flat).all(1)
    out = np.full(T * N, np.nan)
    if ok.any():
        out[ok] = model.predict(flat[ok])
    return out.reshape(T, N)


def importance_top(model, name: str, X: np.ndarray, y: np.ndarray, top: int = 5) -> list[dict]:
    """중요 팩터 상위 top 개 (트리는 내장 중요도, 나머지는 순열 중요도)."""
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, float)
        kind = "내장 중요도(분할 이득)"
    else:
        rng = np.random.default_rng(0)
        sub = rng.choice(len(X), min(20000, len(X)), replace=False)
        imp = permutation_importance(model, X[sub], y[sub], n_repeats=3, random_state=0, n_jobs=4).importances_mean
        kind = "순열 중요도"
    imp = np.maximum(imp, 0)
    tot = imp.sum() or 1.0
    order = np.argsort(-imp)[:top]
    return [{"feature": FEATURES[i], "importance": float(imp[i] / tot), "kind": kind} for i in order]


# ------------------------------------------------------------------ 평가 (기간 분리)

def run_scores(eng: w8.Engine, scores: np.ndarray, k: int, flt: str, gate: str) -> dict:
    sc = np.where(w8.filters(eng)[flt], scores, np.nan)
    r, inv = eng.pick(sc, k)
    g = eng.gates[gate]
    return {"r": r * g, "inv": inv & g}


def evaluate_model(bundle: dict, panel: dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """저장된 모델을 [start, end] 에 그대로 적용 (재학습 없음). panel 은 start 앞 ~150거래일 워밍업을 포함해야 한다."""
    eng = w8.Engine(panel, start=start, end=end)
    scores = predict_scores(bundle["model"], feature_cube(eng), eng)
    run = run_scores(eng, scores, bundle["k"], bundle["filter"], bundle["regime"])
    return {"days": eng.days, **run, "metrics": w8.fast_metrics(run["r"], run["inv"])}


def walk_forward(name: str, target: str, eng: w8.Engine, cube, y, k: int, flt: str, gate: str,
                 first_test_year: int = 2022) -> dict:
    """정직한 참고치: 각 해 Y 를 Y 이전 데이터만으로 학습한 모델로 예측 (공식 모델은 전 구간 학습이라 이 값과 다르다)."""
    years = eng.days.year.to_numpy()
    rs, invs = [], []
    for yr in range(first_test_year, int(years.max()) + 1):
        pos = np.where(years == yr)[0] + eng.t0
        tr_hi = pos[0] - 2  # t 일 정답이 t+1 시가라서 경계 하루를 비운다
        X, yy = training_rows(cube, y, eng.t0, tr_hi)
        model = train_model(name, X, yy)
        sub = cube[pos[0]:pos[-1] + 1]
        T, N, F = sub.shape
        flat = sub.reshape(-1, F)
        ok = np.isfinite(flat).all(1)
        out = np.full(T * N, np.nan)
        out[ok] = model.predict(flat[ok])
        full = np.full(cube.shape[:2], np.nan)
        full[pos[0]:pos[-1] + 1] = out.reshape(T, N)
        res = run_scores(eng, full, k, flt, gate)
        sel = slice(pos[0] - eng.t0, pos[-1] - eng.t0 + 1)
        rs.append(res["r"][sel])
        invs.append(res["inv"][sel])
    r, inv = np.concatenate(rs), np.concatenate(invs)
    return {"sharpe": w8.sharpe(r, inv), "days": len(r), "test_from": first_test_year}


def main() -> None:
    t_all = datetime.now()
    panel = universe.load_panel(end=w8.END.date())
    eng = w8.Engine(panel)
    cube = feature_cube(eng)
    y_raw, y_rank = target_matrices(eng)
    ys = {"raw": y_raw, "rank": y_rank}
    print(f"피처 {len(FEATURES)}개 · 평가 {eng.days[0].date()} ~ {eng.days[-1].date()} ({len(eng.days)}일)")

    results, fitted = [], {}
    for name in MODELS:
        for target in TARGETS:
            t = datetime.now()
            X, yy = training_rows(cube, ys[target], eng.t0, eng.t1)  # 전 구간 (분할 없음)
            model = train_model(name, X, yy)
            scores = predict_scores(model, cube, eng)
            best = None
            for flt in FILTERS:
                for k in KS:
                    for gate in eng.gates:
                        run = run_scores(eng, scores, k, flt, gate)
                        sh = w8.sharpe(run["r"], run["inv"])
                        if best is None or sh > best["sharpe"]:
                            best = {"sharpe": sh, "k": k, "filter": flt, "regime": gate}
            plain = {k: run_scores(eng, scores, k, "none", "none") for k in KS}  # 필터·레짐 없이 모델 점수만 쓴 경우
            none_best = max(({"sharpe": w8.sharpe(v["r"], v["inv"]), "k": k} for k, v in plain.items()), key=lambda d: d["sharpe"])
            row = {"model": name, "target": target, "rows": len(X), **best, "no_gate_best": none_best,
                   "seconds": round((datetime.now() - t).total_seconds(), 1)}
            results.append(row)
            fitted[(name, target)] = (model, X, yy)
            print(f"{name:6} {target:5} 학습행 {len(X):,} · 전 구간 최고 샤프 {best['sharpe']:.3f} "
                  f"(K={best['k']}, {best['filter']}, {best['regime']}) · 게이트없음 {none_best['sharpe']:.3f} · {row['seconds']}s")

    results.sort(key=lambda r: -r["sharpe"])
    # 모델별 정직한 참고치 (walk-forward) — 상위 3개 설정만
    for row in results[:3]:
        wf = walk_forward(row["model"], row["target"], eng, cube, ys[row["target"]], row["k"], row["filter"], row["regime"])
        row["walk_forward"] = wf
        print(f"  walk-forward {row['model']}/{row['target']}: {wf['test_from']}년~ 샤프 {wf['sharpe']:.3f} ({wf['days']}일)")

    top = results[0]
    model, X, yy = fitted[(top["model"], top["target"])]
    imp = importance_top(model, top["model"], X, yy)
    bundle = {"model": model, "model_name": top["model"], "target": top["target"], "features": FEATURES,
              "k": top["k"], "filter": top["filter"], "regime": top["regime"], "cost": w8.COST,
              "train_range": [str(eng.days[0].date()), str(eng.days[-1].date())], "feature_importance_top5": imp,
              "saved_at": datetime.now().isoformat(timespec="seconds")}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "week08_ml_search.json").write_text(
        json.dumps({"results": results, "best": {k: v for k, v in bundle.items() if k != "model"}}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n최고: {top['model']}({top['target']}) K={top['k']} {top['filter']} 레짐 {top['regime']} 샤프 {top['sharpe']:.3f}")
    print("중요 팩터 상위 5:", ", ".join(f"{d['feature']} {d['importance']:.1%}" for d in imp))
    print(f"저장: {MODEL_PATH} · 총 {(datetime.now() - t_all).seconds}s")


if __name__ == "__main__":
    main()
