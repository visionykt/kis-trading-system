"""Week 11 (2~4단계): 교정 비교 · Ridge+GRU 앙상블 · 체결/비용/Placebo 강건성 · 피처 상관/VIF -> results/week11_results.json (+ DB).

입력: week11_train.py 가 cache_w11/ 에 저장한 점수 행렬 (T_full, N).  이 스크립트는 학습하지 않는다.
공통 규칙(8~10주차와 동일): t 일 종가 매수 -> t+1 일 시가 매도, 기본 왕복 0.10%, Top-K 동일비중, 현금일 포함, 샤프 = 일평균/표준편차 x sqrt(252).
  방식 A = 3단 분할 (train/val/test)  ·  방식 B = 2022~2026 연도별 확장형 Walk-Forward (overall + 연도별)
행 키: section · variant · model · protocol · segment · gate · k · cost_bp · fill · seed (seed -1 = 시드 평균 지표)

사용법 (컨테이너 안):  python week11_analysis.py [--no-db]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import week08 as w8
import week10_dataset as ds
from backtest import RESULTS_DIR, metrics as bt_metrics
from week11_train import CACHE

OUT = RESULTS_DIR / "week11_results.json"
GATES = ["none", "ma120", "night40"]  # night40 = 8주차 룰베이스 최고의 '본래' 레짐 (모든 모델에 동일 적용해 공정 비교)
KS = [5, 7]
COSTS_BP = [0, 10, 20, 30]
IC_SEEDS, LOSS_SEEDS = [0, 1, 2, 3, 4], [0, 1, 2]
N_PERM = 1000
LIMIT_UP = 0.29  # 종가 등락률 +29% 이상 = 상한가 마감 (KRX 가격제한폭 30%, 호가단위 절사 고려)


# ------------------------------------------------------------------ 엔진 보조

def drank(S: np.ndarray) -> np.ndarray:
    """날짜별 횡단면 백분위 순위 (NaN 유지)."""
    return pd.DataFrame(S.astype(float)).rank(axis=1, pct=True).to_numpy()


def pick_arr(S: np.ndarray, X: np.ndarray, k: int, blocked: np.ndarray | None = None, fill: str = "none"):
    """(T,N) 점수/다음날수익 -> (일 수익률, 투입비중 0~1, 보유여부).
    fill='backfill' : 미체결 종목을 후보에서 빼고 다음 순위로 채움 / 'cash' : 미체결 몫은 현금(수익 0, 비용 0)으로 남김."""
    if blocked is not None and fill == "backfill":
        S = np.where(blocked, np.nan, S)
    S = np.where(np.isnan(S), -np.inf, S)
    top = np.argpartition(-S, k - 1, axis=1)[:, :k]
    xt = np.take_along_axis(X, top, 1)
    ok = np.isfinite(np.take_along_axis(S, top, 1)) & np.isfinite(xt)
    cnt = ok.sum(1)
    if blocked is not None and fill == "cash":
        filled = ok & ~np.take_along_axis(blocked, top, 1)
    else:
        filled = ok
    r = np.where(filled, xt, 0.0).sum(1) / np.maximum(cnt, 1)
    frac = filled.sum(1) / np.maximum(cnt, 1)
    return r, frac, cnt > 0


def summarize_net(net: np.ndarray, frac: np.ndarray) -> dict:
    m = bt_metrics(pd.Series(net))
    inv = frac > 0
    return {"sharpe": m["sharpe"], "cumulative_return": m["cumulative_return"], "cagr": m["cagr"], "mdd": m["mdd"],
            "calmar": m["cumulative_return"] / abs(m["mdd"]) if m["mdd"] else None,
            "win_rate": float((net[inv] > 0).mean()) if inv.any() else None, "invested_days": int(inv.sum()), "days": int(len(net))}


class Lab:
    def __init__(self):
        self.data = ds.build()
        self.eng = self.data.eng
        t0, t1 = self.eng.t0, self.eng.t1
        self.sl = slice(t0, t1 + 1)
        self.X = self.eng.nxt[self.sl]
        c, v = self.eng.panel["close"].to_numpy(), self.eng.panel["volume"].to_numpy()
        ret = np.vstack([np.full((1, c.shape[1]), np.nan), c[1:] / c[:-1] - 1])
        self.limit_up = ret >= LIMIT_UP
        self.halt = ~(v > 0)
        self.blocked_full = self.limit_up | self.halt
        self.blocked = self.blocked_full[self.sl]
        self.plan = self.data.walk_forward_plan()
        self.B = {"overall": (self.plan[0]["test"][0], self.plan[-1]["test"][1]), **{str(p["year"]): tuple(p["test"]) for p in self.plan}}
        self.A = {k: tuple(v) for k, v in ds.THREE_WAY.items()}
        self._ic: dict = {}

    def segments(self, protocol: str) -> dict:
        return self.A if protocol == "three_way" else self.B

    # 성과 -------------------------------------------------------------
    def backtest(self, S, start, end, gate="none", k=5, cost=0.001, fill="none") -> dict:
        lo, hi = self.data.day_range(start, end)
        r, frac, _ = pick_arr(S[self.sl].astype(float), self.X, k, self.blocked, fill)
        g = self.eng.gates[gate]
        net = g * (r - cost * frac)
        return summarize_net(net[lo:hi + 1], (frac * g)[lo:hi + 1])

    def ic(self, S, start, end) -> dict:
        """일별 횡단면 Spearman IC (예측 점수 vs 다음날 초과수익). 평균 · IR(평균/표준편차) · 양(+)의 날 비율."""
        key = (id(S), start, end)
        if key in self._ic:
            return self._ic[key]
        lo, hi = self.data.day_range(start, end)
        tt = np.arange(lo, hi + 1) + self.eng.t0
        Sw, Y = S[tt].astype(float), self.data.y_excess[tt].astype(float)
        m = np.isfinite(Sw) & np.isfinite(Y)
        Sm, Ym = np.where(m, Sw, np.nan), np.where(m, Y, np.nan)
        a, b = pd.DataFrame(Sm).rank(axis=1).to_numpy(), pd.DataFrame(Ym).rank(axis=1).to_numpy()
        n = m.sum(1)
        a0, b0 = a - np.nanmean(a, 1, keepdims=True), b - np.nanmean(b, 1, keepdims=True)
        den = np.sqrt(np.nansum(a0 ** 2, 1) * np.nansum(b0 ** 2, 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            ic = np.nansum(a0 * b0, 1) / den
        ic = ic[(n >= 10) & np.isfinite(ic)]
        out = {"ic": float(ic.mean()), "ic_ir": float(ic.mean() / ic.std(ddof=1)) if len(ic) > 1 else None,
               "ic_hit": float((ic > 0).mean()), "ic_days": int(len(ic))} if len(ic) else {"ic": None, "ic_ir": None, "ic_hit": None, "ic_days": 0}
        self._ic[key] = out
        return out


# ------------------------------------------------------------------ 점수 행렬 로딩

def load(variant: str, name: str, proto: str, seed: int) -> np.ndarray:
    return np.load(CACHE / f"{variant}_{name}_{'A' if proto == 'three_way' else 'B'}_s{seed}.npy").astype(np.float32)


def base(name: str, proto: str) -> np.ndarray:
    return np.load(CACHE / f"base_{name}_{'A' if proto == 'three_way' else 'B'}.npy").astype(np.float32)


def build_scores(lab: Lab) -> dict:
    """{protocol: {이름: 점수행렬}}  (이름 규칙: <variant>/<model>/s<seed> · <variant>/<model>_ens · ridge · hgb · rule_k5 · rule_k7 · ens/...)"""
    cfg = json.loads((RESULTS_DIR / "week08_nonml_search.json").read_text(encoding="utf-8"))[0]
    rule = w8.build_score(lab.eng, cfg).astype(np.float32)
    out = {}
    for proto in ("three_way", "walk_forward"):
        d = {"rule": rule, "ridge": base("ridge", proto), "hgb": base("hgb", proto)}
        for variant, seeds in (("loss", LOSS_SEEDS), ("ic", IC_SEEDS)):
            for name in ("mlp", "gru"):
                ranks = []
                for s in seeds:
                    S = load(variant, name, proto, s)
                    d[f"{variant}/{name}/s{s}"] = S
                    ranks.append(drank(S))
                d[f"{variant}/{name}_ens"] = np.mean(ranks, axis=0).astype(np.float32)  # 시드별 일별 순위의 평균
                if variant == "ic":
                    d[f"ic_s012/{name}_ens"] = np.mean(ranks[:3], axis=0).astype(np.float32)
        rr = drank(d["ridge"])
        d["ens/ridge_gru_ic"] = (0.5 * rr + 0.5 * drank(d["ic/gru_ens"])).astype(np.float32)   # 0.5*rank(ridge) + 0.5*mean_seeds(rank(gru))
        d["ens/ridge_gru_loss"] = (0.5 * rr + 0.5 * drank(d["loss/gru_ens"])).astype(np.float32)  # 교정 전 GRU 로 만든 같은 앙상블 (참고)
        out[proto] = d
    return out, cfg


# ------------------------------------------------------------------ 섹션별 행 생성

class Rows(list):
    def add(self, section, variant, model, proto, seg, gate, k, cost_bp, fill, seed, m, ic=None):
        self.append({"section": section, "variant": variant, "model": model, "protocol": proto, "segment": seg, "gate": gate, "k": k,
                     "cost_bp": cost_bp, "fill": fill, "seed": seed, **m, **(ic or {})})


def eval_model(lab, rows, scores, section, variant, model, key_fn, seeds, protos=("three_way", "walk_forward"), gates=GATES, ks=KS,
               costs=(10,), fills=("none",), segs=None, with_year=False):
    """key_fn(proto, seed) -> 점수행렬 (seeds=[None] 이면 단일). seed 별 행 + 시드 평균 행(-1)."""
    for proto in protos:
        segments = lab.segments(proto)
        names = segs or (list(segments) if proto == "three_way" else (["overall"] + ([s for s in segments if s != "overall"] if with_year else [])))
        for seg in names:
            a, b = segments[seg]
            for gate in gates:
                for k in ks:
                    for cb in costs:
                        for fill in fills:
                            ms, ics = [], []
                            for sd in seeds:
                                S = key_fn(proto, sd)
                                m = lab.backtest(S, a, b, gate, k, cb / 1e4, fill)
                                ic = lab.ic(S, a, b)
                                rows.add(section, variant, model, proto, seg, gate, k, cb, fill, -1 if sd is None else sd, m, ic)
                                ms.append(m)
                                ics.append(ic)
                            if len(seeds) > 1:
                                mean = lambda L, key: (float(np.mean([x[key] for x in L if x[key] is not None])) if any(x[key] is not None for x in L) else None)
                                rows.add(section, variant, model, proto, seg, gate, k, cb, fill, -1,
                                         {key: mean(ms, key) for key in ms[0]}, {key: mean(ics, key) for key in ics[0]})


def run_placebo(lab: Lab, rows: list, scores: dict, rng) -> list[dict]:
    res = []
    cands = {"ens_ridge_gru": "ens/ridge_gru_ic", "gru_ens5": "ic/gru_ens", "mlp_ens5": "ic/mlp_ens", "ridge": "ridge", "rule_k5": "rule"}
    for proto in ("three_way", "walk_forward"):
        a, b = (lab.A["test"] if proto == "three_way" else lab.B["overall"])
        lo, hi = lab.data.day_range(a, b)
        X = lab.X[lo:hi + 1]
        for gate in ("none", "ma120"):
            g = lab.eng.gates[gate][lo:hi + 1]

            def sh(S):
                r, frac, _ = pick_arr(S, X, 5)
                net = g * (r - 0.001 * frac)
                sd = net.std(ddof=1)
                return float(net.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0

            for label, key in cands.items():
                S = scores[proto][key][lab.sl][lo:hi + 1].astype(float)
                actual, fin = sh(S), np.isfinite(S)
                shuf = np.array([sh(np.where(fin, rng.random(S.shape), np.nan)) for _ in range(N_PERM)])
                n = len(S)
                shifts = [sh(np.roll(S, s, axis=0)) for s in range(10, n - 10)]
                for kind, arr in (("cross_shuffle", shuf), ("time_shift", np.array(shifts))):
                    res.append({"model": label, "protocol": proto, "gate": gate, "kind": kind, "actual_sharpe": actual, "n": int(len(arr)),
                                "p_mean": float(arr.mean()), "p_std": float(arr.std(ddof=1)), "p95": float(np.percentile(arr, 95)),
                                "p99": float(np.percentile(arr, 99)), "p_value": float((1 + (arr >= actual).sum()) / (1 + len(arr))),
                                "z": float((actual - arr.mean()) / arr.std(ddof=1))})
    return res


def fill_stats(lab: Lab, scores: dict) -> list[dict]:
    """상한가/거래정지 필터가 실제로 걸린 정도: 평가구간 내 (날짜,종목) 중 차단 건수와, 모델 Top-5 선정 중 차단된 비율."""
    out = []
    for proto in ("three_way", "walk_forward"):
        a, b = (lab.A["test"] if proto == "three_way" else lab.B["overall"])
        lo, hi = lab.data.day_range(a, b)
        sl = slice(lo, hi + 1)
        lu, ht = lab.limit_up[lab.sl][sl], lab.halt[lab.sl][sl]
        valid = np.isfinite(lab.X[sl])
        nxt_halt = np.vstack([lab.halt[1:], np.ones((1, lab.halt.shape[1]), bool)])[lab.sl][sl]
        base_row = {"protocol": proto, "window": [a, b], "cells": int(valid.sum()), "limit_up_cells": int((lu & valid).sum()),
                    "halt_cells": int((ht).sum()), "next_day_halt_cells": int((nxt_halt & valid).sum())}
        for label, key in {"ens_ridge_gru": "ens/ridge_gru_ic", "gru_ens5": "ic/gru_ens", "ridge": "ridge", "rule_k5": "rule"}.items():
            S = np.where(np.isnan(scores[proto][key][lab.sl][sl].astype(float)), -np.inf, scores[proto][key][lab.sl][sl].astype(float))
            top = np.argpartition(-S, 4, axis=1)[:, :5]
            bl = np.take_along_axis(lab.blocked[sl], top, 1)
            lu_t = np.take_along_axis(lu, top, 1)
            nh = np.take_along_axis(nxt_halt, top, 1)
            out.append({**base_row, "model": label, "picks": int(top.size), "picks_blocked": int(bl.sum()), "picks_limit_up": int(lu_t.sum()),
                        "picks_next_day_halt": int(nh.sum()), "pick_blocked_rate": float(bl.mean())})
    return out


# ------------------------------------------------------------------ 피처 상관 / VIF

def feature_stats(lab: Lab) -> dict:
    cube = lab.data.cube.astype(np.float64)
    F = lab.data.features
    scopes = {"all": lab.data.day_range(str(lab.data.days[0].date()), str(lab.data.days[-1].date())),
              "train": lab.data.day_range(*ds.THREE_WAY["train"])}
    out = {"features": F, "scopes": {}}
    for scope, (lo, hi) in scopes.items():
        tt = np.arange(lo, hi + 1) + lab.eng.t0
        Rs, pooled = [], []
        for t in tt:
            x = cube[t]
            x = x[np.isfinite(x).all(1)]
            if len(x) < 30:
                continue
            Rs.append(np.corrcoef(x, rowvar=False))  # 큐브가 이미 같은 날 횡단면 순위 -> 순위의 피어슨 = 스피어만
            pooled.append(x)
        R = np.nanmean(Rs, axis=0)
        P = np.corrcoef(np.vstack(pooled), rowvar=False)
        vif_d, vif_p = np.diag(np.linalg.inv(R)), np.diag(np.linalg.inv(P))
        iu = np.triu_indices(len(F), 1)
        pairs = sorted(((float(R[i, j]), F[i], F[j]) for i, j in zip(*iu)), key=lambda x: -abs(x[0]))
        out["scopes"][scope] = {"days": len(Rs), "corr_daily_avg": R.round(4).tolist(), "corr_pooled": P.round(4).tolist(),
                                "vif_daily": dict(zip(F, vif_d.round(3).tolist())), "vif_pooled": dict(zip(F, vif_p.round(3).tolist())),
                                "top_pairs": [{"rho": round(r, 3), "a": a, "b": b} for r, a, b in pairs[:12]],
                                "n_abs_ge_0.7": int(sum(abs(r) >= 0.7 for r, _, _ in pairs)), "n_abs_ge_0.5": int(sum(abs(r) >= 0.5 for r, _, _ in pairs)),
                                "n_vif_gt5": int((vif_d > 5).sum()), "n_vif_gt10": int((vif_d > 10).sum()), "max_vif": float(vif_d.max()),
                                "cond_number": float(np.linalg.cond(R))}
    return out


# ------------------------------------------------------------------ DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS w11_metrics (
    id SERIAL PRIMARY KEY, section VARCHAR(16) NOT NULL, variant VARCHAR(12) NOT NULL, model VARCHAR(24) NOT NULL,
    protocol VARCHAR(12) NOT NULL, segment VARCHAR(10) NOT NULL, gate VARCHAR(10) NOT NULL, k SMALLINT NOT NULL,
    cost_bp REAL NOT NULL, fill VARCHAR(10) NOT NULL, seed SMALLINT NOT NULL,
    sharpe DOUBLE PRECISION, cumulative_return DOUBLE PRECISION, cagr DOUBLE PRECISION, mdd DOUBLE PRECISION, calmar DOUBLE PRECISION,
    win_rate DOUBLE PRECISION, invested_days INTEGER, days INTEGER, ic DOUBLE PRECISION, ic_ir DOUBLE PRECISION, ic_hit DOUBLE PRECISION, ic_days INTEGER
);
CREATE INDEX IF NOT EXISTS idx_w11_metrics_q ON w11_metrics (section, model, protocol, segment);
CREATE TABLE IF NOT EXISTS w11_placebo (
    model VARCHAR(24) NOT NULL, protocol VARCHAR(12) NOT NULL, gate VARCHAR(10) NOT NULL, kind VARCHAR(14) NOT NULL,
    actual_sharpe DOUBLE PRECISION, n INTEGER, p_mean DOUBLE PRECISION, p_std DOUBLE PRECISION, p95 DOUBLE PRECISION, p99 DOUBLE PRECISION,
    p_value DOUBLE PRECISION, z DOUBLE PRECISION, PRIMARY KEY (model, protocol, gate, kind)
);
CREATE TABLE IF NOT EXISTS w11_feature_corr (scope VARCHAR(8) NOT NULL, f1 VARCHAR(24) NOT NULL, f2 VARCHAR(24) NOT NULL, rho_daily DOUBLE PRECISION, rho_pooled DOUBLE PRECISION, PRIMARY KEY (scope, f1, f2));
CREATE TABLE IF NOT EXISTS w11_feature_vif (scope VARCHAR(8) NOT NULL, feature VARCHAR(24) NOT NULL, vif_daily DOUBLE PRECISION, vif_pooled DOUBLE PRECISION, PRIMARY KEY (scope, feature));
CREATE TABLE IF NOT EXISTS w11_meta (key VARCHAR(40) PRIMARY KEY, value JSONB NOT NULL);
"""
M_COLS = ["section", "variant", "model", "protocol", "segment", "gate", "k", "cost_bp", "fill", "seed", "sharpe", "cumulative_return", "cagr", "mdd",
          "calmar", "win_rate", "invested_days", "days", "ic", "ic_ir", "ic_hit", "ic_days"]


def save_db(res: dict) -> dict:
    import db
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA)
        cur.execute("TRUNCATE w11_metrics, w11_placebo, w11_feature_corr, w11_feature_vif, w11_meta RESTART IDENTITY")
        cur.executemany(f"INSERT INTO w11_metrics ({','.join(M_COLS)}) VALUES ({','.join(['%s'] * len(M_COLS))})",
                        [tuple(r.get(c) for c in M_COLS) for r in res["metrics"]])
        P = ["model", "protocol", "gate", "kind", "actual_sharpe", "n", "p_mean", "p_std", "p95", "p99", "p_value", "z"]
        cur.executemany(f"INSERT INTO w11_placebo ({','.join(P)}) VALUES ({','.join(['%s'] * len(P))})", [tuple(r[c] for c in P) for r in res["placebo"]])
        F = res["features"]["features"]
        for scope, v in res["features"]["scopes"].items():
            cur.executemany("INSERT INTO w11_feature_corr VALUES (%s,%s,%s,%s,%s)",
                            [(scope, F[i], F[j], v["corr_daily_avg"][i][j], v["corr_pooled"][i][j]) for i in range(len(F)) for j in range(len(F))])
            cur.executemany("INSERT INTO w11_feature_vif VALUES (%s,%s,%s,%s)", [(scope, f, v["vif_daily"][f], v["vif_pooled"][f]) for f in F])
        for key in ("config", "train_log", "fill_stats", "feature_summary", "repro_check"):
            cur.execute("INSERT INTO w11_meta VALUES (%s,%s)", (key, json.dumps(res[key], ensure_ascii=False)))
    return {"metrics": len(res["metrics"]), "placebo": len(res["placebo"])}


# ------------------------------------------------------------------ 메인

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-db", action="store_true")
    a = ap.parse_args()
    t_all = time.time()
    lab = Lab()
    scores, cfg = build_scores(lab)
    rows = Rows()
    S = lambda name: (lambda proto, sd: scores[proto][name])
    SD = lambda variant, model: (lambda proto, sd: scores[proto][f"{variant}/{model}/s{sd}"])

    # (1) 교정 전/후: 시드별 + 시드평균 (게이트 3종 x K 5·7), 방식 A 전 구간 + 방식 B 전체
    for variant, seeds in (("loss", LOSS_SEEDS), ("ic", IC_SEEDS)):
        for model in ("mlp", "gru"):
            eval_model(lab, rows, scores, "correction", variant, model, SD(variant, model), seeds)
    for model in ("mlp", "gru"):  # 같은 3시드(0~2)끼리의 순수 비교용
        eval_model(lab, rows, scores, "correction", "ic_s012", model, SD("ic", model), [0, 1, 2])
    # 시드 앙상블(일별 순위 평균) 점수
    for variant in ("loss", "ic"):
        for model in ("mlp", "gru"):
            eval_model(lab, rows, scores, "correction", variant, f"{model}_ens", S(f"{variant}/{model}_ens"), [None])

    # (1b) 기준선: 룰베이스(K=5/7 병기) · Ridge · HGB — 동일 구간·동일 게이트
    for k in KS:
        eval_model(lab, rows, scores, "baseline", "-", f"rule_k{k}", S("rule"), [None], ks=[k], with_year=True)
    eval_model(lab, rows, scores, "baseline", "-", "ridge", S("ridge"), [None], with_year=True)
    eval_model(lab, rows, scores, "baseline", "-", "hgb", S("hgb"), [None])

    # (2) 앙상블
    eval_model(lab, rows, scores, "ensemble", "ic", "ens_ridge_gru", S("ens/ridge_gru_ic"), [None], with_year=True)
    eval_model(lab, rows, scores, "ensemble", "loss", "ens_ridge_gru", S("ens/ridge_gru_loss"), [None])

    # (3) 강건성: 비용 민감도 / 미체결 필터  (K=5, 게이트 none·ma120, 방식 A test + 방식 B overall)
    rob = {"ens_ridge_gru": "ens/ridge_gru_ic", "gru_ens5": "ic/gru_ens", "mlp_ens5": "ic/mlp_ens", "ridge": "ridge", "rule_k5": "rule"}
    for label, key in rob.items():
        kw = dict(gates=["none", "ma120"], ks=[5])
        for proto, seg in (("three_way", "test"), ("walk_forward", "overall")):
            eval_model(lab, rows, scores, "cost", "ic", label, S(key), [None], protos=(proto,), costs=COSTS_BP, segs=[seg], **kw)
            eval_model(lab, rows, scores, "fill", "ic", label, S(key), [None], protos=(proto,), fills=("none", "backfill", "cash"), segs=[seg], **kw)
    rng = np.random.default_rng(20261125)
    placebo = run_placebo(lab, rows, scores, rng)
    fills = fill_stats(lab, scores)

    # 재현 확인: 교정 전(loss) 시드별 OOS 샤프가 week10 결과 JSON 과 같은가
    old = json.loads((RESULTS_DIR / "week10_nn_results.json").read_text(encoding="utf-8"))
    chk = []
    for model in ("mlp", "gru"):
        for r in old["three_way"]["models"][model]["runs"]:
            new = next(x for x in rows if x["section"] == "correction" and x["variant"] == "loss" and x["model"] == model and x["protocol"] == "three_way"
                       and x["segment"] == "test" and x["gate"] == "none" and x["k"] == 5 and x["seed"] == r["seed"])
            chk.append({"model": model, "seed": r["seed"], "week10_sharpe": r["report"]["none"]["test"]["sharpe"], "rerun_sharpe": round(new["sharpe"], 3)})

    feat = feature_stats(lab)
    sc = feat["scopes"]["all"]
    tlog = {v: json.loads((CACHE / f"train_log_{v}.json").read_text(encoding="utf-8")) for v in ("ic", "loss")}
    res = {"config": {"saved_at": pd.Timestamp.now().isoformat(timespec="seconds"), "gates": GATES, "ks": KS, "costs_bp": COSTS_BP, "n_perm": N_PERM,
                      "limit_up_ret": LIMIT_UP, "ic_seeds": IC_SEEDS, "loss_seeds": LOSS_SEEDS, "rule_cfg": cfg,
                      "early_stop_ic": {"metric": "Val 일별 Spearman IC 평균 최대", "min_epochs": 10, "patience": 7, "max_epochs": 60},
                      "early_stop_loss": {"metric": "Val Huber 손실 최소", "patience": 5, "max_epochs": 60},
                      "ensemble": "core = 0.5*rank(ridge) + 0.5*mean_seeds(rank(gru_ic)) (일별 횡단면 백분위 순위)"},
           "metrics": rows, "placebo": placebo, "fill_stats": fills, "features": feat, "train_log": tlog, "repro_check": chk,
           "feature_summary": {k: sc[k] for k in ("n_abs_ge_0.7", "n_abs_ge_0.5", "n_vif_gt5", "n_vif_gt10", "max_vif", "cond_number", "top_pairs")}}
    OUT.write_text(json.dumps(res, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"저장: {OUT} · 행 {len(rows)} · {time.time() - t_all:.0f}s")
    if not a.no_db:
        print("DB:", save_db(res))


if __name__ == "__main__":
    main()
