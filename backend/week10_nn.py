"""Week 10 (2단계): PyTorch 뉴럴 네트워크 학습 · 과적합 실험 · 룰베이스/Ridge/HistGB 와 동일 구간 비교.

모델
  mlp         [계열 A] (N,17) Linear(17,32)-LayerNorm-GELU-Dropout(.3)-Linear(32,16)-GELU-Dropout(.2)-Linear(16,1)
  gru         [계열 B] (N,20,17) GRU(hidden 32, 1층) -> 마지막 스텝 -> LayerNorm -> Dropout(.3) -> Linear(32,1)
  overfit_mlp [대조군] 128 x 3층, Dropout 0, weight_decay 0, Early Stopping 없음 (고정 에폭)
학습: AdamW(lr 1e-3, wd 1e-3) · HuberLoss(delta=1, 타깃은 Train 표준편차로만 스케일) · Val 손실 기준 patience 5, 최적 에폭 가중치 복원 · 시드 고정
평가: 왕복 0.10% · Top-5 동일비중 · 현금일 포함
  방식 A 3단 분할 (Train/Val/Test = IS/Val/OOS) · 방식 B 연도별 확장형 Walk-Forward 2022~2026 (+ 레짐 게이트 유무)
비교 기준선: 비ML 룰베이스 최고(8주차 json), Ridge, HistGradientBoosting (같은 2D 입력·타깃·구간)

사용법 (컨테이너 안):  python week10_nn.py [--seeds 3] [--quick]
출력: results/week10_nn_results.json, models/week10_best_nn.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from torch import nn

import week08 as w8
import week10_dataset as ds
from backtest import RESULTS_DIR

MODEL_DIR = Path(__file__).resolve().parent / "models"
CKPT_PATH = MODEL_DIR / "week10_best_nn.pt"
RESULT_PATH = RESULTS_DIR / "week10_nn_results.json"

F = len(ds.FEATURES)
LR, WD, BATCH, PATIENCE = 1e-3, 1e-3, 1024, 5
MAX_EPOCHS = 60
OVERFIT_EPOCHS = 60
HEADLINE_GATE = "ma120"  # 사전 고정 (사후 선택 편향 방지). 나머지 게이트는 참고용으로만 기록한다.
torch.set_num_threads(8)


# ------------------------------------------------------------------ 모델

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(F, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(0.3),
                                 nn.Linear(32, 16), nn.GELU(), nn.Dropout(0.2), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GRUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(F, 32, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(32), nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)  # 마지막 스텝(= 당일 t)


class OverfitMLP(nn.Module):
    def __init__(self, h: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(F, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


SPECS = {
    "mlp": {"cls": MLP, "mode": "2D", "wd": WD, "early_stop": True, "epochs": MAX_EPOCHS,
            "desc": "MLP 17-32(LN,GELU,Drop.3)-16(GELU,Drop.2)-1", "input": f"(N,{F})"},
    "gru": {"cls": GRUNet, "mode": "3D", "wd": WD, "early_stop": True, "epochs": MAX_EPOCHS,
            "desc": f"GRU(hidden 32, 1층)-last step-LayerNorm-Drop.3-Linear(32,1)", "input": f"(N,{ds.SEQ_LEN},{F})"},
    "overfit_mlp": {"cls": OverfitMLP, "mode": "2D", "wd": 0.0, "early_stop": False, "epochs": OVERFIT_EPOCHS,
                    "desc": "비규제 대형 MLP 128x3층, Dropout 0, wd 0, Early Stopping 없음", "input": f"(N,{F})"},
}


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------ 학습

class Tens:
    """샘플 집합 -> (X, y/σ) 텐서. σ 는 Train 의 표준편차(스케일러가 미래를 보지 않게 Train 으로만 fit)."""

    def __init__(self, sset, sigma: float):
        X, y = sset.tensors()
        self.X, self.y, self.n = X, y / sigma, len(sset)


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    model.eval()
    return torch.cat([model(X[i:i + 8192]) for i in range(0, len(X), 8192)]).numpy()


@torch.no_grad()
def eval_loss(model, T: Tens, loss_fn) -> float:
    model.eval()
    return float(loss_fn(torch.from_numpy(predict(model, T.X)), T.y))


def fit(name: str, tr: Tens, va: Tens | None, seed: int, snapshot_cb=None) -> tuple[nn.Module, dict]:
    """Train 으로 학습. early_stop 이면 Val 손실이 patience 연속 개선 없을 때 중단하고 최적 에폭 가중치를 복원한다."""
    spec = SPECS[name]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = spec["cls"]()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=spec["wd"])
    loss_fn = nn.HuberLoss(delta=1.0)
    gen = torch.Generator().manual_seed(seed)
    hist = {"train_loss": [], "val_loss": []}
    best, best_state, bad = float("inf"), None, 0
    for ep in range(1, spec["epochs"] + 1):
        model.train()
        perm = torch.randperm(tr.n, generator=gen)
        for i in range(0, tr.n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss_fn(model(tr.X[idx]), tr.y[idx]).backward()
            opt.step()
        hist["train_loss"].append(eval_loss(model, tr, loss_fn))  # dropout 끈 상태의 Train 손실 (Val 과 같은 조건)
        vl = eval_loss(model, va, loss_fn) if va is not None else float("nan")
        hist["val_loss"].append(vl)
        if snapshot_cb:
            snapshot_cb(ep, model)
        if spec["early_stop"] and va is not None:
            if vl < best - 1e-6:
                best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
                hist["best_epoch"] = ep
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    hist["epochs_run"] = len(hist["train_loss"])
    hist.setdefault("best_epoch", hist["epochs_run"])
    return model, hist


# ------------------------------------------------------------------ 점수 행렬 / 지표

def score_matrix(data: ds.Week10Data, parts: list[tuple]) -> np.ndarray:
    """[(SampleSet, 예측 배열), ...] -> (T_full, N) 점수 (그 외 칸은 NaN = 후보 아님)."""
    S = np.full(data.cube.shape[:2], np.nan)
    for sset, p in parts:
        S[sset.t, sset.i] = p
    return S


def summarize(bt: dict) -> dict:
    m = bt["metrics"]
    return {"sharpe": round(float(m["sharpe"]), 3), "cumulative_return": round(float(m["cumulative_return"]), 4),
            "cagr": round(float(m["cagr"]), 4) if m.get("cagr") is not None else None,
            "mdd": round(float(m["mdd"]), 4), "win_rate": round(float(m["win_rate_invested"]), 4) if m.get("win_rate_invested") is not None else None,
            "calmar": round(float(m["calmar"]), 3) if m.get("calmar") is not None else None,
            "invested_days": int(m["invested_days"]), "days": int(bt["days"])}


def net_curve(data, S: np.ndarray, start: str | None = None, end: str | None = None) -> np.ndarray:
    """일별 순수익률(왕복 0.10% 차감, 현금일 0) — 누적수익률 곡선용. 게이트 없음."""
    bt = data.backtest(S, start, end, ds.TOP_K, "none")
    return bt["r"] - ds.COST * bt["inv"]


def period_report(data, S, ranges: dict, gate: str = "none") -> dict:
    return {k: summarize(data.backtest(S, a, b, ds.TOP_K, gate)) for k, (a, b) in ranges.items()}


# ------------------------------------------------------------------ 기준선

def rule_baseline(data: ds.Week10Data) -> tuple[np.ndarray, dict]:
    cfg = json.loads((RESULTS_DIR / "week08_nonml_search.json").read_text(encoding="utf-8"))[0]
    return w8.build_score(data.eng, cfg), cfg


def make_sk(name: str):
    if name == "ridge":
        return Ridge(alpha=100.0)
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_depth=3, max_leaf_nodes=7, min_samples_leaf=100,
                                         l2_regularization=10.0, random_state=0)


def sk_fit_predict(name: str, tr, others: list) -> list[np.ndarray]:
    m = make_sk(name).fit(tr.x_batch(np.arange(len(tr))), tr.y)
    return [m.predict(o.x_batch(np.arange(len(o)))) for o in others]


# ------------------------------------------------------------------ 방식 A: 3단 분할

def run_three_way(data, seeds: list[int], log=print, collect: dict | None = None) -> tuple[dict, dict]:
    out, keep = {}, {}
    ranges = ds.THREE_WAY
    sets = {m: data.three_way(m) for m in ("2D", "3D")}
    sigma = float(sets["2D"]["train"].y.std())
    T = {m: {k: Tens(v, sigma) for k, v in s.items()} for m, s in sets.items()}
    log(f"[3단 분할] Train {len(sets['2D']['train']):,} / Val {len(sets['2D']['val']):,} / Test {len(sets['2D']['test']):,} · 타깃 σ(Train)={sigma:.5f}")

    for name, spec in SPECS.items():
        mode = spec["mode"]
        runs = []
        for seed in seeds:
            t0 = time.time()
            snaps = {}

            def cb(ep, model, name=name, mode=mode, snaps=snaps):  # 과적합 대조군: 에폭별 IS/OOS 샤프 (X자 역전 관찰)
                if name == "overfit_mlp" and ep in (1, 2, 3, 5, 8, 12, 20, 30, 45, 60):
                    S = score_matrix(data, [(sets[mode][k], predict(model, T[mode][k].X)) for k in ("train", "val", "test")])
                    r = period_report(data, S, ranges)
                    snaps[ep] = {k: r[k]["sharpe"] for k in r}

            model, hist = fit(name, T[mode]["train"], T[mode]["val"], seed, cb)
            S = score_matrix(data, [(sets[mode][k], predict(model, T[mode][k].X)) for k in ("train", "val", "test")])
            if collect is not None and seed == seeds[0]:
                collect.setdefault("three_way", {})[name] = net_curve(data, S)
            rep = {"none": period_report(data, S, ranges, "none"), HEADLINE_GATE: period_report(data, S, ranges, HEADLINE_GATE)}
            runs.append({"seed": seed, "params": n_params(model), "report": rep, "history": hist, **({"epoch_sharpe": snaps} if snaps else {})})
            r = rep["none"]
            log(f"  {name:12} seed {seed} · 파라미터 {n_params(model):,} · best/run 에폭 {hist['best_epoch']}/{hist['epochs_run']} · "
                f"샤프 IS {r['train']['sharpe']:.2f} Val {r['val']['sharpe']:.2f} OOS {r['test']['sharpe']:.2f} · {time.time() - t0:.0f}s")
            keep[(name, seed)] = (model, rep)
        out[name] = {"params": runs[0]["params"], "arch": spec["desc"], "input": spec["input"], "runs": runs,
                     "mean_sharpe": {g: {k: round(float(np.mean([r["report"][g][k]["sharpe"] for r in runs])), 3) for k in ranges}
                                     for g in ("none", HEADLINE_GATE)}}
    out["_sigma"] = sigma
    return out, keep


def baselines_three_way(data, log=print, collect: dict | None = None) -> dict:
    ranges = ds.THREE_WAY
    s2 = data.three_way("2D")
    out = {}
    rule, cfg = rule_baseline(data)
    if collect is not None:
        collect.setdefault("three_way", {})["rule_best"] = net_curve(data, rule)
    out["rule_best"] = {"desc": w8.describe(cfg), "params": 0, "own_gate": {"gate": cfg["regime"], **period_report(data, rule, ranges, cfg["regime"])},
                        "none": period_report(data, rule, ranges, "none"), HEADLINE_GATE: period_report(data, rule, ranges, HEADLINE_GATE)}
    for name in ("ridge", "hgb"):
        preds = sk_fit_predict(name, s2["train"], [s2["train"], s2["val"], s2["test"]])
        S = score_matrix(data, list(zip((s2["train"], s2["val"], s2["test"]), preds)))
        if collect is not None:
            collect.setdefault("three_way", {})[name] = net_curve(data, S)
        out[name] = {"none": period_report(data, S, ranges, "none"), HEADLINE_GATE: period_report(data, S, ranges, HEADLINE_GATE)}
        r = out[name]["none"]
        log(f"  {name:12} · 샤프 IS {r['train']['sharpe']:.2f} Val {r['val']['sharpe']:.2f} OOS {r['test']['sharpe']:.2f}")
    r = out["rule_best"]["none"]
    log(f"  rule_best    · (게이트 none) 샤프 IS {r['train']['sharpe']:.2f} Val {r['val']['sharpe']:.2f} OOS {r['test']['sharpe']:.2f}")
    return out


# ------------------------------------------------------------------ 방식 B: 연도별 확장형 Walk-Forward

def run_walk_forward(data, seeds: list[int], log=print, collect: dict | None = None) -> dict:
    plan = data.walk_forward_plan()
    folds = {m: data.walk_forward(m) for m in ("2D", "3D")}
    lo, hi = plan[0]["test"][0], plan[-1]["test"][1]
    years = {p["year"]: p["test"] for p in plan}
    gates = list(data.gates)
    out = {"plan": plan, "models": {}}

    def finish(S_by_key):  # {키: 점수행렬} -> 전체/연도별 x 게이트
        res = {}
        for key, S in S_by_key.items():
            res[key] = {g: {"overall": summarize(data.backtest(S, lo, hi, ds.TOP_K, g)),
                            "per_year": {y: summarize(data.backtest(S, a, b, ds.TOP_K, g)) for y, (a, b) in years.items()}} for g in gates}
        return res

    for name, spec in SPECS.items():
        mode = spec["mode"]
        mats, hists = [], []
        for seed in seeds:
            S = np.full(data.cube.shape[:2], np.nan)
            h = {}
            for fold in folds[mode]:
                sigma = float(fold["train"].y.std())
                Ttr, Tva = Tens(fold["train"], sigma), Tens(fold["val"], sigma)
                model, hist = fit(name, Ttr, Tva, seed)
                te = fold["test"]
                S[te.t, te.i] = predict(model, Tens(te, sigma).X)
                h[fold["year"]] = hist
            mats.append(S)
            hists.append(h)
        if collect is not None:
            collect.setdefault("walk_forward", {})[name] = net_curve(data, mats[0], lo, hi)
        per_seed = finish({f"seed{s}": S for s, S in zip(seeds, mats)})
        out["models"][name] = {"params": n_params(spec["cls"]()), "seeds": per_seed,
                               "mean_sharpe_overall": {g: round(float(np.mean([per_seed[f"seed{s}"][g]["overall"]["sharpe"] for s in seeds])), 3) for g in gates},
                               "history_seed0": hists[0]}
        m = out["models"][name]["mean_sharpe_overall"]
        log(f"  WF {name:12} 시드평균 OOS 샤프 게이트 none {m['none']:.2f} · {HEADLINE_GATE} {m[HEADLINE_GATE]:.2f} · "
            f"시드별 {[per_seed[f'seed{s}']['none']['overall']['sharpe'] for s in seeds]}")

    # 기준선
    rule, cfg = rule_baseline(data)
    base = {"rule_best": finish({"rule": rule})["rule"]}
    if collect is not None:
        collect.setdefault("walk_forward", {})["rule_best"] = net_curve(data, rule, lo, hi)
    for name in ("ridge", "hgb"):
        S = np.full(data.cube.shape[:2], np.nan)
        for fold in folds["2D"]:
            te = fold["test"]
            S[te.t, te.i] = sk_fit_predict(name, fold["train"], [te])[0]
        base[name] = finish({name: S})[name]
        if collect is not None:
            collect.setdefault("walk_forward", {})[name] = net_curve(data, S, lo, hi)
    out["baselines"] = base
    for n, v in base.items():
        log(f"  WF {n:12} OOS 샤프 none {v['none']['overall']['sharpe']:.2f} · {HEADLINE_GATE} {v[HEADLINE_GATE]['overall']['sharpe']:.2f}")
    out["period"] = [lo, hi]
    return out


# ------------------------------------------------------------------ 누적곡선용 일별 수익률

def save_curves(data, collect: dict, three: dict, wf: dict) -> None:
    """모델별 일별 순수익률(게이트 없음, 시드 0)을 저장하고, 본 결과 JSON(시드 0)과 샤프가 같은지 대조한다."""
    ref = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    lo, hi = wf["period"]
    a, b = data.day_range(lo, hi)
    out = {"dates": {"three_way": [str(d.date()) for d in data.days], "walk_forward": [str(d.date()) for d in data.days[a:b + 1]]},
           "returns": {k: {m: np.round(v, 6).tolist() for m, v in d.items()} for k, d in collect.items()}, "checks": {}}
    sh = lambda r: float(r.mean() / r.std(ddof=1) * np.sqrt(252))
    for m in collect["walk_forward"]:
        r = np.asarray(collect["walk_forward"][m])
        want = (ref["walk_forward"]["models"][m]["seeds"]["seed0"]["none"]["overall"]["sharpe"] if m in ref["walk_forward"]["models"]
                else ref["walk_forward"]["baselines"][m]["none"]["overall"]["sharpe"])
        out["checks"][f"wf:{m}"] = {"curve_sharpe": round(sh(r), 3), "json_sharpe": want, "match": abs(sh(r) - want) < 0.01}
    for m in collect["three_way"]:
        r = np.asarray(collect["three_way"][m])
        out["checks"][f"3way_full:{m}"] = {"curve_sharpe": round(sh(r), 3)}
    path = RESULTS_DIR / "week10_curves.json"
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    bad = [k for k, v in out["checks"].items() if v.get("match") is False]
    print(f"\n저장: {path} · 본 결과(시드0) 샤프 재현 {'전부 일치' if not bad else '불일치: ' + str(bad)}")
    for k, v in out["checks"].items():
        if k.startswith("wf:"):
            print(f"  {k:18} 곡선 {v['curve_sharpe']:.3f} / JSON {v['json_sharpe']:.3f}")


# ------------------------------------------------------------------ 메인

def save_checkpoint(keep: dict, sigma: float) -> dict:
    """정규화된 NN(mlp, gru) 중 Val 샤프(게이트 없음) 최고 (모델, 시드) 를 저장. 테스트 성과로 고르지 않는다."""
    cand = [(k, v) for k, v in keep.items() if k[0] != "overfit_mlp"]
    (name, seed), (model, rep) = max(cand, key=lambda kv: kv[1][1]["none"]["val"]["sharpe"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model": name, "seed": seed, "state_dict": model.state_dict(), "features": ds.FEATURES, "seq_len": ds.SEQ_LEN,
                "target_sigma": sigma, "params": n_params(model), "report": rep, "selected_by": "Val Sharpe (gate none)",
                "saved_at": datetime.now().isoformat(timespec="seconds")}, CKPT_PATH)
    return {"model": name, "seed": seed, "val_sharpe": rep["none"]["val"]["sharpe"], "path": str(CKPT_PATH.name)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--curves", action="store_true", help="시드 0 만 재학습해 일별 수익률(누적곡선용)을 results/week10_curves.json 에 저장. 본 결과 JSON/체크포인트는 건드리지 않는다")
    ap.add_argument("--quick", action="store_true", help="에폭을 줄여 동작만 확인")
    a = ap.parse_args()
    if a.quick:
        for s in SPECS.values():
            s["epochs"] = 4
    seeds = [0] if a.curves else list(range(a.seeds))
    collect: dict | None = {} if a.curves else None
    t_all = time.time()
    data = ds.build()
    print(f"torch {torch.__version__} · 평가 {data.days[0].date()} ~ {data.days[-1].date()} · 시드 {seeds} · 헤드라인 게이트 {HEADLINE_GATE}")
    print("\n=== 방식 A: 3단 분할 ===")
    three, keep = run_three_way(data, seeds, collect=collect)
    sigma = three.pop("_sigma")
    base3 = baselines_three_way(data, collect=collect)
    if not a.curves:
        best = save_checkpoint(keep, sigma)
        print(f"  체크포인트: {best}")
    print("\n=== 방식 B: 연도별 확장형 Walk-Forward ===")
    wf = run_walk_forward(data, seeds, collect=collect)
    if a.curves:
        save_curves(data, collect, three, wf)
        return
    res = {"meta": {"saved_at": datetime.now().isoformat(timespec="seconds"), "torch": torch.__version__, "seeds": seeds,
                    "cost_roundtrip": ds.COST, "top_k": ds.TOP_K, "features": ds.FEATURES, "seq_len": ds.SEQ_LEN,
                    "headline_gate": HEADLINE_GATE, "ranges": ds.THREE_WAY,
                    "train": {"optimizer": "AdamW", "lr": LR, "weight_decay": WD, "batch": BATCH, "loss": "HuberLoss(delta=1, y/σ_train)",
                              "early_stopping": f"Val Huber 손실, patience {PATIENCE}, 최적 에폭 복원 (overfit_mlp 제외)"},
                    "quick": a.quick, "seconds": round(time.time() - t_all)},
           "three_way": {"models": three, "baselines": base3, "best_checkpoint": best}, "walk_forward": wf}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {RESULT_PATH} · {CKPT_PATH} · 총 {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
