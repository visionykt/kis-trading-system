"""Week 11 (1단계): 조기 종료 기준 교정 학습 — Val Huber 손실 종료(교정 전) vs Val Daily Spearman IC 종료(교정 후).

변하지 않는 것: 타깃(y_excess / σ_train), 17개 횡단면 순위 피처, 모델 구조, AdamW/Huber 손실로 학습, 3단 분할·Walk-Forward 구간.
변하는 것(variant)
  loss  교정 전  Val Huber 손실 최소, patience 5  (week10_nn.fit 를 그대로 호출 -> 기존 결과 재현 확인용)
  ic    교정 후  Val '일별 Spearman IC 의 평균' 최대, 최소 10 에폭 보장 후 patience 7, 최적 에폭 가중치 복원
점수 행렬 (T_full, N) 을 cache_w11/ 에 저장한다 (분석 단계가 학습 없이 재사용).

사용법 (컨테이너 안):  python week11_train.py --variant ic --seeds 0 1 2 3 4 [--threads 7]
                     python week11_train.py --variant loss --seeds 0 1 2
                     python week11_train.py --variant base      # Ridge / HGB 점수 행렬
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch import nn

import week10_dataset as ds
import week10_nn as nn_

CACHE = Path(__file__).resolve().parent / "cache_w11"
MIN_EPOCHS, PATIENCE_IC, MAX_EPOCHS = 10, 7, 60
MODELS = ("mlp", "gru")


class ValIC:
    """Val 샘플 집합의 '일별 횡단면 Spearman IC 평균' 계산기 (샘플은 t 오름차순으로 정렬돼 있다)."""

    def __init__(self, sset):
        self.t = sset.t
        _, self.start = np.unique(self.t, return_index=True)
        self.end = np.append(self.start[1:], len(self.t))
        y = sset.y
        self.yr = [rankdata(y[a:b]) for a, b in zip(self.start, self.end)]
        self.ok = [b - a >= 10 for a, b in zip(self.start, self.end)]

    def __call__(self, pred: np.ndarray) -> float:
        ics = []
        for (a, b), yr, ok in zip(zip(self.start, self.end), self.yr, self.ok):
            if not ok:
                continue
            pr = rankdata(pred[a:b])
            if pr.std() == 0:
                continue
            ics.append(np.corrcoef(pr, yr)[0, 1])
        return float(np.mean(ics))


def fit_ic(name: str, tr: nn_.Tens, va: nn_.Tens, val_ic: ValIC, seed: int) -> tuple[nn.Module, dict]:
    """week10_nn.fit 과 같은 학습 루프. 종료 기준만 Val 일별 IC 최대화로 교체 (손실은 기록용)."""
    spec = nn_.SPECS[name]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = spec["cls"]()
    opt = torch.optim.AdamW(model.parameters(), lr=nn_.LR, weight_decay=spec["wd"])
    loss_fn = nn.HuberLoss(delta=1.0)
    gen = torch.Generator().manual_seed(seed)
    hist = {"train_loss": [], "val_loss": [], "val_ic": []}
    best, best_state, bad = -float("inf"), None, 0
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(tr.n, generator=gen)
        for i in range(0, tr.n, nn_.BATCH):
            idx = perm[i:i + nn_.BATCH]
            opt.zero_grad()
            loss_fn(model(tr.X[idx]), tr.y[idx]).backward()
            opt.step()
        hist["train_loss"].append(nn_.eval_loss(model, tr, loss_fn))
        hist["val_loss"].append(nn_.eval_loss(model, va, loss_fn))
        ic = val_ic(nn_.predict(model, va.X))
        hist["val_ic"].append(ic)
        if ic > best + 1e-6:
            best, best_state, bad = ic, copy.deepcopy(model.state_dict()), 0
            hist["best_epoch"] = ep
        else:
            bad += 1
            if ep >= MIN_EPOCHS and bad >= PATIENCE_IC:
                break
    model.load_state_dict(best_state)
    hist["epochs_run"] = len(hist["train_loss"])
    return model, hist


def train_one(variant: str, name: str, tr_set, va_set, seed: int):
    sigma = float(tr_set.y.std())
    Ttr, Tva = nn_.Tens(tr_set, sigma), nn_.Tens(va_set, sigma)
    if variant == "ic":
        return fit_ic(name, Ttr, Tva, ValIC(va_set), seed), sigma
    return nn_.fit(name, Ttr, Tva, seed), sigma


def run(variant: str, seeds: list[int], data: ds.Week10Data, only: list[str] | None = None) -> None:
    CACHE.mkdir(exist_ok=True)
    log = {}
    log_path = CACHE / f"train_log_{variant}.json"
    if log_path.exists():
        log = json.loads(log_path.read_text(encoding="utf-8"))
    sets = {m: data.three_way(m) for m in ("2D", "3D")}
    folds = {m: data.walk_forward(m) for m in ("2D", "3D")}
    for name in (only or MODELS):
        mode = nn_.SPECS[name]["mode"]
        for seed in seeds:
            t0 = time.time()
            # --- 방식 A
            key = f"{variant}_{name}_A_s{seed}"
            if not (CACHE / f"{key}.npy").exists():
                s = sets[mode]
                (model, hist), sigma = train_one(variant, name, s["train"], s["val"], seed)
                parts = [(s[k], nn_.predict(model, nn_.Tens(s[k], sigma).X)) for k in ("train", "val", "test")]
                np.save(CACHE / f"{key}.npy", nn_.score_matrix(data, parts).astype(np.float32))
                log[key] = {"best_epoch": hist["best_epoch"], "epochs_run": hist["epochs_run"],
                            "val_ic": hist.get("val_ic"), "val_loss": hist["val_loss"]}
                print(f"[{variant}] {name} A seed{seed}: best/run {hist['best_epoch']}/{hist['epochs_run']}", flush=True)
            # --- 방식 B
            key = f"{variant}_{name}_B_s{seed}"
            if not (CACHE / f"{key}.npy").exists():
                S = np.full(data.cube.shape[:2], np.nan, dtype=np.float32)
                fl = {}
                for fold in folds[mode]:
                    (model, hist), sigma = train_one(variant, name, fold["train"], fold["val"], seed)
                    te = fold["test"]
                    S[te.t, te.i] = nn_.predict(model, nn_.Tens(te, sigma).X)
                    fl[fold["year"]] = {"best_epoch": hist["best_epoch"], "epochs_run": hist["epochs_run"]}
                np.save(CACHE / f"{key}.npy", S)
                log[key] = fl
                print(f"[{variant}] {name} B seed{seed}: " + " ".join(f"{y}:{v['best_epoch']}/{v['epochs_run']}" for y, v in fl.items()), flush=True)
            log_path.write_text(json.dumps(log, ensure_ascii=False), encoding="utf-8")
            print(f"  -> {variant} {name} seed{seed} 완료 {time.time() - t0:.0f}s", flush=True)


def run_base(data: ds.Week10Data) -> None:
    CACHE.mkdir(exist_ok=True)
    s2 = data.three_way("2D")
    folds = data.walk_forward("2D")
    for name in ("ridge", "hgb"):
        preds = nn_.sk_fit_predict(name, s2["train"], [s2["train"], s2["val"], s2["test"]])
        np.save(CACHE / f"base_{name}_A.npy", nn_.score_matrix(data, list(zip((s2["train"], s2["val"], s2["test"]), preds))).astype(np.float32))
        S = np.full(data.cube.shape[:2], np.nan, dtype=np.float32)
        for f in folds:
            te = f["test"]
            S[te.t, te.i] = nn_.sk_fit_predict(name, f["train"], [te])[0]
        np.save(CACHE / f"base_{name}_B.npy", S)
        print(f"[base] {name} 저장", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["ic", "loss", "base"], required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t = time.time()
    data = ds.build()
    print(f"데이터 준비 {time.time() - t:.0f}s · 스레드 {a.threads}", flush=True)
    run_base(data) if a.variant == "base" else run(a.variant, a.seeds, data, a.models)
    print(f"전체 {time.time() - t:.0f}s")
