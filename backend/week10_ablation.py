"""Week 10 후속 대조 실험: 시장 공통 변수(mkt_*)를 다시 넣으면 비규제 MLP 의 IS→OOS 붕괴가 나타나는가?

가설(리포트 3-①): 비규제 Overfit_MLP 가 붕괴하지 않은 이유는 날짜 바코드(시장 공통 변수)를 입력에서 뺐기 때문이다.
설계: Overfit_MLP(128x3, Dropout 0, wd 0, Early Stopping 없음, 60 에폭)를 week10_nn 과 '완전히 같은 조건'(데이터·분할·타깃·옵티마이저·시드)으로 두고
      입력에 시장 공통 변수 3개(mkt_night_20, mkt_ma20_gap, mkt_ma120_gap)만 원값으로 추가한다 (F 17 -> 20). 같은 날 모든 종목이 같은 값이다.
      표준화 평균/표준편차는 Train 구간에서만 fit (누수 없음).
비교 기준: 이미 저장된 기존 Overfit_MLP(mkt 없음) 결과 (results/week10_nn_results.json, 같은 시드 0,1,2).
테스트하지 않는 것: 횡단면 순위 정규화의 단독 효과 (원값 피처 대조는 하지 않았다).

사용법 (컨테이너 안):  python week10_ablation.py     # 약 2분 -> results/week10_ablation_mkt.json
"""
import json
import time

import numpy as np
import torch

import week10_dataset as ds
import week10_nn as nn_

MKT = ["mkt_night_20", "mkt_ma20_gap", "mkt_ma120_gap"]
SEEDS = [0, 1, 2]
SNAP_EPOCHS = (1, 2, 3, 5, 8, 12, 20, 30, 45, 60)
RESULT = nn_.RESULTS_DIR / "week10_ablation_mkt.json"


class MktSet:
    """TabularSet + 시장 공통 변수 3개 (NaN 인 날은 제외)."""

    def __init__(self, sset, mkt_by_day: np.ndarray, mu=None, sd=None):
        X = sset.x_batch(np.arange(len(sset)))
        m = mkt_by_day[sset.t]
        ok = np.isfinite(m).all(1)
        self.t, self.i, self.y = sset.t[ok], sset.i[ok], sset.y[ok]
        self.m = m[ok]
        self.X17 = X[ok]
        self.mu = self.m.mean(0) if mu is None else mu  # Train 로만 fit
        self.sd = self.m.std(0) if sd is None else sd

    def __len__(self):
        return len(self.t)


class MktTens:
    def __init__(self, ms: MktSet, sigma: float):
        X = np.concatenate([ms.X17, (ms.m - ms.mu) / ms.sd], axis=1).astype(np.float32)
        self.X, self.y, self.n = torch.from_numpy(X), torch.from_numpy(ms.y) / sigma, len(ms)


def main() -> None:
    t_all = time.time()
    data = ds.build()
    mkt = np.stack([data.eng.F[k].iloc[:, 0].to_numpy() for k in MKT], axis=1)  # (T_full, 3) 날짜별 값
    sets = data.three_way("2D")
    sigma = float(sets["train"].y.std())
    tr_ms = MktSet(sets["train"], mkt)
    ms = {"train": tr_ms, "val": MktSet(sets["val"], mkt, tr_ms.mu, tr_ms.sd), "test": MktSet(sets["test"], mkt, tr_ms.mu, tr_ms.sd)}
    T = {k: MktTens(v, sigma) for k, v in ms.items()}
    print(f"입력 F={T['train'].X.shape[1]} (17 순위 피처 + 시장 공통 {len(MKT)}) · Train {len(tr_ms):,} / Val {len(ms['val']):,} / Test {len(ms['test']):,}")

    nn_.SPECS["overfit_mlp_mkt"] = {**nn_.SPECS["overfit_mlp"], "cls": lambda: nn_.OverfitMLP(f=len(ds.FEATURES) + len(MKT))}
    ranges = ds.THREE_WAY

    def report(model):
        S = nn_.score_matrix(data, [(ms[k], nn_.predict(model, T[k].X)) for k in ("train", "val", "test")])
        return nn_.period_report(data, S, ranges)

    runs = []
    for seed in SEEDS:
        t0, snaps = time.time(), {}

        def cb(ep, model):
            if ep in SNAP_EPOCHS:
                r = report(model)
                snaps[ep] = {k: r[k]["sharpe"] for k in r}

        model, hist = nn_.fit("overfit_mlp_mkt", T["train"], T["val"], seed, cb)
        r = report(model)
        runs.append({"seed": seed, "params": nn_.n_params(model), "report": r, "history": hist, "epoch_sharpe": snaps})
        print(f"  seed {seed} · 파라미터 {nn_.n_params(model):,} · 샤프 IS {r['train']['sharpe']:.2f} Val {r['val']['sharpe']:.2f} OOS {r['test']['sharpe']:.2f} · "
              f"Train손실 {hist['train_loss'][-1]:.4f} Val손실 {hist['val_loss'][-1]:.4f} · {time.time() - t0:.0f}s")

    base = json.loads(nn_.RESULT_PATH.read_text(encoding="utf-8"))["three_way"]["models"]["overfit_mlp"]["runs"]
    out = {"design": "Overfit_MLP 동일 조건, 입력만 17 -> 20 (시장 공통 변수 3개 원값 추가)", "mkt_features": MKT, "seeds": SEEDS,
           "with_mkt": runs, "baseline_no_mkt": [{"seed": r["seed"], "params": r["params"], "report": r["report"]["none"], "history": r["history"],
                                                  "epoch_sharpe": r.get("epoch_sharpe", {})} for r in base],
           "seconds": round(time.time() - t_all)}
    RESULT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    mean = lambda rs, seg: float(np.mean([r["report"][seg]["sharpe"] for r in rs]))
    print("\n시드 평균 샤프 (IS / Val / OOS)")
    print(f"  mkt 제외(기존) : {mean(out['baseline_no_mkt'], 'train'):.2f} / {mean(out['baseline_no_mkt'], 'val'):.2f} / {mean(out['baseline_no_mkt'], 'test'):.2f}")
    print(f"  mkt 추가(대조) : {mean(runs, 'train'):.2f} / {mean(runs, 'val'):.2f} / {mean(runs, 'test'):.2f}")
    print(f"저장: {RESULT} · {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
