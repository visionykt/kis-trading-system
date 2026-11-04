"""Week 8: 2020-01-02 ~ 2026-10-21 통일 조건(분할 없음, 왕복 0.10%) 샤프 극대화 — 비ML 탐색 엔진.

규칙
  - 평가 구간 = 2020-01-02 ~ 2026-10-21 의 1,668 거래일. 팩터 워밍업용 2019년 말 데이터는 신호 계산에만 쓴다.
  - 신호는 t 일 장 마감까지의 팩터로 만들고, 수익은 t+1 일 밤 수익률 R_night(t+1) = O(t+1)/C(t) - 1 (= mask.shift(1)).
  - 현금 보유일(수익 0, 비용 0)도 1,668일 모수에 그대로 포함한다. 보유일마다 왕복 0.10% 차감.
  - 샤프 = 일평균/표준편차(ddof=1) x sqrt(252).

사용법 (컨테이너 안):  python week08.py [--top 15]
"""
import argparse
import heapq
import itertools
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd

import factors
import portfolio as pf
import universe
from backtest import RESULTS_DIR

START = pd.Timestamp("2020-01-02")
END = pd.Timestamp("2026-10-21")
COST = 0.001
TRADING_DAYS = 252
EPS = 1e-9

# 탐색 공간 ------------------------------------------------------------------
# 점수 재료: 같은 날 횡단면 백분위 순위. '-x' 는 값이 작을수록 좋은 팩터(순위 뒤집기).
SCORE_POOL = ["vol_ratio", "vol_surge_up", "turnover_rank", "night_sharpe_20", "night_sharpe_60", "night_win_20",
              "night_mean_20", "ret_5d", "ma20_gap", "-upper_shadow", "r_day", "rsi_14"]
KS = [3, 4, 5, 6, 8, 10]
WEIGHTS2 = [(0.4, 0.6), (0.6, 0.4)]  # 2요소 점수의 가중 변형 (Gemini 제안: 0.4*수급 + 0.6*야간샤프 류)


# ------------------------------------------------------------------ 팩터 확장

def ext_factors(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """week7 팩터 9개 + 8주차 확장 팩터. 전부 t 일 장 마감까지의 값 (rolling 은 과거 방향, shift 는 양수만)."""
    F = factors.compute_factors(panel)
    o, h, l, c, v = (panel[k] for k in ("open", "high", "low", "close", "volume"))
    r_night = o / c.shift(1) - 1
    rn = lambda w: r_night.rolling(w, min_periods=w)
    like = c

    def bcast(s: pd.Series) -> pd.DataFrame:  # 시장 전체 값(모든 종목 같은 값)을 패널 모양으로
        return pd.DataFrame(np.repeat(s.to_numpy()[:, None], c.shape[1], axis=1), index=c.index, columns=c.columns)

    d = c.diff()
    up = d.clip(lower=0).rolling(14, min_periods=14).mean()
    dn = (-d.clip(upper=0)).rolling(14, min_periods=14).mean()
    ew_night = r_night.mean(axis=1)
    idx = (c.pct_change(fill_method=None).mean(axis=1).fillna(0) + 1).cumprod()
    F.update({
        "r_day": c / o - 1,
        "vol_surge_up": F["vol_ratio"] * (c > o),  # 거래량 폭발 x 양봉 (샘플 C 신호)
        "turnover_rank": F["turnover"].rank(axis=1, pct=True),
        "night_mean_20": rn(20).mean(),
        "night_sharpe_60": rn(60).mean() / rn(60).std(),
        "ret_20d": c / c.shift(20) - 1,
        "range_pct": (h - l) / c,
        "gap_today": r_night,  # 오늘 아침 갭 (t 일 시가에 확정)
        "rsi_14": 100 - 100 / (1 + up / (dn + EPS)),
        "mkt_night_10": bcast(ew_night.rolling(10, min_periods=10).mean()),
        "mkt_night_20": bcast(ew_night.rolling(20, min_periods=20).mean()),
        "mkt_night_40": bcast(ew_night.rolling(40, min_periods=40).mean()),
        "mkt_ma20_gap": bcast(idx / idx.rolling(20, min_periods=20).mean() - 1),
        "mkt_ma120_gap": bcast(idx / idx.rolling(120, min_periods=120).mean() - 1),
    })
    return F


def regimes(F: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """시장 레짐 게이트(t 일 값, bool). False 면 그날 신호는 전액 현금. NaN(워밍업)은 현금."""
    g = lambda k: (F[k].iloc[:, 0] > 0).to_numpy()
    ma120, ma20, night = g("mkt_ma120_gap"), g("mkt_ma20_gap"), g("mkt_night_20")
    return {"none": np.ones_like(ma120), "ma120": ma120, "night20": night,
            "ma120&night20": ma120 & night, "ma20&night20": ma20 & night,
            "night10": g("mkt_night_10"), "night40": g("mkt_night_40")}


# ------------------------------------------------------------------ 고속 엔진

class Engine:
    """패널을 numpy 로 들고 있다가 (점수, K, 게이트) -> 평가 구간 일 수익률을 빠르게 계산한다."""

    def __init__(self, panel: dict[str, pd.DataFrame], F: dict[str, pd.DataFrame] | None = None,
                 start: pd.Timestamp = START, end: pd.Timestamp = END):
        self.panel = panel
        self.F = F if F is not None else ext_factors(panel)
        self.index = panel["close"].index
        self.cols = list(panel["close"].columns)
        night = pf.returns(panel)["night"].to_numpy()
        self.nxt = np.vstack([night[1:], np.full((1, night.shape[1]), np.nan)])  # nxt[t] = R_night(t+1)
        self.t0 = self.index.get_loc(self.index[self.index >= start][0]) - 1  # 첫 평가일의 수익은 직전 거래일 신호로 번다
        self.t1 = self.index.get_loc(self.index[self.index <= end][-1]) - 1
        self.days = self.index[self.t0 + 1: self.t1 + 2]
        self.gates = {k: v[self.t0:self.t1 + 1] for k, v in regimes(self.F).items()}
        self.ranks: dict[str, np.ndarray] = {}

    def rank(self, name: str) -> np.ndarray:
        if name not in self.ranks:
            sign = -1 if name.startswith("-") else 1
            self.ranks[name] = (sign * self.F[name.lstrip("-")]).rank(axis=1, pct=True).to_numpy()
        return self.ranks[name]

    def pick(self, score: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """점수 상위 k 종목(후보가 부족하면 그만큼만) 동일비중 -> (일 수익률, 보유 여부)."""
        S = np.where(np.isnan(score[self.t0:self.t1 + 1]), -np.inf, score[self.t0:self.t1 + 1])
        X = self.nxt[self.t0:self.t1 + 1]
        top = np.argpartition(-S, k - 1, axis=1)[:, :k]
        ok = np.isfinite(np.take_along_axis(S, top, 1)) & np.isfinite(np.take_along_axis(X, top, 1))
        cnt = ok.sum(1)
        r = np.where(ok, np.take_along_axis(X, top, 1), 0.0).sum(1) / np.maximum(cnt, 1)
        return r, cnt > 0

    def mask(self, score: np.ndarray, k: int, gate: str = "none") -> pd.DataFrame:
        """t 일에 사고 싶은 종목 마스크(전체 구간). mission07 파이프라인과 같은 모양 (수익과 곱할 땐 shift(1))."""
        S = np.where(np.isnan(score), -np.inf, score)
        top = np.argpartition(-S, k - 1, axis=1)[:, :k]
        m = np.zeros(S.shape, bool)
        np.put_along_axis(m, top, True, axis=1)
        m &= np.isfinite(S)
        gate_full = regimes(self.F)[gate]
        return pd.DataFrame(m & gate_full[:, None], index=self.index, columns=self.cols)


def sharpe(r: np.ndarray, inv: np.ndarray, cost: float = COST) -> float:
    net = r - cost * inv
    sd = net.std(ddof=1)
    return float(net.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else -9.0


def fast_metrics(r: np.ndarray, inv: np.ndarray, cost: float = COST) -> dict:
    net = pd.Series(r - cost * inv)
    from backtest import metrics
    m = metrics(net)
    m["calmar"] = m["cumulative_return"] / abs(m["mdd"]) if m["mdd"] else None
    m["invested_days"] = int(inv.sum())
    m["win_rate_invested"] = float((net[inv] > 0).mean()) if inv.any() else None
    return m


# ------------------------------------------------------------------ 비ML 탐색

def filters(eng: Engine) -> dict[str, np.ndarray]:
    F, c, o = eng.F, eng.panel["close"], eng.panel["open"]
    yang = (c > o).to_numpy()
    return {
        "none": np.ones_like(yang),
        "양봉": yang,
        "양봉&윗꼬리<0.3": yang & (F["upper_shadow"] < 0.3).to_numpy(),
        "20일선위": (F["ma20_gap"] > 0).to_numpy(),
        "양봉&거래량>1": yang & (F["vol_ratio"] > 1).to_numpy(),
        "양봉&20일선위": yang & (F["ma20_gap"] > 0).to_numpy(),
    }


def score_space():
    """(이름, 재료 목록, 가중치) 후보: 재료 2~4개 동일가중 + 2개짜리 가중 변형."""
    for n in (2, 3, 4):
        for combo in itertools.combinations(SCORE_POOL, n):
            yield combo, (1.0,) * n
    for combo in itertools.combinations(SCORE_POOL, 2):
        for w in WEIGHTS2:
            yield combo, w


def search(eng: Engine, keep: int = 300, log=print) -> list[dict]:
    """전 구간(2020-01-02~2026-10-21) 샤프(0.10%) 기준 그리드 탐색. 상위 keep 개 반환."""
    flt = filters(eng)
    heap: list[tuple] = []
    n_eval = 0
    t_start = datetime.now()
    for si, (combo, w) in enumerate(score_space()):
        score = sum(wi * eng.rank(f) for wi, f in zip(w, combo))
        for fname, fm in flt.items():
            sc = np.where(fm, score, np.nan)
            for k in KS:
                r, inv = eng.pick(sc, k)
                for gname, g in eng.gates.items():
                    n_eval += 1
                    s = sharpe(r * g, inv & g)
                    item = (s, n_eval, combo, w, fname, k, gname)
                    if len(heap) < keep:
                        heapq.heappush(heap, item)
                    elif s > heap[0][0]:
                        heapq.heapreplace(heap, item)
        if si % 100 == 0:
            log(f"  점수 {si} 번째 · 평가 {n_eval:,}건 · {(datetime.now() - t_start).seconds}s · 현재 1위 샤프 {max(heap)[0]:.3f}")
    out = []
    for s, _, combo, w, fname, k, gname in sorted(heap, reverse=True):
        out.append({"sharpe": s, "factors": list(combo), "weights": list(w), "filter": fname, "k": k, "regime": gname})
    log(f"탐색 완료: {n_eval:,}개 조합")
    return out


REFINE_POOL = ["vol_ratio", "vol_surge_up", "turnover_rank", "night_sharpe_20", "night_sharpe_60", "night_win_20",
               "night_mean_20", "r_day"]


def refine(eng: Engine, keep: int = 200, log=print) -> list[dict]:
    """2차: 1차에서 반복 등장한 재료 8개로 3~5개 조합 x 정수 가중치(1~2) x 필터 2 x K 5~8 x 레짐 7."""
    flt = {k: v for k, v in filters(eng).items() if k in ("none", "양봉")}
    heap: list[tuple] = []
    n_eval = 0
    for n in (3, 4, 5):
        for combo in itertools.combinations(REFINE_POOL, n):
            for w in itertools.product((1.0, 2.0), repeat=n):
                if min(w) == 2.0:  # (2,2,2) 는 (1,1,1) 과 같다
                    continue
                score = sum(wi * eng.rank(f) for wi, f in zip(w, combo))
                for fname, fm in flt.items():
                    sc = np.where(fm, score, np.nan)
                    for k in (5, 6, 7, 8):
                        r, inv = eng.pick(sc, k)
                        for gname, g in eng.gates.items():
                            n_eval += 1
                            sh = sharpe(r * g, inv & g)
                            item = (sh, n_eval, combo, w, fname, k, gname)
                            if len(heap) < keep:
                                heapq.heappush(heap, item)
                            elif sh > heap[0][0]:
                                heapq.heapreplace(heap, item)
        log(f"  {n}개 재료까지 · 평가 {n_eval:,}건 · 현재 1위 샤프 {max(heap)[0]:.3f}")
    return [{"sharpe": s_, "factors": list(c), "weights": list(w), "filter": f, "k": k, "regime": g}
            for s_, _, c, w, f, k, g in sorted(heap, reverse=True)]


def build_score(eng: Engine, cfg: dict) -> np.ndarray:
    score = sum(w * eng.rank(f) for w, f in zip(cfg["weights"], cfg["factors"]))
    return np.where(filters(eng)[cfg["filter"]], score, np.nan)


def evaluate_cfg(eng: Engine, cfg: dict) -> dict:
    """설정 -> 일 수익률/보유 여부와 전 구간·앞/뒤 구간 성과 (앞/뒤는 평가 구간을 반으로 가른 것: 분할 학습이 아니라 안정성 확인용)."""
    r, inv = eng.pick(build_score(eng, cfg), cfg["k"])
    g = eng.gates[cfg["regime"]]
    r, inv = r * g, inv & g
    half = len(r) // 2
    return {"r": r, "inv": inv, "full": fast_metrics(r, inv),
            "first_half_sharpe": sharpe(r[:half], inv[:half]), "second_half_sharpe": sharpe(r[half:], inv[half:])}


def describe(cfg: dict) -> str:
    parts = " + ".join(f"{w:g}x{f}" if cfg["weights"][0] != 1.0 or len(set(cfg["weights"])) > 1 else f for w, f in zip(cfg["weights"], cfg["factors"]))
    return f"[{cfg['filter']}] 후보 중 순위합({parts}) 상위 {cfg['k']}종목 · 레짐 {cfg['regime']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    panel = universe.load_panel(end=END.date())
    eng = Engine(panel)
    print(f"평가 {eng.days[0].date()} ~ {eng.days[-1].date()} ({len(eng.days)}거래일), 워밍업 {eng.t0 + 1}일, 종목 {len(eng.cols)}")
    top = search(eng) + refine(eng)
    top.sort(key=lambda c: -c["sharpe"])
    rows = []
    for cfg in top[:60]:
        ev = evaluate_cfg(eng, cfg)
        rows.append({**cfg, "cum": ev["full"]["cumulative_return"], "mdd": ev["full"]["mdd"], "h1": ev["first_half_sharpe"],
                     "h2": ev["second_half_sharpe"], "inv": ev["full"]["invested_days"]})
    for i, r in enumerate(rows[:args.top], 1):
        print(f"{i:2}. 샤프 {r['sharpe']:.3f} 앞 {r['h1']:.2f}/뒤 {r['h2']:.2f} 누적 {r['cum']:+.0%} MDD {r['mdd']:.1%} 보유 {r['inv']}일 | {describe(r)}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "week08_nonml_search.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
