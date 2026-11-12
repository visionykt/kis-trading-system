"""Week 10 (1단계): 뉴럴 네트워크용 데이터셋 — 8·9주차와 동일 규격, 누수 없는 2D / 3D 텐서.

통일 규격
  - 매매: t 일(calendar[i-1]) 종가 매수 -> t+1 일(calendar[i]) 시가 매도, 왕복 0.10%, Top-5 동일비중. 현금일(0%)도 모수에 포함.
  - 입력 X: t 일 장 마감까지 확정된 종목별 팩터 17개를 '같은 날 100종목 횡단면 백분위 순위 - 0.5' (-0.5 ~ +0.5) 로 변환.
      스케일러(fit)를 쓰지 않으므로 미래 평균·분산이 과거로 새지 않는다 (순위는 그날 횡단면만 본다).
      시장 공통 팩터(mkt_*)는 횡단면 순위가 상수라 정보가 없고 날짜 바코드 암기의 원인이라 입력에서 뺀다
      (9주차 처방 ①: 레짐은 신경망 입력이 아니라 앞단 게이트로만 사용 -> data.gates).
  - 타깃 y_excess(i,t) = R_night(i,t+1) - mean_i R_night(t+1)  (시장 중립 초과수익률). y_rank = 날짜별 순위 - 0.5.
  - 구간 소속은 '수익이 실현되는 날(= t+1)' 기준. 라벨이 t+1 시가 하나뿐이라 경계에서 라벨이 구간을 넘나들지 않는다.

텐서
  2D (N, F)    : TabularSet   — MLP 용
  3D (N, L, F) : SequenceSet  — LSTM/GRU/1D-CNN/Transformer 용 (L=20, 윈도우 = t-L+1 .. t 일 팩터 순위; 전 구간 값이 유한한 샘플만)

검증 구간
  three_way():    Train 2020-01~2023-12 / Val 2024 / OOS Test 2025-01~2026-10-21
  walk_forward(): 각 테스트 연도 Y(2022~2026) 마다 'Y 이전 데이터만' 학습 (마지막 6개월 = Validation, 나머지 = Train)

사용법 (컨테이너 안):  python week10_dataset.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import universe
import week08 as w8

FEATURES = ["close_pos", "upper_shadow", "body_ratio", "vol_ratio", "vol_surge_up", "turnover_rank", "ma20_gap", "ret_5d",
            "ret_20d", "range_pct", "r_day", "gap_today", "rsi_14", "night_win_20", "night_mean_20", "night_sharpe_20",
            "night_sharpe_60"]
SEQ_LEN = 20
TOP_K = 5
COST = w8.COST  # 왕복 0.10%

# 3단 분할 (수익 실현일 기준, 양끝 포함)
THREE_WAY = {"train": ("2020-01-01", "2023-12-31"), "val": ("2024-01-01", "2024-12-31"), "test": ("2025-01-01", "2026-10-21")}
WF_YEARS = [2022, 2023, 2024, 2025, 2026]
WF_VAL_MONTHS = 6


# ------------------------------------------------------------------ 샘플 집합 (torch Dataset 호환: __len__/__getitem__)

class _SampleSet:
    def __init__(self, cube: np.ndarray, t: np.ndarray, i: np.ndarray, y: np.ndarray):
        self.cube, self.t, self.i, self.y = cube, t, i, y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.t)

    def x_batch(self, sel: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __getitem__(self, k):
        sel = np.atleast_1d(k)
        x, y = self.x_batch(sel), self.y[sel]
        return (x[0], y[0]) if np.isscalar(k) or np.ndim(k) == 0 else (x, y)

    def tensors(self):
        """(X, y) 를 torch.Tensor 로 한 번에 (샘플 수가 크면 메모리 주의: 3D 는 N*20*17*4B)."""
        import torch
        return torch.from_numpy(self.x_batch(np.arange(len(self)))), torch.from_numpy(self.y)

    def loader(self, batch_size: int = 1024, shuffle: bool = False, seed: int = 0):
        """윈도우를 배치 단위로만 만들어 메모리를 아끼는 torch DataLoader (3D 에서도 안전)."""
        import torch
        from torch.utils.data import DataLoader, Dataset

        outer = self

        class _DS(Dataset):
            def __len__(self):
                return len(outer)

            def __getitem__(self, k):
                x, y = outer[k]
                return torch.from_numpy(np.asarray(x)), torch.tensor(y)

        gen = torch.Generator().manual_seed(seed)
        return DataLoader(_DS(), batch_size=batch_size, shuffle=shuffle, generator=gen)


class TabularSet(_SampleSet):
    """2D 입력 (N, F): 당일 t 의 팩터 순위."""
    kind = "2D"

    def x_batch(self, sel):
        return self.cube[self.t[sel], self.i[sel]]


class SequenceSet(_SampleSet):
    """3D 입력 (N, L, F): t-L+1 .. t 일의 팩터 순위 궤적 (맨 뒤가 당일)."""
    kind = "3D"

    def __init__(self, cube, t, i, y, seq_len: int = SEQ_LEN):
        super().__init__(cube, t, i, y)
        self.seq_len = seq_len

    def x_batch(self, sel):
        tt = self.t[sel][:, None] - np.arange(self.seq_len - 1, -1, -1)[None, :]  # (B, L)
        return self.cube[tt, self.i[sel][:, None]]  # (B, L, F)


# ------------------------------------------------------------------ 데이터

@dataclass
class Week10Data:
    eng: w8.Engine
    features: list[str]
    cube: np.ndarray        # (T_full, N, F) float32, 순위 - 0.5, 계산 불가는 NaN
    y_excess: np.ndarray    # (T_full, N) t 일 신호의 정답 = R_night(t+1) - 횡단면 평균
    y_rank: np.ndarray      # (T_full, N) 같은 날 횡단면 순위 - 0.5
    ok_2d: np.ndarray       # (T_full, N) bool: t 일 피처가 전부 유한
    ok_3d: np.ndarray       # (T_full, N) bool: t-L+1..t 일 피처가 전부 유한
    seq_len: int

    # --- 평가 달력 (수익 실현일 기준)
    @property
    def days(self) -> pd.DatetimeIndex:
        return self.eng.days

    @property
    def gates(self) -> dict[str, np.ndarray]:
        return self.eng.gates

    def day_range(self, start: str, end: str) -> tuple[int, int]:
        """[start, end] 에 실현일이 들어가는 평가일 위치 (k_lo, k_hi) 포함. 비면 (0, -1)."""
        pos = np.where((self.days >= pd.Timestamp(start)) & (self.days <= pd.Timestamp(end)))[0]
        return (int(pos[0]), int(pos[-1])) if len(pos) else (0, -1)

    def samples(self, start: str, end: str, mode: str = "2D", target: str = "excess", require_target: bool = True) -> _SampleSet:
        """실현일이 [start, end] 인 (t, 종목) 샘플. 피처/타깃이 NaN 인 샘플은 제외한다."""
        lo, hi = self.day_range(start, end)
        if hi < lo:
            raise ValueError(f"구간 {start}~{end} 에 평가일이 없다")
        t = np.arange(lo, hi + 1) + self.eng.t0  # 신호일 위치 (큐브 인덱스)
        y = {"excess": self.y_excess, "rank": self.y_rank}[target]
        ok = (self.ok_2d if mode == "2D" else self.ok_3d)[t]
        if require_target:
            ok = ok & np.isfinite(y[t])
        ti, ii = np.nonzero(ok)
        tt = t[ti]
        cls = TabularSet if mode == "2D" else SequenceSet
        args = (self.cube, tt, ii, y[tt, ii])
        return cls(*args) if mode == "2D" else cls(*args, seq_len=self.seq_len)

    # --- 검증 구간
    def three_way(self, mode: str = "2D", target: str = "excess") -> dict[str, _SampleSet]:
        return {k: self.samples(a, b, mode, target) for k, (a, b) in THREE_WAY.items()}

    def walk_forward_plan(self, years=WF_YEARS, val_months: int = WF_VAL_MONTHS, rolling_years: int | None = None) -> list[dict]:
        """연도별 (train, val, test) 날짜 구간. 전부 test 연도 시작일 '이전'만 학습/검증에 쓴다.
        rolling_years=None 이면 2020-01 부터 누적(expanding), 숫자면 그 햇수의 롤링 창."""
        first = self.days[0]
        plan = []
        for y in years:
            t_start = pd.Timestamp(f"{y}-01-01")
            if t_start > self.days[-1]:
                continue
            val_start = t_start - pd.DateOffset(months=val_months)
            tr_start = first if rolling_years is None else max(first, t_start - pd.DateOffset(years=rolling_years))
            d = lambda x: x.strftime("%Y-%m-%d")
            plan.append({"year": y, "train": (d(tr_start), d(val_start - pd.Timedelta(days=1))),
                         "val": (d(val_start), d(t_start - pd.Timedelta(days=1))),
                         "test": (f"{y}-01-01", f"{y}-12-31" if y < self.days[-1].year else d(self.days[-1]))})
        return plan

    def walk_forward(self, mode: str = "2D", target: str = "excess", **plan_kw) -> list[dict]:
        out = []
        for p in self.walk_forward_plan(**plan_kw):
            out.append({"year": p["year"], **{k: self.samples(*p[k], mode, target) for k in ("train", "val", "test")},
                        "ranges": {k: p[k] for k in ("train", "val", "test")}})
        return out

    # --- 백테스트 (8주차 엔진과 같은 규칙; 점수 (T_full, N) -> 구간 성과)
    def backtest(self, scores: np.ndarray, start: str | None = None, end: str | None = None, k: int = TOP_K,
                 gate: str = "none") -> dict:
        """scores[t, i] = t 일 신호. Top-k 동일비중, 왕복 0.10%, 현금일 포함 샤프/누적/MDD/승률. start~end 는 실현일 기준."""
        r, inv = self.eng.pick(scores, k)
        g = self.gates[gate]
        r, inv = r * g, inv & g
        lo, hi = self.day_range(start or str(self.days[0].date()), end or str(self.days[-1].date()))
        return {"days": hi - lo + 1, "r": r[lo:hi + 1], "inv": inv[lo:hi + 1], "metrics": w8.fast_metrics(r[lo:hi + 1], inv[lo:hi + 1])}


def build(panel: dict[str, pd.DataFrame] | None = None, seq_len: int = SEQ_LEN, start: pd.Timestamp = w8.START,
          end: pd.Timestamp = w8.END) -> Week10Data:
    """패널(없으면 DB 의 KOSPI100 일봉)에서 2D/3D 공용 큐브와 정답을 만든다."""
    if panel is None:
        panel = universe.load_panel(end=end.date())
    eng = w8.Engine(panel, start=start, end=end)
    cols = []
    for f in FEATURES:
        rk = eng.F[f].rank(axis=1, pct=True).to_numpy() - 0.5  # 같은 날 횡단면 백분위 순위 (NaN 은 그대로)
        cols.append(rk)
    cube = np.stack(cols, axis=2).astype(np.float32)  # (T_full, N, F)

    nxt = eng.nxt  # nxt[t] = R_night(t+1)
    cnt = np.isfinite(nxt).sum(1, keepdims=True)  # 마지막 날(t+1 없음)은 전부 NaN -> 평균도 NaN
    mean_t = np.where(cnt > 0, np.nansum(nxt, axis=1, keepdims=True) / np.maximum(cnt, 1), np.nan)
    y_excess = nxt - mean_t
    y_rank = pd.DataFrame(nxt).rank(axis=1, pct=True).to_numpy() - 0.5

    fin = np.isfinite(cube).all(2)  # (T, N)
    c = np.cumsum(np.vstack([np.zeros((1, fin.shape[1])), fin]), axis=0)  # 윈도우 안 유한 개수 = 구간합
    ok_3d = np.zeros_like(fin)
    ok_3d[seq_len - 1:] = (c[seq_len:] - c[:-seq_len]) == seq_len
    return Week10Data(eng, list(FEATURES), cube, y_excess, y_rank, fin, ok_3d, seq_len)


# ------------------------------------------------------------------ 요약 출력

def summarize(data: Week10Data) -> str:
    L = []
    eng = data.eng
    L.append(f"평가 {eng.days[0].date()} ~ {eng.days[-1].date()} ({len(eng.days)}거래일) · 종목 {len(eng.cols)} · 피처 F={len(data.features)}")
    L.append(f"큐브 {data.cube.shape} · 값 범위 [{np.nanmin(data.cube):.3f}, {np.nanmax(data.cube):.3f}] · y_excess 일평균 |mean| 최대 "
             f"{np.nanmax(np.abs(np.nanmean(data.y_excess[eng.t0:eng.t1 + 1], axis=1))):.2e}")
    L.append("\n[3단 분할]")
    for mode in ("2D", "3D"):
        s = data.three_way(mode)
        for k, v in s.items():
            sh = tuple(v.x_batch(np.arange(1)).shape)
            full = (len(v),) + sh[1:] if mode == "2D" else (len(v),) + sh[1:]
            a, b = THREE_WAY[k]
            L.append(f"  {mode} {k:5} {a}~{b}  X{full}  y({len(v)},)")
    L.append("\n[연도별 Walk-Forward (expanding)]")
    for p in data.walk_forward_plan():
        n2 = {k: len(data.samples(*p[k], '2D')) for k in ('train', 'val', 'test')}
        n3 = {k: len(data.samples(*p[k], '3D')) for k in ('train', 'val', 'test')}
        L.append(f"  {p['year']}: train {p['train'][0]}~{p['train'][1]} | val {p['val'][0]}~{p['val'][1]} | test {p['test'][0]}~{p['test'][1]}")
        L.append(f"        2D 샘플 {n2} · 3D 샘플 {n3}")
    return "\n".join(L)


if __name__ == "__main__":
    import torch
    print(f"torch {torch.__version__} · cuda {torch.cuda.is_available()}")
    d = build()
    print(summarize(d))
