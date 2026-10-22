"""Week 7 MISSION 01: OHLCV 옆에 붙이는 파생 팩터 컬럼 (4그룹, 9개).

입력은 두 가지 모양을 모두 받는다 (연산이 전부 원소별/rolling 이라 같은 코드로 동작).
  - 패널:  {'open','high','low','close','volume'} -> DataFrame(index=ts, columns=symbol)   (universe.load_panel)
  - 단일 종목: columns 에 open/high/low/close/volume 을 가진 DataFrame(index=ts)            (candles.load_daily_candles)

미래 참조 규칙: factor(t) 는 t 일 15:20 까지 알 수 있는 값(<= t 일 OHLCV)만으로 계산한다.
  - 모든 rolling 은 과거 방향 창(현재 포함)이고 shift 는 양수(과거)만 쓴다. 음수 shift / center=True 금지.
  - R_night(t) = O(t)/C(t-1) - 1 은 t 일 시가에 확정되므로 t 일 장 마감 시점에 이미 알려져 있다.
  - 이 팩터로 고른 종목의 수익은 다음 날 밤 수익률 R_night(t+1) 이므로 수익률과 곱할 때는
    portfolio.signal() (= top_k_mask(score).shift(1)) 를 거친다. 이 모듈은 신호가 아니라 팩터 값만 만든다.
"""
import numpy as np
import pandas as pd

EPS = 1e-9
WINDOW = 20
OHLCV = ("open", "high", "low", "close", "volume")

# 이름 -> (그룹, 수식, 가설) : 프론트엔드/보고서에서 그대로 표시한다
FACTORS = {
    "close_pos": ("캔들 모양", "(close-low)/(high-low+1e-9)", "1에 가까울수록 고가 근처 마감 = 장 막판까지 매수세 유지"),
    "upper_shadow": ("캔들 모양", "(high-max(open,close))/(high-low+1e-9)", "클수록 장 막판 매물 출회 -> 제거 대상"),
    "body_ratio": ("캔들 모양", "|close-open|/(high-low+1e-9)", "클수록 꽉 찬 캔들 = 방향성이 뚜렷한 하루"),
    "turnover": ("수급·거래대금", "close*volume", "실제로 큰돈이 몰린 종목 (유동성 필터)"),
    "vol_ratio": ("수급·거래대금", "volume/MA20(volume)", "평소 대비 거래량 폭발 = 수급 주도주"),
    "ma20_gap": ("추세·이격도", "close/MA20(close)-1", "양수 = 20일선 위 상승 추세, 과도하면 과열"),
    "ret_5d": ("추세·이격도", "close/close[t-5]-1", "최근 5일 단기 모멘텀"),
    "night_win_20": ("야간 관성", "mean(R_night>0, 20일)", "밤마다 갭상승이 반복되는 종목의 고유 습성"),
    "night_sharpe_20": ("야간 관성", "mean(R_night,20)/std(R_night,20)", "수익/변동성 기준으로 꾸준히 밤에 오르는 종목"),
}


def _std_ohlcv(data) -> dict:
    """DataFrame(단일 종목) 또는 패널 dict -> {'open',...} dict."""
    if isinstance(data, dict):
        missing = [c for c in OHLCV if c not in data]
        src = data
    else:
        missing = [c for c in OHLCV if c not in data.columns]
        src = {c: data[c] for c in OHLCV if c in data.columns}
    if missing:
        raise ValueError(f"OHLCV 컬럼 누락: {missing}")
    out = {c: src[c].astype(float) for c in OHLCV}
    if not out["close"].index.is_monotonic_increasing:
        raise ValueError("ts 오름차순이어야 한다 (rolling/shift 가 과거 방향이 되려면 필수)")
    return out


def compute_factors(data) -> dict:
    """OHLCV -> {factor_name: 같은 모양의 DataFrame/Series}. 창이 덜 찬 초반은 NaN (min_periods=window)."""
    p = _std_ohlcv(data)
    o, h, l, c, v = (p[k] for k in OHLCV)
    rng = h - l + EPS  # 고가=저가(상·하한가 등) 이어도 0 으로 나누지 않는다 -> 해당 비율은 0
    roll = lambda x: x.rolling(WINDOW, min_periods=WINDOW)
    r_night = o / c.shift(1) - 1  # 첫 행은 전일 종가가 없어 NaN
    return {
        "close_pos": (c - l) / rng,
        "upper_shadow": (h - np.maximum(o, c)) / rng,
        "body_ratio": (c - o).abs() / rng,
        "turnover": c * v,
        "vol_ratio": v / roll(v).mean(),
        "ma20_gap": c / roll(c).mean() - 1,
        "ret_5d": c / c.shift(5) - 1,
        # NaN 비교는 False 라 첫 날(NaN)이 '하락'으로 세지는 것을 막으려고 notna 로 마스킹한 뒤 rolling
        "night_win_20": roll((r_night > 0).astype(float).where(r_night.notna())).mean(),
        "night_sharpe_20": roll(r_night).mean() / roll(r_night).std(),
    }


def add_factor_columns(df: pd.DataFrame) -> pd.DataFrame:
    """단일 종목 OHLCV DataFrame 옆에 팩터 컬럼 9개를 붙여 돌려준다 (원본은 건드리지 않음)."""
    return df.join(pd.DataFrame(compute_factors(df)))


def factor_panel(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """유니버스 패널 -> {factor: DataFrame(날짜 x 종목)}. 전략 점수(score)의 재료로 쓴다."""
    return compute_factors(panel)

