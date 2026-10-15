"""누적수익률 라인 차트 (matplotlib, 없으면 건너뜀).

범주색은 고정 순서로만 쓰고(색 = 전략 identity), 끝점 직접 라벨 + 범례로 색 외에도 구분되게 한다.
"""
import math
from pathlib import Path

import pandas as pd

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BASELINE = "#52514e"  # 비교 기준선(중립 회색, 점선)
SEQUENTIAL_BLUE = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # 연하게 -> 진하게


def _spread(values: list[float], gap: float) -> list[float]:
    """오름차순 값들을 최소 gap 간격으로 위로 밀어 라벨 겹침을 막는다."""
    out: list[float] = []
    for v in values:
        out.append(v if not out or v - out[-1] >= gap else out[-1] + gap)
    return out


def line_chart(path: Path, panels: list[dict], title: str, log: bool = False) -> bool:
    """panels: [{"title", "series": [{"label", "y": 누적배수 Series, "color", "ls"}]}] 를 가로로 나란히 그린다."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels) + 3, 5.8), dpi=120, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, panel in zip(axes[0], panels):
        ax.set_facecolor(SURFACE)
        ax.axhline(1 if log else 0, color=INK_2, lw=1, ls=":")
        for s in panel["series"]:
            y = s["y"] if log else (s["y"] - 1) * 100
            ax.plot(y.index, y, color=s["color"], lw=2, ls=s.get("ls", "-"), label=s["label"])
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{(v - 1) * 100:+,.0f}%"))
        # 끝점 직접 라벨 (축 오른쪽 바깥, 겹치면 위로 밀기)
        to_axis = (lambda v: math.log10(v)) if log else (lambda v: v)
        from_axis = (lambda v: 10 ** v) if log else (lambda v: v)
        lo, hi = (to_axis(v) for v in ax.get_ylim())
        ends = sorted(((to_axis(float(s["y"].iloc[-1] if log else (s["y"].iloc[-1] - 1) * 100)), s)
                       for s in panel["series"]), key=lambda e: e[0])
        for pos, (v, s) in zip(_spread([e[0] for e in ends], (hi - lo) * 0.05), ends):
            ret = (s["y"].iloc[-1] - 1) * 100
            ax.annotate(f"{s['label']}  {ret:+,.0f}%", xy=(1.0, from_axis(v)), xytext=(1.02, from_axis(pos)),
                        xycoords=("axes fraction", "data"), textcoords=("axes fraction", "data"),
                        va="center", fontsize=8.5, color=INK, annotation_clip=False)
        ax.set_title(panel.get("title", ""), loc="left", fontsize=10.5, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK_2, labelsize=9)
        ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    axes[0][0].set_ylabel("Cumulative return" + (" (log scale)" if log else " (%)"), color=INK_2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.subplots_adjust(right=1 - 2.4 / fig.get_figwidth(), wspace=2.6 / fig.get_figwidth() * len(panels))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return True


def cum(r: pd.Series) -> pd.Series:
    return (1 + r).cumprod()
