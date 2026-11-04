"""Week 8 MISSION 03: 비ML 최고 vs ML 최고 vs 벤치마크 · 샘플 A~D 한 화면 비교 데이터 (리더보드 · 일별 수익률 · 검증 · 선정 근거).

고속 탐색 엔진(week08.Engine)이 고른 설정을 Week 7 의 독립 파이프라인(mission07.run_mask / score_metrics)으로 다시 계산해
두 계산이 같은지(engine_crosscheck), 미래 데이터를 망가뜨려도 과거 신호가 안 변하는지(lookahead) 함께 증명한다.

사용법 (컨테이너 안):  python week08_report.py      # week08.py, week08_ml.py 를 먼저 실행해 탐색 결과/모델이 있어야 한다
"""
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

import mission07 as m7
import portfolio as pf
import universe
import week08 as w8
import week08_ml as ml
from backtest import RESULTS_DIR

COSTS = [0.0, 0.0005, 0.001, 0.002]
NONML, ML = "W8_NONML_BEST", "W8_ML_BEST"
SAMPLES = ["A_momentum", "B_mean_reversion", "C_volume_surge", "D_rolling_edge"]
EXPLAIN_COLS = ["close_pos", "upper_shadow", "vol_ratio", "turnover_rank", "night_sharpe_20", "night_sharpe_60", "r_day"]

FACTOR_DOC = {
    "vol_ratio": ("수급", "volume / MA20(volume)", "평소보다 거래가 터진 날 = 새 수급이 들어온 날"),
    "vol_surge_up": ("수급", "vol_ratio x I(양봉)", "거래량이 터졌는데 양봉 = 사려는 쪽이 이긴 수급 (샘플 C 신호)"),
    "turnover_rank": ("수급", "그날 100종목 중 거래대금 백분위", "시장의 큰돈이 몰린 종목"),
    "night_sharpe_20": ("야간 관성", "mean(R_night,20) / std(R_night,20)", "수익/변동성 기준으로 꾸준히 밤에 오르는 종목"),
    "night_sharpe_60": ("야간 관성", "mean(R_night,60) / std(R_night,60)", "더 긴 기간의 밤 습성 (잡음이 적다)"),
    "night_win_20": ("야간 관성", "mean(R_night>0, 20일)", "밤마다 갭상승이 반복되는 종목"),
    "night_mean_20": ("야간 관성", "mean(R_night, 20)", "최근 밤 평균 수익"),
    "r_day": ("캔들", "C / O - 1", "오늘 낮 수익 = 장중 매수 우위"),
    "upper_shadow": ("캔들", "(H - max(O,C)) / (H - L)", "윗꼬리가 짧을수록 장 막판 매물이 적다 (강의 IC +0.079)"),
    "ret_5d": ("추세", "C / C(t-5) - 1", "단기 모멘텀"),
    "ma20_gap": ("추세", "C / MA20(C) - 1", "20일선 위 여부"),
    "rsi_14": ("추세", "14일 RSI", "과열/과매도"),
    "mkt_night_20": ("시장", "KOSPI100 동일가중 밤수익의 20일 평균", "시장 전체 종배가 최근 통하고 있는가"),
}
REGIME_DOC = {
    "none": "시장 필터 없음 (매일 매수)",
    "ma120": "KOSPI100 동일가중 지수가 120일선 아래면 쉼",
    "night20": "시장 밤수익 20일 평균(mkt_night_20)이 0 이하면 쉼",
    "night10": "시장 밤수익 10일 평균이 0 이하면 쉼",
    "night40": "시장 밤수익 40일 평균이 0 이하면 쉼",
    "ma120&night20": "120일선 아래이거나 시장 밤수익 20일 평균이 0 이하면 쉼",
    "ma20&night20": "20일선 아래이거나 시장 밤수익 20일 평균이 0 이하면 쉼",
}


def best_nonml_cfg() -> dict:
    rows = json.loads((RESULTS_DIR / "week08_nonml_search.json").read_text(encoding="utf-8"))
    return {k: rows[0][k] for k in ("factors", "weights", "filter", "k", "regime", "sharpe")}


def nonml_mask_fn(cfg: dict):
    return lambda eng: eng.mask(w8.build_score(eng, cfg), cfg["k"], cfg["regime"])


def ml_mask_fn(bundle: dict):
    def fn(eng: w8.Engine) -> pd.DataFrame:
        scores = ml.predict_scores(bundle["model"], ml.feature_cube(eng), eng)
        sc = np.where(w8.filters(eng)[bundle["filter"]], scores, np.nan)
        return eng.mask(sc, bundle["k"], bundle["regime"])
    return fn


def lookahead_ok(panel: dict[str, pd.DataFrame], mask_fn, cut: pd.Timestamp) -> bool:
    """cut 일 이후 OHLCV 를 모두 망가뜨려도 cut 일까지 실제 적용 신호(mask.shift(1))가 같은지."""
    base = mask_fn(w8.Engine(panel)).shift(1, fill_value=False)
    rng = np.random.default_rng(0)
    broken = {f: df.copy() for f, df in panel.items()}
    future = broken["open"].index >= cut
    for df in broken.values():
        df.loc[future] = rng.uniform(1, 1e6, size=df.loc[future].shape)
    with np.errstate(all="ignore"):
        other = mask_fn(w8.Engine(broken)).shift(1, fill_value=False)
    return bool(base.loc[:cut].equals(other.loc[:cut]))


def _hyper(model) -> dict:
    keep = ("max_iter", "learning_rate", "max_depth", "min_samples_leaf", "l2_regularization", "n_estimators", "num_leaves",
            "min_child_samples", "max_features", "alpha", "hidden_layer_sizes")
    params = model.get_params()
    if "steps" in params:  # pipeline
        params = model.steps[-1][1].get_params()
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items() if k in keep}


def build() -> dict:
    panel = universe.load_panel(end=w8.END.date())
    eng = w8.Engine(panel)
    names = {s["symbol"]: s["name"] for s in universe.list_symbols()}
    rets = pf.returns(panel)
    night = rets["night"]
    cfg, bundle = best_nonml_cfg(), joblib.load(ml.MODEL_PATH)
    ml_search = json.loads((RESULTS_DIR / "week08_ml_search.json").read_text(encoding="utf-8"))

    masks = {k: m7.STRATEGIES[k][2](panel, eng.F) for k in SAMPLES}
    masks[NONML], masks[ML] = nonml_mask_fn(cfg)(eng), ml_mask_fn(bundle)(eng)
    days = eng.days
    runs = {k: m7.run_mask(m, night) for k, m in masks.items()}
    ew = pf.equal_weight(rets)["night"].reindex(days)

    half = days[len(days) // 2]
    first, second = days < half, days >= half

    def row(key, name, criteria, r, inv):
        r, inv = r.reindex(days).fillna(0.0), inv.reindex(days).fillna(False)
        net = [{"cost": c, **m7.score_metrics(r, inv, c)} for c in COSTS]
        main = next(n for n in net if abs(n["cost"] - w8.COST) < 1e-12)
        return {"key": key, "name": name, "criteria": criteria, "net": net, **main,
                "sharpe_first_half": m7.score_metrics(r[first], inv[first], w8.COST)["sharpe"],
                "sharpe_second_half": m7.score_metrics(r[second], inv[second], w8.COST)["sharpe"]}, r, inv

    strategies = {"KOSPI100_EW": {"name": "KOSPI100 동일가중 종배 (벤치마크)", "group": "base",
                                  "criteria": "100종목 전체 동일비중 매일 종배"}}
    for k in SAMPLES:
        strategies[k] = {"name": m7.STRATEGIES[k][0], "group": "sample", "criteria": m7.STRATEGIES[k][1]}
    strategies[NONML] = {"name": "[비ML 최고] 수급·야간관성 순위합", "group": "nonml", "criteria": w8.describe(cfg)}
    strategies[ML] = {"name": f"[ML 최고] {bundle['model_name'].upper()} 예측 상위 {bundle['k']}종목", "group": "ml",
                      "criteria": f"{bundle['model_name']} 모델이 예측한 '내일 밤 수익' 점수 상위 {bundle['k']}종목 · 필터 {bundle['filter']} · 레짐 {bundle['regime']}"}

    board, curves, invested = [], {}, {}
    b, r, inv = row("KOSPI100_EW", strategies["KOSPI100_EW"]["name"], strategies["KOSPI100_EW"]["criteria"], ew,
                    pd.Series(True, index=days))
    board.append(b)
    curves["KOSPI100_EW"] = r
    for k in strategies:
        if k == "KOSPI100_EW":
            continue
        b, r, inv = row(k, strategies[k]["name"], strategies[k]["criteria"], *runs[k][:2])
        board.append(b)
        curves[k], invested[k] = r, inv

    # 고속 엔진과 독립 파이프라인이 같은 수익률을 내는지
    fast = {NONML: w8.evaluate_cfg(eng, cfg)["r"],
            ML: ml.run_scores(eng, ml.predict_scores(bundle["model"], ml.feature_cube(eng), eng), bundle["k"],
                              bundle["filter"], bundle["regime"])["r"]}
    cross = {k: float(np.abs(fast[k] - curves[k].to_numpy()).max()) for k in fast}

    cuts = [panel["open"].index[int(len(panel["open"].index) * q)] for q in (0.3, 0.6, 0.9)]
    fns = {NONML: nonml_mask_fn(cfg), ML: ml_mask_fn(bundle)}
    look = {k: all(lookahead_ok(panel, fn, c) for c in cuts) for k, fn in fns.items()}

    explain = {NONML: explain_strategy(eng, masks[NONML], runs[NONML], names, cfg=cfg),
               ML: explain_strategy(eng, masks[ML], runs[ML], names, bundle=bundle)}
    ml_models = [{k: v for k, v in r_.items() if k in ("model", "target", "sharpe", "k", "filter", "regime", "walk_forward")}
                 for r_ in ml_search["results"]]
    nonml_top = json.loads((RESULTS_DIR / "week08_nonml_search.json").read_text(encoding="utf-8"))[:10]
    best = {m: max(board[1:], key=lambda b_: b_[m] if b_[m] is not None else -1e9)["key"]
            for m in ("cumulative_return", "calmar", "sharpe")}
    res = {"period": {"start": days[0].date().isoformat(), "end": days[-1].date().isoformat(), "days": len(days)},
           "warmup_days": int(eng.t0 + 1), "cost": w8.COST, "costs": COSTS, "half_cut": half.date().isoformat(),
           "strategies": strategies, "board": board, "best": best, "lookahead_check": look,
           "lookahead_cuts": [c.date().isoformat() for c in cuts], "engine_crosscheck_max_abs_diff": cross,
           "explain": explain, "factor_doc": {k: dict(zip(("group", "formula", "hypothesis"), v)) for k, v in FACTOR_DOC.items()},
           "regime_doc": REGIME_DOC, "nonml_cfg": cfg, "nonml_top10": nonml_top, "ml_models": ml_models,
           "ml": {"model": bundle["model_name"], "target": bundle["target"], "features": bundle["features"],
                  "hyper": _hyper(bundle["model"]), "k": bundle["k"], "filter": bundle["filter"], "regime": bundle["regime"],
                  "importance_top5": bundle["feature_importance_top5"], "train_range": bundle["train_range"],
                  "model_file": "backend/models/week08_best_model.joblib",
                  "walk_forward": ml_search["results"][0].get("walk_forward")}}
    rows = []
    for ts in days:
        rows.append({"ts": ts.date().isoformat(), **{k: float(v.at[ts]) for k, v in curves.items()},
                     **{f"inv_{k}": int(v.at[ts]) for k, v in invested.items()}})
    res["curve"] = rows
    return m7._clean(res)


def explain_strategy(eng: w8.Engine, mask: pd.DataFrame, run: tuple, names: dict, cfg: dict | None = None,
                     bundle: dict | None = None) -> dict:
    """'왜 이 수익률이 나왔나': 고른 종목 vs 같은 날 100종목, 자주 뽑힌 종목, 마지막 신호일 종목과 팩터 값."""
    r, inv, sig = run
    days = eng.days
    nxt = pd.DataFrame(eng.nxt, index=eng.index, columns=eng.cols)
    sigdays = eng.index[eng.t0:eng.t1 + 1]  # 신호가 만들어지는 날 (수익은 다음 거래일 밤)
    picked_mean = nxt.where(mask).mean(axis=1).loc[sigdays]  # t 일에 고른 종목의 t+1 밤 평균
    uni_mean = nxt.mean(axis=1).loc[sigdays]
    sel = mask.loc[sigdays].any(axis=1)
    edge = (picked_mean - uni_mean)[sel].dropna()
    s = sig.reindex(days)
    freq = s.sum().sort_values(ascending=False).head(8)
    last = mask.index[-1]
    pk = list(mask.columns[mask.loc[last].to_numpy()])
    F = eng.F
    picks = [{"symbol": c, "name": names.get(c), "factors": {f: F[f].at[last, c] for f in EXPLAIN_COLS}} for c in pk]
    out = {"picked_next_night_mean": float(picked_mean[sel].mean()), "universe_next_night_mean": float(uni_mean[sel].mean()),
           "edge_per_day": float(edge.mean()), "selected_days": int(sel.sum()),
           "avg_factors_picked": {f: float(np.nanmean(F[f].where(mask).loc[sigdays].to_numpy())) for f in EXPLAIN_COLS},
           "avg_factors_universe": {f: float(np.nanmean(F[f].loc[sigdays].to_numpy())) for f in EXPLAIN_COLS},
           "frequent": [{"symbol": c, "name": names.get(c), "days": int(v)} for c, v in freq.items()],
           "last_signal_date": last.date().isoformat(), "last_picks": picks}
    if cfg:
        out["recipe"] = {"factors": [{"name": f, "weight": w, **dict(zip(("group", "formula", "hypothesis"), FACTOR_DOC.get(f.lstrip("-"), ("", "", ""))))}
                                     for f, w in zip(cfg["factors"], cfg["weights"])],
                         "filter": cfg["filter"], "k": cfg["k"], "regime": cfg["regime"], "regime_text": REGIME_DOC[cfg["regime"]]}
    return out


def format_report(res: dict) -> str:
    p = res["period"]
    L = [f"[WEEK 08] 비ML vs ML 샤프 챌린지  {p['start']} ~ {p['end']} ({p['days']}거래일, 워밍업 {res['warmup_days']}일) · 왕복 {res['cost']:.2%} · 분할 없음",
         f"{'전략':<34}{'샤프':>7}{'누적':>12}{'CAGR':>9}{'MDD':>8}{'칼마':>9}{'승률':>8}{'보유일':>7}{'샤프 앞/뒤':>13}"]
    for b in res["board"]:
        L.append(f"{b['name']:<34}{b['sharpe']:>7.2f}{b['cumulative_return']:>+12.0%}{b['cagr']:>+9.1%}{b['mdd']:>+8.1%}"
                 f"{b['calmar']:>9.0f}{b['win_rate']:>8.1%}{b['invested_days']:>7}{b['sharpe_first_half']:>7.2f}/{b['sharpe_second_half']:.2f}")
    L += ["", "비ML: " + res["strategies"][NONML]["criteria"], "ML  : " + res["strategies"][ML]["criteria"],
          "ML 중요 팩터 상위 5: " + ", ".join(f"{d['feature']} {d['importance']:.1%}" for d in res["ml"]["importance_top5"]),
          f"엔진 교차검증 최대 오차: {res['engine_crosscheck_max_abs_diff']} · 미래참조 점검: {res['lookahead_check']}"]
    return "\n".join(L)


def main() -> None:
    res = build()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "week08_leaderboard.json").write_text(json.dumps({**res, "generated_at": datetime.now().isoformat(timespec="seconds")},
                                                                    ensure_ascii=False), encoding="utf-8")
    text = format_report(res)
    (RESULTS_DIR / "week08_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
