"""Week 10 (5단계): 교수자 11주차 통합분석용 WEEK10_NN_REPORT.md 를 결과 JSON 에서 생성한다 (표의 숫자는 전부 JSON 에서 읽는다).

사용법 (컨테이너 안):  python week10_report.py      # -> /app/../WEEK10_NN_REPORT.md (프로젝트 루트는 마운트되지 않으므로 --out 으로 지정)
"""
import argparse
import json
from statistics import mean

import week10_db as w10db
import week10_nn as nn_

ORDER = ["rule_best", "ridge", "hgb", "mlp", "gru", "overfit_mlp"]
NAME = {"rule_best": "비ML 룰베이스 최고", "ridge": "Ridge (alpha=100)", "hgb": "HistGradientBoosting", "mlp": "MLP (2D)",
        "gru": "GRU (3D, L=20)", "overfit_mlp": "Overfit_MLP (대조군)"}
GEN = {"rule_best": "비ML", "ridge": "전통 ML", "hgb": "전통 ML", "mlp": "NN", "gru": "NN", "overfit_mlp": "NN(비규제)"}
KEYS = ["sharpe", "cumulative_return", "mdd", "win_rate"]


def pct(v, d=1):
    return f"{v * 100:+,.{d}f}%"


def load():
    return json.loads(nn_.RESULT_PATH.read_text(encoding="utf-8"))


def mean_metrics(ms):
    return {k: mean(m[k] for m in ms) for k in KEYS}


def three_way(res, model, seg, gate="none"):
    tw = res["three_way"]
    if model in tw["models"]:
        return mean_metrics([r["report"][gate][seg] for r in tw["models"][model]["runs"]])
    return tw["baselines"][model][gate][seg]


def walk(res, model, seg="overall", gate="none"):
    wf = res["walk_forward"]
    if model in wf["models"]:
        s = wf["models"][model]["seeds"]
        get = lambda x: x[gate]["overall"] if seg == "overall" else x[gate]["per_year"][seg]
        return mean_metrics([get(v) for v in s.values()])
    b = wf["baselines"][model][gate]
    return b["overall"] if seg == "overall" else b["per_year"][seg]


def seed_range(res, model, gate="none"):
    if model not in res["walk_forward"]["models"]:
        return "-"
    v = [s[gate]["overall"]["sharpe"] for s in res["walk_forward"]["models"][model]["seeds"].values()]
    return f"{min(v):.2f} ~ {max(v):.2f}"


def ablation_block() -> list[str]:
    """시장 공통 변수 재투입 대조 실험(week10_ablation.py) 결과. 파일이 없으면 '미검증' 문구."""
    path = nn_.RESULTS_DIR / "week10_ablation_mkt.json"
    if not path.exists():
        return ["- **해석(가설)**: 시장 공통 변수를 뺀 것이 날짜 암기를 막았을 가능성이 있으나 대조 실험을 하지 않아 인과는 검증되지 않았다.\n"]
    a = json.loads(path.read_text(encoding="utf-8"))
    off, on = a["baseline_no_mkt"], a["with_mkt"]
    mn = lambda rs, seg: mean(r["report"][seg]["sharpe"] for r in rs)
    per = lambda rs: " / ".join(f"{r['report']['test']['sharpe']:.2f}" for r in rs)
    ep = lambda rs, e, k: mean(r["epoch_sharpe"][e][k] for r in rs)
    B = []
    B.append("\n**대조 실험 — 시장 공통 변수(`mkt_*`)를 다시 넣으면 붕괴가 나타나는가?** (`backend/week10_ablation.py`, `results/week10_ablation_mkt.json`)\n")
    B.append(f"- 설계: Overfit_MLP(128×3, Dropout 0, wd 0, Early Stopping 없음, 60 에폭)를 위와 **완전히 같은 조건**(데이터·3단 분할·타깃·옵티마이저·시드 {a['seeds']})으로 두고 "
             f"**입력만 17→20**으로 바꿨다. 추가한 3개는 `{', '.join(a['mkt_features'])}`(같은 날 모든 종목이 같은 값, **원값**, Train 구간 평균·표준편차로만 표준화).\n")
    B.append("| 입력 | IS 샤프 | Val 샤프 | **OOS 샤프** | OOS 시드별 | 60에폭 Train 손실 |")
    B.append("|---|---:|---:|---:|---|---:|")
    for lab, rs in (("17 순위 피처 (기존, mkt 제외)", off), ("17 순위 피처 + mkt 3개 원값 (대조)", on)):
        B.append(f"| {lab} | {mn(rs, 'train'):.2f} | {mn(rs, 'val'):.2f} | **{mn(rs, 'test'):.2f}** | {per(rs)} | {mean(r['history']['train_loss'][-1] for r in rs):.4f} |")
    B.append("\n에폭별 시드평균 샤프 (IS / Val / OOS):\n")
    B.append("| 에폭 | mkt 제외 | mkt 추가 |")
    B.append("|---:|---|---|")
    for e in ("1", "8", "20", "45", "60"):
        f = lambda rs: " / ".join(f"{ep(rs, e, k):.2f}" for k in ("train", "val", "test"))
        B.append(f"| {e} | {f(off)} | {f(on)} |")
    gap_off, gap_on = mn(off, "train") - mn(off, "test"), mn(on, "train") - mn(on, "test")
    B.append(f"\n- **결과**: mkt 를 넣자 IS 는 에폭마다 오르고({ep(on, '1', 'train'):.2f}→{ep(on, '60', 'train'):.2f}) OOS 는 내려갔다({ep(on, '1', 'test'):.2f}→{ep(on, '60', 'test'):.2f}). "
             f"60에폭 OOS {mn(off, 'test'):.2f}→{mn(on, 'test'):.2f}(**{mn(on, 'test') - mn(off, 'test'):+.2f}**), IS−OOS 격차 {gap_off:+.2f}→{gap_on:+.2f}. "
             f"3개 시드 모두에서 OOS 가 내려갔고 두 조건의 시드 범위는 겹치지 않는다. Train 손실도 더 낮아졌다 — 날짜를 외우기 시작했다는 신호다. 반면 mkt 가 없으면 IS·OOS 모두 유지되거나 소폭 오른다.")
    B.append("- **해석**: 시장 공통 변수는 날짜를 식별하는 '바코드'로 작동해 비규제 신경망의 IS→OOS **X자 역전을 실제로 만들어 낸다**. 따라서 \"시장 공통 변수를 입력에서 뺀 것이 과적합 억제에 기여했다\"는 가설은 이 실험에서 **지지**된다.")
    B.append("- **한계(과대해석 금지)**: ① 역전 폭은 9주차 깊은 LightGBM(IS 20.5→OOS 2.3)만큼 극단적이지 않다 — 완전한 '붕괴'가 아니라 OOS 가 약 1.3 낮아지는 정도다(MLP 와 트리는 암기 방식·용량이 다르다). "
             "② 시드 3개·3단 분할 한 가지 구간에서만 확인했다. ③ **횡단면 순위 정규화의 단독 효과는 테스트하지 않았다**(원값 개별 팩터 대조 없음) — 두 처방 중 mkt 제거만 검증됐다. "
             "④ mkt 는 3개만, 원값으로 넣었다. 다른 표현(순위·더 많은 변수)에서의 크기는 모른다.\n")
    return B


def build(res: dict) -> str:
    tw, wf = res["three_way"], res["walk_forward"]
    rng = res["meta"]["ranges"]
    L = []
    w = L.append
    w("# Week 10 · 뉴럴 네트워크 개별 실습 리포트 (11주차 통합분석용)\n")
    w(f"> 생성: `backend/week10_report.py` (결과 파일: `backend/results/week10_nn_results.json`, 시드 {res['meta']['seeds']}) · 모든 수치는 왕복 비용 0.10% 차감 후, Top-5 동일비중, 현금일(0%) 포함\n")
    w("## 0. 한눈에 보기\n")
    best_wf = {g: max((m for m in ORDER if GEN[m].startswith(g)), key=lambda m: walk(res, m)["sharpe"]) for g in ("비ML", "전통 ML", "NN")}
    for g, m in best_wf.items():
        w(f"- **{g} 최고** (Walk-Forward OOS 샤프): {NAME[m]} **{walk(res, m)['sharpe']:.2f}**")
    w("- 신경망은 같은 입력으로 학습한 Ridge 를 **뚜렷하게 넘지 못했다** (GRU 는 동등, MLP 는 오히려 낮음). 비규제 대조군도 붕괴하지 않았다.")
    if (nn_.RESULTS_DIR / "week10_ablation_mkt.json").exists():
        w("- 대조 실험: 비규제 MLP 에 **시장 공통 변수를 다시 넣으면** IS 는 오르고 OOS 는 내려가는 X자 역전이 나타났다(3-①). 즉 \"시장 공통 변수 제거가 과적합 억제에 기여했다\"는 가설은 지지되며, 횡단면 순위 정규화의 단독 효과는 아직 미검증이다.\n")
    else:
        w("")

    w("## 1. 사용한 신경망 아키텍처 · 입력 · 규제\n")
    w("| 모델 | 구조 | 입력 텐서 | 총 파라미터 | 규제 |")
    w("|---|---|---|---:|---|")
    reg = {"mlp": "Dropout .3/.2 · LayerNorm · AdamW wd 1e-3 · Val 손실 Early Stopping(patience 5, 최적 에폭 복원)",
           "gru": "Dropout .3 · LayerNorm · AdamW wd 1e-3 · Val 손실 Early Stopping(patience 5, 최적 에폭 복원)",
           "overfit_mlp": "**없음** (Dropout 0 · weight decay 0 · Early Stopping 없음, 고정 60 에폭)"}
    for k, v in tw["models"].items():
        w(f"| {NAME[k]} | {v['arch']} | `{v['input']}` | {v['params']:,} | {reg[k]} |")
    t = res["meta"]["train"]
    w(f"\n- 옵티마이저 {t['optimizer']} (lr {t['lr']}) · 배치 {t['batch']} · 손실 **{t['loss']}** (타깃은 Train 구간 표준편차로만 스케일) · 시드 {res['meta']['seeds']} 고정\n")
    w("**입력·타깃·검증 규격**\n")
    w(f"- 입력 X: 종목별 팩터 **{len(res['meta']['features'])}개**({', '.join(res['meta']['features'])})를 **날짜별 100종목 횡단면 백분위 순위(-0.5~+0.5)**로 변환. 스케일러를 fit 하지 않으므로 미래 평균·분산 누수가 없다. **시장 공통 팩터(`mkt_*`)는 입력에서 제외**(레짐은 앞단 게이트로만 사용).")
    w("- 타깃 y: 시장 중립 초과수익률 `R_night(i,t+1) − mean_i R_night(t+1)` · 2D `(N, 17)` / 3D `(N, 20, 17)` (최근 20거래일 궤적, 마지막 행 = 당일 t).")
    w(f"- **방식 A 3단 분할**: Train {rng['train'][0]}~{rng['train'][1]} (IS) / Val {rng['val'][0]}~{rng['val'][1]} / OOS Test {rng['test'][0]}~{rng['test'][1]}.")
    w("- **방식 B 연도별 확장형 Walk-Forward**: 2022~2026 각 해를 그 해 이전 데이터로만 학습(마지막 6개월 = Val/Early Stopping)해 예측, 이어 붙인 OOS 전체 성과 산출.")
    w(f"- 레짐 게이트 `{res['meta']['headline_gate']}`(KOSPI100 동일가중 지수가 120일선 아래면 쉼)는 **사전에 고정**했고, 나머지 6종 게이트는 JSON 에만 참고용으로 있다.\n")

    w("## 2. 성과 비교표\n")
    w("### 2-A. 3단 분할 (IS = Train / Val / OOS = Test) — 게이트 없음\n")
    w("신경망은 시드 3개 평균. 샤프·누적수익률·MDD·승률(보유일 기준).\n")
    w("| 모델 | 세대 | IS 샤프 | IS 누적 | IS MDD | IS 승률 | Val 샤프 | **OOS 샤프** | OOS 누적 | OOS MDD | OOS 승률 |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in ORDER:
        a, v, o = (three_way(res, m, s) for s in ("train", "val", "test"))
        w(f"| {NAME[m]} | {GEN[m]} | {a['sharpe']:.2f} | {pct(a['cumulative_return'], 0)} | {pct(a['mdd'])} | {a['win_rate'] * 100:.1f}% | {v['sharpe']:.2f} | **{o['sharpe']:.2f}** | {pct(o['cumulative_return'], 0)} | {pct(o['mdd'])} | {o['win_rate'] * 100:.1f}% |")
    w("\n신경망 시드별 샤프 (IS / Val / OOS):\n")
    for k, v in tw["models"].items():
        w(f"- {NAME[k]}: " + " · ".join(f"seed{r['seed']} {r['report']['none']['train']['sharpe']:.2f}/{r['report']['none']['val']['sharpe']:.2f}/{r['report']['none']['test']['sharpe']:.2f}" for r in v["runs"]))
    w(f"\n레짐 게이트 `{res['meta']['headline_gate']}` 적용 시 OOS 샤프: " + " · ".join(f"{NAME[m]} {three_way(res, m, 'test', res['meta']['headline_gate'])['sharpe']:.2f}" for m in ORDER) + "\n")

    w("### 2-B. 연도별 Walk-Forward OOS (2022-01 ~ 2026-10, 1,172 거래일) — 게이트 없음\n")
    w("| 모델 | 세대 | 샤프 | 누적 | MDD | 승률 | 2022 | 2023 | 2024 | 2025 | 2026 | 시드별 샤프 범위 |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    ys = ["2022", "2023", "2024", "2025", "2026"]
    for m in ORDER:
        o = walk(res, m)
        w(f"| {NAME[m]} | {GEN[m]} | **{o['sharpe']:.2f}** | {pct(o['cumulative_return'], 0)} | {pct(o['mdd'])} | {o['win_rate'] * 100:.1f}% | " +
          " | ".join(f"{walk(res, m, y)['sharpe']:.2f}" for y in ys) + f" | {seed_range(res, m)} |")
    g = res["meta"]["headline_gate"]
    w(f"\n레짐 게이트 `{g}` 적용 시 Walk-Forward 샤프: " + " · ".join(f"{NAME[m]} {walk(res, m, 'overall', g)['sharpe']:.2f}" for m in ORDER) + "\n")

    # 관찰용 수치
    sd = lambda m: [s["none"]["overall"]["sharpe"] for s in wf["models"][m]["seeds"].values()]
    mlp, gru, ovf = walk(res, "mlp")["sharpe"], walk(res, "gru")["sharpe"], walk(res, "overfit_mlp")["sharpe"]
    ridge, hgb, rule = walk(res, "ridge")["sharpe"], walk(res, "hgb")["sharpe"], walk(res, "rule_best")["sharpe"]
    tmlp, tgru = three_way(res, "mlp", "test")["sharpe"], three_way(res, "gru", "test")["sharpe"]
    vmlp, vgru = three_way(res, "mlp", "val")["sharpe"], three_way(res, "gru", "val")["sharpe"]
    ovr = tw["models"]["overfit_mlp"]["runs"][0]
    h = ovr["history"]
    w("## 3. 실험 중 관찰한 핵심 현상\n")
    w("### ① IS→OOS 붕괴(X자 역전)가 재현되지 않았다 — 비규제 대조군도 OOS 가 유지됨\n")
    w(f"- Overfit_MLP(3.5만 파라미터, Dropout 0, weight decay 0, Early Stopping 없음)의 3단 분할 샤프는 IS {three_way(res, 'overfit_mlp', 'train')['sharpe']:.2f} / Val {three_way(res, 'overfit_mlp', 'val')['sharpe']:.2f} / **OOS {three_way(res, 'overfit_mlp', 'test')['sharpe']:.2f}**, Walk-Forward {ovf:.2f}. 9주차의 깊은 LightGBM(IS 20.5 → OOS 2.3)과 같은 붕괴가 없다.")
    w(f"- 에폭을 늘려도 Train 손실이 거의 내려가지 않는다 (시드0 Overfit_MLP: 1에폭 {h['train_loss'][0]:.4f} → 60에폭 {h['train_loss'][-1]:.4f}, Val {h['val_loss'][-1]:.4f}). 즉 이 입력으로는 **외울 거리 자체가 적다**.")
    L.extend(ablation_block())
    w("### ② 시드 간 편차\n")
    w(f"- Walk-Forward 샤프 시드별 범위: MLP {min(sd('mlp')):.2f}~{max(sd('mlp')):.2f} · GRU {min(sd('gru')):.2f}~{max(sd('gru')):.2f} · Overfit_MLP {min(sd('overfit_mlp')):.2f}~{max(sd('overfit_mlp')):.2f}. 시드 하나의 폭(약 0.2~0.4)이 GRU·Overfit_MLP·Ridge 사이의 평균 차이(0.1 안팎)보다 크다.")
    w("- 따라서 한 번의 학습 결과로 모델 우열을 말하면 안 된다. 시드 3개는 최소한이며, 더 엄밀하게는 10개 이상이 필요하다.\n")
    w("### ③ GRU(20일 시계열 궤적)의 효과 — 방향은 일관되나 확정할 수 없다\n")
    w(f"- 3단 분할 OOS: GRU {tgru:.2f} vs MLP {tmlp:.2f} (GRU +{tgru - tmlp:.2f}) · Walk-Forward: GRU {gru:.2f} vs MLP {mlp:.2f} (GRU +{gru - mlp:.2f}). 두 검증 모두 GRU 가 앞선다.")
    w(f"- 그러나 (a) 3단 분할 **Val 에서는 반대**(GRU {vgru:.2f} < MLP {vmlp:.2f}), (b) GRU 시드 범위가 MLP 와 일부 겹치고, (c) GRU(4,993)는 MLP(1,185)보다 파라미터가 4배라 '20일 궤적'과 '용량' 효과가 섞여 있으며, (d) Walk-Forward 에서 GRU({gru:.2f})는 Ridge({ridge:.2f})·비규제 MLP({ovf:.2f})와 사실상 같다.")
    w("- 결론: 최근 20일 흐름이 **약간의 추가 정보를 줄 가능성**은 있으나, 이 실험만으로 '시계열 궤적 효과'를 주장하기는 어렵다.\n")
    w("### ④ 전통 ML 대비 위치\n")
    w(f"- Walk-Forward 샤프: Ridge {ridge:.2f} ≈ GRU {gru:.2f} ≈ Overfit_MLP {ovf:.2f} > MLP {mlp:.2f} > 룰베이스 {rule:.2f} > HistGB {hgb:.2f}. 선형 모델이 비선형 신경망과 대등하다 — 이 팩터들의 신호가 대부분 **선형·단조**라는 뜻이다.")
    w("- HistGB 는 학습 데이터가 적은 초기 폴드(2022)에서 특히 약했다 (연도별 표 참고). 신경망·Ridge 는 같은 폴드에서 더 안정적이었다.\n")
    w("### ⑤ 기타\n")
    w(f"- 레짐 게이트 `{g}` 는 Walk-Forward 에서 HistGB 를 뺀 5개 모델의 샤프를 낮췄다 (위 2-B). 3단 분할에서는 MLP·GRU 의 Val/OOS 가 소폭 오르고 IS 는 내려갔지만 Overfit_MLP 는 모두 내려가, 일관된 이득이 아니다.")
    w("- Walk-Forward 연도별 샤프가 2022(약 2) → 2026(약 5)로 매년 오르는 것은 학습 데이터 증가 효과와 시장 환경 변화가 섞인 결과로, 이 실험만으로 구분되지 않는다.\n")

    w("## 4. 한계 · 주의\n")
    w("- 유니버스가 **현재** KOSPI 100 구성종목이라 생존편향이 있다. 슬리피지·거래세는 반영하지 않았다. 절대 샤프(4~5)는 이 조건의 산물이므로 **모델 간 상대 비교**로만 읽는다.")
    w("- 비ML 룰베이스는 8주차에 2020~2026 **전 구간을 보고 고른** 설정이라 IS/OOS 구분이 의미 없다(사후 최적화). 표의 값은 같은 점수를 게이트 없이 적용한 것이다.")
    w("- 누적곡선(대시보드)의 신경망은 시드 0 한 개라 표의 시드평균 샤프와 약간 다르다.")
    w("- 체크포인트(`backend/models/week10_best_nn.pt`)는 테스트가 아닌 **Val 샤프**로 골랐다 (선택 편향 방지).\n")

    w("## 5. 재현 · 파일\n")
    w("```bash\n# 컨테이너(kis-backend) 안에서\npython week10_nn.py --seeds 3        # 학습·평가 -> results/week10_nn_results.json, models/week10_best_nn.pt (약 33분, CPU)\npython week10_nn.py --curves         # 시드0 일별 수익률(누적곡선용) -> results/week10_curves.json\npython week10_db.py                  # DB(w10_*) 저장\npython week10_ablation.py            # 시장 공통 변수 재투입 대조 실험 (약 2분) -> results/week10_ablation_mkt.json\npython -m pytest tests/test_week10_dataset.py\n```\n")
    w("| 파일 | 설명 |\n|---|---|\n| `backend/week10_dataset.py` | 누수 없는 2D/3D 데이터셋 · 3단 분할 · Walk-Forward 구간 · 백테스트 헬퍼 |\n| `backend/week10_nn.py` | MLP/GRU/Overfit_MLP 학습, 기준선(룰베이스·Ridge·HistGB), 평가 |\n| `backend/week10_ablation.py` | 대조 실험: 비규제 MLP 에 시장 공통 변수(`mkt_*`) 재투입 |\n| `backend/week10_db.py` | DB 저장(`w10_models/metrics/history/daily`) · API 페이로드 |\n| `backend/main.py` | `GET /backtest/week10` |\n| `frontend/backtest.html` | WEEK 10 탭(`#m10`) |\n| `backend/results/week10_dashboard.png` | 대시보드 캡처 |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="WEEK10_NN_REPORT.md")
    a = ap.parse_args()
    # week10_db.HEADLINE 과 결과 파일의 게이트가 같은지 확인
    res = load()
    assert res["meta"]["headline_gate"] == w10db.HEADLINE
    text = build(res)
    open(a.out, "w", encoding="utf-8").write(text)
    print(f"저장: {a.out} ({len(text.splitlines())}줄)")
