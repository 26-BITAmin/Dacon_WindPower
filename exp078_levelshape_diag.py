# ═══════════════════════════════════════════════════════
# exp-078 진단: 챔피언 오차의 레벨-모양 분해 (일평균 수준 vs 일중 모양)
#   — "완전 새로운 파이프라인" 후보 선별용 (학습 없음, 캐시 재사용)
#
# 질문: 점수 손실(NMAE·FiCR 밴드 이탈)이 "그날 전체 수준을 틀려서"인가
#   "하루 안 시간 배분(모양)을 틀려서"인가?
#   - 레벨 지배 → 일 에너지 총량 전용 모델(시놉틱 피처) + 모양 분리 파이프라인 여지
#   - 모양 지배 → 레벨 분리 파이프라인 무의미 (현 시간별 조인트가 정당)
#
# 방법: 챔피언(hpo×dC_w65·PP_NEW) 후처리 예측을 make_ev 경로로 재현 →
#   (그룹, 일) 단위 err_day = 일평균 오차, err_shape = 시간 오차 − err_day.
#   (a) 평가시간 오차 분산 점유율, (b) 오라클 반사실: 레벨 오차 제거 시 /
#   모양 오차 제거 시 Score·FiCR·1-NMAE (완벽 교정 상한, exp-030/057 방식).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp078_levelshape_diag.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import FOLD_LIST, score_from_tidy
from exp039_dl_diverse import build_slices
from exp049_clean import PP_NEW, postprocess_flex
from exp060_pergroup_hpo import load_cache_full

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def tidy_champion(cache, champ_bias):
    blend = 0.65 * cache["cf_hpo"].to_numpy() + 0.35 * cache["dl_cln03"].to_numpy()
    parts = []
    for (fname, group), sub in build_slices(cache):
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        pred = postprocess_flex(blend[sub.index], cap, sub["ws"].to_numpy(),
                                sub["pA"].to_numpy(), group, champ_bias[fname], **PP_NEW)
        parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred, cap=cap))
    return pd.concat(parts, ignore_index=True)


def main():
    t0 = time.time()
    print("### exp-078 진단: 챔피언 오차 레벨-모양 분해 ###", flush=True)
    cache, champ_bias = load_cache_full()
    tidy = tidy_champion(cache, champ_bias)
    hold = HOLD_FOLD["name"]
    cv = tidy[(tidy["fold"] != hold) & tidy["target"].notna()].copy()
    base_cv = score_from_tidy(cv, "pred")
    print(f"챔피언 CV 재현: Score {base_cv[0]:.4f} (1-NMAE {base_cv[1]:.4f} FiCR {base_cv[2]:.4f})"
          f"  (기대 0.6556)", flush=True)

    cv["date"] = cv["forecast_kst_dtm"].dt.date
    cv["err"] = (cv["pred"] - cv["target"]) / cv["cap"]          # CF 단위 오차
    day = cv.groupby(["group", "date"])["err"].transform("mean")
    cv["err_day"], cv["err_shape"] = day, cv["err"] - day

    ev = cv["target"] >= 0.10 * cv["cap"]
    v_day = cv.loc[ev, "err_day"].var()
    v_shp = cv.loc[ev, "err_shape"].var()
    cov = 2 * np.cov(cv.loc[ev, "err_day"], cv.loc[ev, "err_shape"])[0, 1]
    v_tot = cv.loc[ev, "err"].var()
    print(f"\n(a) 평가시간 오차 분산 분해 (CF 단위, n={ev.sum()}):", flush=True)
    print(f"  전체 {v_tot:.5f} = 레벨 {v_day:.5f} ({v_day/v_tot:.1%}) + "
          f"모양 {v_shp:.5f} ({v_shp/v_tot:.1%}) + 공분산 {cov:.5f} ({cov/v_tot:.1%})", flush=True)

    # (b) 오라클 반사실 — 완벽 교정 상한
    for tag, newerr in (("레벨 오차 제거 (완벽 일평균)", cv["err_shape"]),
                        ("모양 오차 제거 (완벽 일중 모양)", cv["err_day"])):
        c2 = cv.copy()
        c2["pred"] = np.clip(c2["target"] + newerr * c2["cap"], 0, c2["cap"])
        s = score_from_tidy(c2, "pred")
        print(f"  (b) {tag}: Score {s[0]:.4f} (Δ{s[0]-base_cv[0]:+.4f}) | "
              f"1-NMAE {s[1]:.4f} (Δ{s[1]-base_cv[1]:+.4f}) | FiCR {s[2]:.4f} (Δ{s[2]-base_cv[2]:+.4f})",
              flush=True)

    # 밴드 이탈 귀속: 평가시간 중 error_rate>8% 행에서 |err_day| vs |err_shape| 우세
    e = cv[ev].copy()
    e["err_rate"] = (e["pred"] - e["target"]).abs() / e["target"]
    miss = e[e["err_rate"] > 0.08]
    dom_day = (miss["err_day"].abs() > miss["err_shape"].abs()).mean()
    print(f"\n(c) 0원 밴드(>8%) 이탈 {len(miss)}행 중 레벨 오차 우세 비율: {dom_day:.1%}", flush=True)
    # 그룹별
    for g in (1, 2, 3):
        mg = miss[miss["group"] == g]
        print(f"  그룹{g}: 이탈 {len(mg)}행, 레벨 우세 {(mg['err_day'].abs() > mg['err_shape'].abs()).mean():.1%}",
              flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
