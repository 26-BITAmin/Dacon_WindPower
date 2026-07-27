# ═══════════════════════════════════════════════════════
# exp-046: 예측 CF 구간 조건부 블렌드 가중 (LOFO 적합, 재학습 0)
#
# 배경: exp-046 사전진단(exp046_diag_regime.py)에서
#   - 관측 축(풍속·시간대·pA)별 최적 w는 0.70 부근 평탄 (여지 없음)
#   - **실측 CF 구간별로는 큰 비대칭**: CF 0.7-0.9 → w*=0.30 (MAE −0.0041),
#     0.9+ → w*=0.00, 0.1-0.3 → w*=1.00.
#     CF 0.7-0.9는 exp-030의 "71% 과소예측 = FiCR 최대 손실 지대"와 일치 —
#     상향 편향 q60 DL이 그 구간에서 구조적으로 우위.
#   실측 CF는 관측 불가지만 **예측 CF(블렌드 자체)는 관측 가능** → 예측 CF 구간별
#   조건부 w로 근사 시도. 전역 상수 w(exp-036·042·044)와 다른 미탐색 축.
#
# 설계 (사전등록 단일 스킴 — 사후 변형 금지):
#   zone: blend70(=0.70·gspd+0.30·q60) 기준 [<0.5 / 0.5~0.7 / ≥0.7]
#   w_row = w_lo (zone1) / 0.70 (zone2) / w_hi (zone3)
#   그리드: w_lo ∈ {0.70, 0.80, 0.90, 1.00} × w_hi ∈ {0.30, 0.40, 0.50, 0.60, 0.70}
#   **LOFO**: CV fold f의 (w_lo,w_hi)는 나머지 3개 CV fold pooled 점수로 선택.
#             홀드아웃의 (w_lo,w_hi)는 4개 CV fold pooled로 선택.
#   → 선택이 평가 fold를 절대 보지 않음 (다중비교 무해화)
#   판정: ref(gspd·q60·w0.70·신PP) 대비 3중 조건.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp046_condw.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, apply_rolling_smoothing, apply_bias_correction_v2
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import load_gbm_cache, score_from_tidy, FOLD_LIST
from exp039_dl_diverse import DL_PRED_CACHE as E39_CACHE, build_slices
from exp040_gbm_spd import GBM_CACHE as EXT_GBM_CACHE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

KEY = ["fold", "group", "forecast_kst_dtm"]
PP_CHAMP = dict(clip_lo=0.16, alpha=0.50, snap_thr=0.30, snap_to=0.94)
PP_NEW   = dict(clip_lo=0.16, alpha=0.40, snap_thr=0.27, snap_to=0.93)
Z1, Z2 = 0.50, 0.70
W_LO = [0.70, 0.80, 0.90, 1.00]
W_HI = [0.30, 0.40, 0.50, 0.60, 0.70]


def postprocess_flex(cf_seq, cap, ws_seq, pA_seq, group, bias_map,
                     clip_lo, alpha, snap_thr, snap_to):
    base = apply_bias_correction_v2(cf_seq * cap, ws_seq, group, bias_map)
    base = apply_rolling_smoothing(pd.Series(base), 3, alpha).to_numpy().copy()
    base[pA_seq >= snap_thr] = np.maximum(base[pA_seq >= snap_thr], cap * snap_to)
    return np.clip(base, cap * clip_lo, cap)


def main():
    t0 = time.time()
    print("### exp-046: 예측 CF 구간 조건부 w (LOFO, 재학습 0) ###", flush=True)
    cache, champ_bias = load_gbm_cache()
    cache = cache.rename(columns={"gbm_cf": "cf_g132"})
    ext = pd.read_parquet(EXT_GBM_CACHE)
    m = cache[KEY].merge(ext[KEY + ["cf_gspd"]], on=KEY, how="left")
    assert m["cf_gspd"].notna().all()
    cache["cf_gspd"] = m["cf_gspd"].to_numpy()
    dl = pd.read_parquet(E39_CACHE)
    for c in ["dl_q60", "dl_hub"]:
        mm = cache[KEY].merge(dl[KEY + [c]], on=KEY, how="left")
        assert mm[c].notna().all()
        cache[c] = mm[c].to_numpy()

    fold_names = [fn for fn, _ in FOLD_LIST]
    hold = HOLD_FOLD["name"]
    cv_folds = [fn for fn in fold_names if fn != hold]
    slices = build_slices(cache)

    gspd = cache["cf_gspd"].to_numpy()
    q60 = cache["dl_q60"].to_numpy()
    blend70 = 0.70 * gspd + 0.30 * q60
    zone = np.where(blend70 < Z1, 0, np.where(blend70 < Z2, 1, 2))
    n_z = [int((zone == z).sum()) for z in range(3)]
    print(f"zone 분포 (전체 행): <0.5 {n_z[0]} / 0.5-0.7 {n_z[1]} / ≥0.7 {n_z[2]}")

    def cond_blend(w_lo, w_hi):
        w_row = np.where(zone == 0, w_lo, np.where(zone == 1, 0.70, w_hi))
        return w_row * gspd + (1 - w_row) * q60

    def tidy_for(blend_cf, pp):
        parts = []
        for (fname, group), sub in slices:
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            pred = postprocess_flex(blend_cf[sub.index], cap, sub["ws"].to_numpy(),
                                    sub["pA"].to_numpy(), group, champ_bias[fname], **pp)
            parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
        return pd.concat(parts, ignore_index=True)

    # 후보별 tidy 캐시 (fold별 점수는 부분집합 채점으로)
    cands = [(wl, wh) for wl in W_LO for wh in W_HI]
    tidies = {}
    for wl, wh in cands:
        tidies[(wl, wh)] = tidy_for(cond_blend(wl, wh), PP_NEW)
    print(f"{len(cands)}개 후보 tidy 생성 {time.time()-t0:.0f}s", flush=True)

    def pooled_score(tidy, folds):
        return score_from_tidy(tidy[tidy["fold"].isin(folds)], "pred")

    # ── LOFO 선택 ──
    chosen = {}
    for f in cv_folds:
        fit_folds = [x for x in cv_folds if x != f]
        best = max(cands, key=lambda c: pooled_score(tidies[c], fit_folds)[0])
        chosen[f] = best
    chosen[hold] = max(cands, key=lambda c: pooled_score(tidies[c], cv_folds)[0])
    print("\nLOFO 선택 결과 (fold → (w_lo, w_hi)):")
    for f, c in chosen.items():
        print(f"  {f}: w_lo={c[0]:.2f}, w_hi={c[1]:.2f}")

    # ── LOFO 예측 조립 ──
    parts = []
    for f in fold_names:
        t = tidies[chosen[f]]
        parts.append(t[t["fold"] == f])
    lofo = pd.concat(parts, ignore_index=True)

    # ── 기준선 ──
    ref_tidy = tidy_for(0.70 * gspd + 0.30 * q60, PP_NEW)
    champ_tidy = tidy_for(0.70 * cache["cf_g132"].to_numpy() + 0.30 * cache["dl_hub"].to_numpy(), PP_CHAMP)
    ref_cv = pooled_score(ref_tidy, cv_folds); ref_hd = pooled_score(ref_tidy, [hold])
    ch_cv = pooled_score(champ_tidy, cv_folds); ch_hd = pooled_score(champ_tidy, [hold])
    lo_cv = pooled_score(lofo, cv_folds); lo_hd = pooled_score(lofo, [hold])

    print(f"\n챔피언(구PP): CV {ch_cv[0]:.4f} | HOLD {ch_hd[0]:.4f}  (기대 0.6448/0.6434)")
    print(f"ref(w0.70 상수·신PP): CV {ref_cv[0]:.4f} (1-NMAE {ref_cv[1]:.4f} FiCR {ref_cv[2]:.4f}) | HOLD {ref_hd[0]:.4f}")
    print(f"LOFO 조건부 w:        CV {lo_cv[0]:.4f} (1-NMAE {lo_cv[1]:.4f} FiCR {lo_cv[2]:.4f}) | HOLD {lo_hd[0]:.4f}")
    dcv, dhd = lo_cv[0] - ref_cv[0], lo_hd[0] - ref_hd[0]
    dn, df_ = lo_cv[1] - ref_cv[1], lo_cv[2] - ref_cv[2]
    print(f"Δref: CV {dcv:+.4f} / HOLD {dhd:+.4f} | 1-NMAE {dn:+.4f} / FiCR {df_:+.4f}")

    print("\n--- fold별 (LOFO − ref) ---")
    for f in cv_folds:
        s_l = score_from_tidy(lofo[lofo["fold"] == f], "pred")[0]
        s_r = score_from_tidy(ref_tidy[ref_tidy["fold"] == f], "pred")[0]
        print(f"  {f}: {s_l - s_r:+.4f}  (선택: {chosen[f]})")

    # 참고: 오라클(전 fold에서 사후 최적 단일 후보) — 상한 가늠용, 판정에는 미사용
    best_post = max(cands, key=lambda c: pooled_score(tidies[c], cv_folds)[0])
    bp_cv = pooled_score(tidies[best_post], cv_folds)
    bp_hd = pooled_score(tidies[best_post], [hold])
    print(f"\n[참고] 사후 최적 단일 (w_lo,w_hi)={best_post}: CV {bp_cv[0]:.4f} ({bp_cv[0]-ref_cv[0]:+.4f}) | "
          f"HOLD {bp_hd[0]:.4f} ({bp_hd[0]-ref_hd[0]:+.4f})")

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
