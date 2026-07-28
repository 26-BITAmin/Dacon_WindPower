# ═══════════════════════════════════════════════════════
# exp-057 진단: FiCR 기대값 직접 최적화의 천장 측정 (경험 잔차 분포 + LOFO, 재학습 0)
#
# 공식 (metric() 확인): error_rate = |pred−actual| / **capacity**. er ≤ 0.06 → 크레딧 4 /
#   ≤ 0.08 → 3 / 초과 → 0, 실측 가중. NMAE도 cap 정규화. 평가시간 = actual ≥ 0.10cap.
#   → 밴드는 CF 공간 절대 폭(±0.06/±0.08). 최적 제출값 = 조건부 분포에서
#   "±0.06 창의 질량 × 실측가중 크레딧"과 "MAE 비용"의 균형점 (중앙값 아님).
#
# exp-005(실패)와의 차이: 분위수 모델 스프레드(미캘리브레이션) 대신 **후처리 후 OOF 잔차의
#   경험 분포**(구성상 캘리브레이션 자동)를 쓰고, bin별 시프트 δ를 LOFO로 선택(누수 방지,
#   exp-046 프로토콜). base도 0.625 시절이 아닌 현 챔피언(hpo 블렌드).
#
# 측정: pred-CF bin × 그룹별 δ*(기대 점수 최대) → LOFO 적용 ΔCV·Δhold, 천장(δ 무제약)과
#   mild(|δ|≤0.03) 두 버전. bin 진단표(잔차 폭·현 적중률·δ*) 동반.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp057_dist_diag.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import FOLD_LIST, score_from_tidy
from exp039_dl_diverse import build_slices
from exp049_clean import (GBM_CLEAN_CACHE, DL_CLEAN_CACHE, PP_NEW,
                          postprocess_flex, load_eval_frame)
from exp050_lambda_sweep import merge_cached_cols
from exp054_gate import GBM_EC_CACHE
from exp056_hpo import GBM_HPO_CACHE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

W_GBM = 0.65
BIN_EDGES = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01])
DELTAS = np.arange(-0.08, 0.0801, 0.004)


def build_tidy(cache, champ_bias):
    """챔피언 구성(hpo 블렌드 w65 + 신PP)의 후처리 후 fold 예측 tidy (CF 단위)"""
    slices = build_slices(cache)
    blend = W_GBM * cache["cf_hpo"].to_numpy() + (1 - W_GBM) * cache["dl_cln03"].to_numpy()
    parts = []
    for (fname, group), sub in slices:
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        pred = postprocess_flex(blend[sub.index], cap, sub["ws"].to_numpy(),
                                sub["pA"].to_numpy(), group, champ_bias[fname], **PP_NEW)
        parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]]
                     .assign(pred=pred, pred_cf=pred / cap, a_cf=sub["target"].to_numpy() / cap))
    return pd.concat(parts, ignore_index=True)


def pick_delta(pool_r, pool_a, n_g, s_g, cap_delta=None):
    """bin 잔차 풀에서 기대 점수 기여 최대 δ. r = a_cf − p_cf."""
    deltas = DELTAS if cap_delta is None else DELTAS[np.abs(DELTAS) <= cap_delta + 1e-9]
    best_d, best_v = 0.0, -np.inf
    for d in deltas:
        er = np.abs(d - pool_r)                      # |v−a| in CF
        up = np.select([er <= 0.06, er <= 0.08], [4.0, 3.0], default=0.0)
        val = -0.5 * er.mean() / n_g + 0.5 * (pool_a * up).mean() / (4.0 * s_g)
        if val > best_v:
            best_v, best_d = val, d
    return best_d


def apply_lofo(tidy, cv_folds, cap_delta=None, hold=None):
    """LOFO: 각 fold의 bin×그룹 δ를 나머지 CV fold 잔차로 선택. hold는 전 CV fold로 선택."""
    tidy = tidy.copy()
    tidy["r"] = tidy["a_cf"] - tidy["pred_cf"]
    tidy["bin"] = np.digitize(tidy["pred_cf"], BIN_EDGES) - 1
    ev = tidy["a_cf"] >= 0.10                        # 평가시간 (실측 기준)
    out_pred = tidy["pred"].to_numpy().copy()
    delta_log = {}
    for fname in cv_folds + ([hold] if hold else []):
        donor = tidy[ev & tidy["fold"].isin([f for f in cv_folds if f != fname])]
        sel_f = tidy["fold"] == fname
        for group in [1, 2, 3]:
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            dg = donor[donor["group"] == group]
            n_g = max(len(dg), 1)
            s_g = max(dg["a_cf"].sum() / n_g, 1e-6)  # 평균 실측 (스케일 상수)
            for b in range(len(BIN_EDGES) - 1):
                pool = dg[dg["bin"] == b]
                if len(pool) < 40:
                    continue
                d = pick_delta(pool["r"].to_numpy(), pool["a_cf"].to_numpy(), 1.0, s_g, cap_delta)
                delta_log[(fname, group, b)] = (d, len(pool))
                m = (sel_f & (tidy["group"] == group) & (tidy["bin"] == b)).to_numpy()
                out_pred[m] = np.clip(tidy.loc[m, "pred"].to_numpy() + d * cap, 0.16 * cap, cap)
    tidy["pred_adj"] = out_pred
    return tidy, delta_log


def main():
    t0 = time.time()
    print("### exp-057 진단: FiCR 직접 최적화 천장 (경험 잔차 + LOFO) ###", flush=True)
    cache, champ_bias = load_eval_frame()
    assert merge_cached_cols(cache, GBM_CLEAN_CACHE, ["cf_cln03"])
    assert merge_cached_cols(cache, DL_CLEAN_CACHE, ["dl_cln03"])
    assert merge_cached_cols(cache, GBM_EC_CACHE, ["cf_ec"])
    assert merge_cached_cols(cache, GBM_HPO_CACHE, ["cf_hpo"])

    fold_names = [fn for fn, _ in FOLD_LIST]
    hold = HOLD_FOLD["name"]
    cv_folds = [fn for fn in fold_names if fn != hold]

    tidy = build_tidy(cache, champ_bias)
    base_cv = score_from_tidy(tidy[tidy["fold"] != hold], "pred")
    base_hd = score_from_tidy(tidy[tidy["fold"] == hold], "pred")
    print(f"기준(hpo×dC_w65·신PP): CV {base_cv[0]:.4f} (1-NMAE {base_cv[1]:.4f} FiCR {base_cv[2]:.4f}) | "
          f"HOLD {base_hd[0]:.4f}  (기대 0.6556/0.6535)", flush=True)

    # ── bin 진단표 (CV fold 전체 잔차, 참고용) ──
    d = tidy[(tidy["fold"] != hold) & (tidy["a_cf"] >= 0.10)].copy()
    d["r"] = d["a_cf"] - d["pred_cf"]
    d["bin"] = np.digitize(d["pred_cf"], BIN_EDGES) - 1
    print("\n--- bin 진단 (CV 전체, 그룹 합산): n | 잔차 mean/std | P(≤6%cap) P(6~8) P(>8) ---")
    for b in range(len(BIN_EDGES) - 1):
        s = d[d["bin"] == b]
        if len(s) == 0:
            continue
        er = s["r"].abs()
        print(f"  [{BIN_EDGES[b]:.2f}~{BIN_EDGES[b+1]:.2f}) n={len(s):>5} | r {s['r'].mean():+.4f}/{s['r'].std():.4f} | "
              f"{(er <= .06).mean():.3f} {((er > .06) & (er <= .08)).mean():.3f} {(er > .08).mean():.3f}", flush=True)

    # ── LOFO 적용: 천장(무제약) vs mild(|δ|≤0.03) ──
    for tag, capd in [("무제약(천장)", None), ("mild |δ|≤0.03", 0.03), ("초mild |δ|≤0.016", 0.016)]:
        tt, dlog = apply_lofo(tidy, cv_folds, cap_delta=capd, hold=hold)
        cv = score_from_tidy(tt[tt["fold"] != hold], "pred_adj")
        hd = score_from_tidy(tt[tt["fold"] == hold], "pred_adj")
        folds_d = {fn: score_from_tidy(tt[tt["fold"] == fn], "pred_adj")[0] -
                   score_from_tidy(tidy[tidy["fold"] == fn], "pred")[0] for fn in cv_folds}
        print(f"\n[{tag}] ΔCV {cv[0]-base_cv[0]:+.4f} (Δn {cv[1]-base_cv[1]:+.4f} Δf {cv[2]-base_cv[2]:+.4f}) | "
              f"Δhold {hd[0]-base_hd[0]:+.4f}")
        print("  fold별 ΔCV: " + " ".join(f"{fn.replace('fold_','')}{dv:+.4f}" for fn, dv in folds_d.items()))
        # 대표 δ (Q1 fold 선택값)
        ds = [(g, b, v[0]) for (fn, g, b), v in dlog.items() if fn == cv_folds[0]]
        by_bin = {}
        for g, b, dd in ds:
            by_bin.setdefault(b, []).append(dd)
        print("  δ(bin 평균, Q1 선택): " + " ".join(
            f"[{BIN_EDGES[b]:.1f}]{np.mean(v):+.3f}" for b, v in sorted(by_bin.items())), flush=True)

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
