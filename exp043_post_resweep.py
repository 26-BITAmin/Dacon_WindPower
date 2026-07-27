# ═══════════════════════════════════════════════════════
# exp-043: gspd·q60·w0.70 전용 후처리 재최적화
#
# 배경: exp-042에서 gspd·q60·w0.70이 챔피언 대비 ΔCV +0.0015 / Δhold +0.0015
#   (이중 게이트 + 두 지표 동반 상승 + w 평탄 능선). 그런데 그 평가는
#   **챔피언(c16·a50)의 후처리를 그대로** 적용했다.
#   c16·a50는 exp-027에서 JMA-base + standard DL(hub) 기준으로 최적화됐고,
#   gspd·q60 base의 특성이 다를 수 있다:
#     - gspd GBM: MAE 0.1209→0.1189, 편향 분포 변화
#     - q60 DL: signed error +0.0075 (상향 편향, eval 시간 기준)
#   → 후처리를 재최적화하면 ΔCV +0.0015 위에서 추가 이득 가능.
#
# 설계: 재학습 없음. exp-040·039 캐시에서 로드 후 채점만.
#   기준 = gspd·q60·w0.70·c16·a50 (= exp-042 최선, 챔피언 대비 +0.0015)
#   챔피언 = g132·hub·w0.70·c16·a50 (= exp-035 LB 최선)
#
# 스윕 전략 (OAT → 조합 확인):
#   A. clip_lo: {0.10, 0.12, 0.14, 0.16, 0.18, 0.20}
#   B. alpha:   {0.30, 0.40, 0.50, 0.70, 0.90}
#   C. snap_to: {0.92, 0.94, 0.96}
#   D. snap_thr: {0.25, 0.30, 0.35}
#   E. 편향맵:   champ_bias (132-feat 기준) vs ext_bias[gspd] (348-feat 기준)
#   Stage2: 축별 승자 조합 확인 (사후선택 방지: 사전등록 조합만)
#
# 판정: 기준 대비 ΔCV > 0 & Δhold > 0 (기준 자체가 챔피언 대비 +0.0015라,
#   챔피언 대비 총 +0.001 초과가 최종 목표)
#
# 실행: C:\miniconda3\python.exe -X utf8 exp043_post_resweep.py
#       C:\miniconda3\python.exe -X utf8 exp043_post_resweep.py --stage2
# ═══════════════════════════════════════════════════════
import argparse
import pickle
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, apply_rolling_smoothing, apply_bias_correction_v2
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import load_gbm_cache, score_from_tidy, FOLD_LIST
from exp039_dl_diverse import DL_PRED_CACHE, build_slices
from exp040_gbm_spd import GBM_CACHE as EXT_GBM_CACHE, BIAS_CACHE as EXT_BIAS_CACHE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

KEY = ["fold", "group", "forecast_kst_dtm"]

# 챔피언 후처리 파라미터 (c16·a50)
CHAMP_PP = dict(clip_lo=0.16, alpha=0.50, snap_thr=0.30, snap_to=0.94)

# OAT 스윕 정의: 각 축은 챔피언 후처리를 기준으로 하나씩만 바꿈
OAT_CLIP   = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]   # A
OAT_ALPHA  = [0.30, 0.40, 0.50, 0.70, 0.90]           # B
OAT_SNTO   = [0.92, 0.94, 0.96]                        # C
OAT_SNTHR  = [0.25, 0.30, 0.35]                        # D


def postprocess_flex(cf_seq, cap, ws_seq, pA_seq, group, bias_map,
                     clip_lo, alpha, snap_thr, snap_to):
    """파라미터화된 후처리."""
    base = apply_bias_correction_v2(cf_seq * cap, ws_seq, group, bias_map)
    base = apply_rolling_smoothing(pd.Series(base), 3, alpha).to_numpy().copy()
    base[pA_seq >= snap_thr] = np.maximum(base[pA_seq >= snap_thr], cap * snap_to)
    return np.clip(base, cap * clip_lo, cap)


def load_all():
    """캐시에서 g132·gspd GBM, q60·hub DL 예측 로드."""
    # GBM 기준선 (g132 + 챔피언 편향맵)
    champ, champ_bias = load_gbm_cache()
    cache = champ.rename(columns={"gbm_cf": "cf_g132"})

    # 확장 GBM (gspd, 348피처)
    ext = pd.read_parquet(EXT_GBM_CACHE)
    with open(EXT_BIAS_CACHE, "rb") as f:
        ext_bias = pickle.load(f)
    for c in [c for c in ext.columns if c.startswith("cf_")]:
        m = cache[KEY].merge(ext[KEY + [c]], on=KEY, how="left")
        if m[c].notna().all():
            cache[c] = m[c].to_numpy()
    have_gspd = "cf_gspd" in cache.columns

    # DL 예측 (hub = 기준선, q60 = 상향 편향 expectile)
    dl = pd.read_parquet(DL_PRED_CACHE)
    for c in ["dl_hub", "dl_q60"]:
        if c in dl.columns:
            m = cache[KEY].merge(dl[KEY + [c]], on=KEY, how="left")
            if m[c].notna().all():
                cache[c] = m[c].to_numpy()

    have_q60 = "dl_q60" in cache.columns
    have_hub = "dl_hub" in cache.columns

    print(f"캐시 로드: {cache.shape}")
    print(f"  GBM: g132 O, gspd {'O' if have_gspd else 'X'}")
    print(f"  DL:  hub {'O' if have_hub else 'X'}, q60 {'O' if have_q60 else 'X'}")
    return cache, champ_bias, ext_bias, have_gspd, have_hub, have_q60


def build_blend_col(cache, gbm_col, dl_col, w):
    """블렌드 CF 컬럼 반환."""
    gbm = cache[gbm_col].to_numpy()
    if dl_col is None or w >= 1.0:
        return gbm
    dl = cache[dl_col].to_numpy()
    return w * gbm + (1 - w) * dl


def evaluate_pp(cache, slices, bias_maps, blend_cf, cv_folds, hold, pp):
    """주어진 후처리 파라미터로 전 fold 채점."""
    parts = []
    for (fname, group), sub in slices:
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        idx = sub.index
        cf = blend_cf[idx]
        pred = postprocess_flex(cf, cap,
                                sub["ws"].to_numpy(),
                                sub["pA"].to_numpy(),
                                group, bias_maps[fname],
                                **pp)
        parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
    tidy = pd.concat(parts, ignore_index=True)
    sc_cv   = score_from_tidy(tidy[tidy["fold"] != hold], "pred")
    sc_hold = score_from_tidy(tidy[tidy["fold"] == hold], "pred")
    folds   = {fn: score_from_tidy(tidy[tidy["fold"] == fn], "pred")[0] for fn in cv_folds}
    return sc_cv, sc_hold, folds


def run(args):
    cache, champ_bias, ext_bias, have_gspd, have_hub, have_q60 = load_all()

    if not have_gspd or not have_q60:
        print("필수 캐시 없음: gspd GBM 또는 q60 DL 예측이 존재하지 않습니다.")
        print("  exp040_gbm_spd.py와 exp039_dl_diverse.py를 먼저 실행하세요.")
        return

    fold_names = [fn for fn, _ in FOLD_LIST]
    hold = HOLD_FOLD["name"]
    cv_folds = [fn for fn in fold_names if fn != hold]

    # 슬라이스 구성 (fold·group별 정렬된 서브셋)
    slices = build_slices(cache)

    # 블렌드 CF (gspd·q60·w0.70 — exp-042 최선)
    blend_gspd_q60 = build_blend_col(cache, "cf_gspd", "dl_q60", 0.70)
    # 기준선(챔피언): g132·hub·w0.70
    blend_g132_hub = build_blend_col(cache, "cf_g132", "dl_hub", 0.70)

    # ── 기준선 채점 ──
    pp_champ = CHAMP_PP.copy()
    sc_ref_cv, sc_ref_hold, folds_ref = evaluate_pp(
        cache, slices, champ_bias, blend_gspd_q60, cv_folds, hold, pp_champ)
    sc_champ_cv, sc_champ_hold, folds_champ = evaluate_pp(
        cache, slices, champ_bias, blend_g132_hub, cv_folds, hold, pp_champ)

    ref_cv   = sc_ref_cv[0]
    ref_hold = sc_ref_hold[0]
    champ_cv   = sc_champ_cv[0]
    champ_hold = sc_champ_hold[0]

    print(f"\n챔피언(g132·hub·w0.70·c16·a50):  CV {champ_cv:.4f} | HOLD {champ_hold:.4f}")
    print(f"기준(gspd·q60·w0.70·c16·a50):    CV {ref_cv:.4f} (Δchamp {ref_cv-champ_cv:+.4f}) | "
          f"HOLD {ref_hold:.4f} (Δchamp {ref_hold-champ_hold:+.4f})")
    print(f"  ※ exp-042 로그 gspd·q60·w0.70: ΔCV +0.0015 / Δhold +0.0015 와 비교 (동일 런 Δ)")

    # ── OAT 스윕 ──
    VARIANTS = {}

    # A. clip_lo
    for c in OAT_CLIP:
        pp = CHAMP_PP | {"clip_lo": c}
        key = f"cl_{int(c*100):02d}"
        VARIANTS[key] = (pp, champ_bias, blend_gspd_q60)

    # B. alpha
    for a in OAT_ALPHA:
        pp = CHAMP_PP | {"alpha": a}
        key = f"al_{int(a*100):02d}"
        VARIANTS[key] = (pp, champ_bias, blend_gspd_q60)

    # C. snap_to
    for st in OAT_SNTO:
        pp = CHAMP_PP | {"snap_to": st}
        key = f"sn_{int(st*100):02d}"
        VARIANTS[key] = (pp, champ_bias, blend_gspd_q60)

    # D. snap_thr
    for sh in OAT_SNTHR:
        pp = CHAMP_PP | {"snap_thr": sh}
        key = f"pt_{int(sh*100):02d}"
        VARIANTS[key] = (pp, champ_bias, blend_gspd_q60)

    # E. 편향맵 교체: gspd 전용 재추정 맵 (ext_bias['gspd'])
    if "gspd" in ext_bias:
        VARIANTS["bm_gspd"] = (CHAMP_PP.copy(), ext_bias["gspd"], blend_gspd_q60)

    # F. Stage2 사전등록 조합 (--stage2 플래그가 있을 때만, 또는 항상 포함)
    # Stage2 조합: OAT 탐색 범위에서 유망한 구간을 사전등록
    STAGE2 = {
        "s2_cl14_al40": CHAMP_PP | {"clip_lo": 0.14, "alpha": 0.40},
        "s2_cl14_al50": CHAMP_PP | {"clip_lo": 0.14, "alpha": 0.50},
        "s2_cl16_al40": CHAMP_PP | {"clip_lo": 0.16, "alpha": 0.40},
        "s2_cl18_al40": CHAMP_PP | {"clip_lo": 0.18, "alpha": 0.40},
        "s2_cl14_sn92": CHAMP_PP | {"clip_lo": 0.14, "snap_to": 0.92},
        "s2_cl16_sn92": CHAMP_PP | {"clip_lo": 0.16, "snap_to": 0.92},
        "s2_cl14_al40_sn92": CHAMP_PP | {"clip_lo": 0.14, "alpha": 0.40, "snap_to": 0.92},
        "s2_cl16_al40_sn92": CHAMP_PP | {"clip_lo": 0.16, "alpha": 0.40, "snap_to": 0.92},
    }
    for k, pp in STAGE2.items():
        VARIANTS[k] = (pp, champ_bias, blend_gspd_q60)

    # ── 채점 ──
    t0 = time.time()
    results = {}
    for name, (pp, bm, blend) in VARIANTS.items():
        sc_cv, sc_hold, folds = evaluate_pp(cache, slices, bm, blend, cv_folds, hold, pp)
        results[name] = dict(cv=sc_cv, hold=sc_hold, folds=folds, pp=pp)
    print(f"\n{len(VARIANTS)}개 변형 채점 완료 ({time.time()-t0:.1f}s)")

    # ── 리포트 ──
    print("\n" + "=" * 112)
    print(f"{'변형':<24}{'CV':>8}{'ΔCV_ref':>10}{'ΔCV_champ':>11}{'HOLD':>8}{'Δhold_ref':>11}{'Δhold_champ':>12}"
          f"{'1-NMAE':>9}{'FiCR':>8}  판정")
    print("=" * 112)

    rows = []
    for name, r in sorted(results.items(),
                          key=lambda kv: -(kv[1]["cv"][0] + kv[1]["hold"][0])):
        cv   = r["cv"][0]
        hold = r["hold"][0]
        dcv_ref    = cv - ref_cv
        dhold_ref  = hold - ref_hold
        dcv_champ  = cv - champ_cv
        dhold_champ = hold - champ_hold
        n1 = r["cv"][1]; f1 = r["cv"][2]
        fold_ds = [r["folds"][fn] - folds_ref[fn] for fn in cv_folds]
        mag_ref   = dcv_ref > 0.001 and dhold_ref > 0.001
        mag_champ = dcv_champ > 0.001 and dhold_champ > 0.001
        both_up   = dcv_ref > 0 and dhold_ref > 0
        fold_cons = all(x > -0.0002 for x in fold_ds)
        verdict = ""
        if mag_champ:
            verdict = "★챔피언초과"
        elif mag_ref:
            verdict = "★기준초과"
        elif both_up:
            verdict = "기준↑"
        print(f"{name:<24}{cv:>8.4f}{dcv_ref:>+10.4f}{dcv_champ:>+11.4f}"
              f"{hold:>8.4f}{dhold_ref:>+11.4f}{dhold_champ:>+12.4f}"
              f"{n1:>9.4f}{f1:>8.4f}  {verdict}")
        rows.append(dict(name=name, cv=cv, hold=hold,
                        dcv_ref=dcv_ref, dhold_ref=dhold_ref,
                        dcv_champ=dcv_champ, dhold_champ=dhold_champ,
                        n1=n1, f1=f1, fold_ds=fold_ds,
                        both_up=both_up, fold_cons=fold_cons,
                        mag_ref=mag_ref, mag_champ=mag_champ))

    print("\n--- 기준선(gspd·q60·c16·a50) 리포트 ---")
    print(f"  CV:   {ref_cv:.4f} (1-NMAE {sc_ref_cv[1]:.4f}, FiCR {sc_ref_cv[2]:.4f})")
    print(f"  HOLD: {ref_hold:.4f} (1-NMAE {sc_ref_hold[1]:.4f}, FiCR {sc_ref_hold[2]:.4f})")

    # 상위 후보 fold 분해
    top = sorted([r for r in rows if r["both_up"]],
                 key=lambda r: -min(r["dcv_ref"], r["dhold_ref"]))[:6]
    if top:
        print(f"\n--- 기준 대비 양 게이트 통과 후보 (fold별 ΔCV) ---")
        print(f"{'변형':<24}" + "".join(f"{fn.replace('fold_',''):>10}" for fn in cv_folds)
              + "  fold일관  양지표")
        for r in top:
            print(f"{r['name']:<24}"
                  + "".join(f"{x:>+10.4f}" for x in r["fold_ds"])
                  + f"{'  O' if r['fold_cons'] else '  X':>9}"
                  + f"{'  O' if r['both_up'] else '  X':>8}")
    else:
        print("\n※ 기준 대비 양 게이트 통과 후보 없음.")
        best = sorted(rows, key=lambda r: -(r["cv"] + r["hold"]))[0]
        print(f"  최고 총합 변형: {best['name']} | CV {best['cv']:.4f} | HOLD {best['hold']:.4f}")
        print(f"  기준 대비: ΔCV {best['dcv_ref']:+.4f} / Δhold {best['dhold_ref']:+.4f}")

    # bias map 비교
    if "bm_gspd" in results:
        r = results["bm_gspd"]
        print(f"\n--- 편향맵 비교 ---")
        print(f"  champ_bias (기준): CV {ref_cv:.4f} | HOLD {ref_hold:.4f}")
        print(f"  gspd_bias:         CV {r['cv'][0]:.4f} | HOLD {r['hold'][0]:.4f}"
              f" (Δ: {r['cv'][0]-ref_cv:+.4f} / {r['hold'][0]-ref_hold:+.4f})")

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분 (채점만)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2", action="store_true", help="Stage2 조합만 평가")
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-043: gspd·q60·w0.70 후처리 재최적화 ###", flush=True)
    run(args)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
