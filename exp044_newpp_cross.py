# ═══════════════════════════════════════════════════════
# exp-044: 새 후처리(c16·al40·sn93·pt27) 아래 미탐색 캐시 축 교차 스윕
#
# 배경: exp-043이 gspd·q60·w0.70 **한 칸**에 대해서만 후처리를 재최적화해
#   CV 0.6481 / HOLD 0.6464 (챔피언 대비 +0.0033/+0.0030)를 얻었다.
#   그런데 exp-042의 (base × DL × w) 능선은 **구 후처리(c16·a50·sn94·pt30)** 기준이었다.
#   후처리가 바뀌면 능선 위치가 이동할 수 있으므로, 새 후처리 아래에서
#   재학습 없이 캐시로 채점 가능한 미탐색 축을 교차한다:
#     A. w 재스윕 (0.55~0.80) — 새 후처리에서 최적 w 이동 여부
#     B. DL 가중 혼합 — exp-039/042는 단순평균만 시도. q60 위주 + aw/hub 소량 혼합 미탐색
#     C. GBM 혼합 (g132⊕gspd) — 두 GBM 자체의 블렌드는 완전 미탐색
#     D. 그룹별 w OAT — 그룹마다 최적 블렌드 비율이 다를 가능성, 미탐색
#     E. 스무딩 윈도우 (3 고정이었음) — {1(off), 5} × al40
#     F. 사전등록 조합: A~C 블록 승자(양 게이트 통과분만)의 pairwise/삼중 조합
#
# 기준: ref = gspd·q60·w0.70·c16·al40·sn93·pt27 (exp-043 최선)
#       champ = g132·hub·w0.70·c16·a50·sn94·pt30 (exp-035 LB 챔피언)
#   재현 기대값(3-seed 캐시): ref CV 0.6481 / HOLD 0.6464, champ CV 0.6448 / HOLD 0.6434
#
# 판정(다중비교 방어, exp-042와 동일 3중 조건을 ref 대비로):
#   (a) ΔCV_ref > +0.001 AND Δhold_ref > +0.001
#   (b) fold 부호 일관 (음수 fold 없음, 완화 −0.0002)
#   (c) 두 지표(1-NMAE·FiCR) 동반 상승
#
# 실행: C:\miniconda3\python.exe -X utf8 exp044_newpp_cross.py
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

PP_CHAMP = dict(clip_lo=0.16, alpha=0.50, snap_thr=0.30, snap_to=0.94, win=3)
PP_NEW   = dict(clip_lo=0.16, alpha=0.40, snap_thr=0.27, snap_to=0.93, win=3)


def postprocess_flex(cf_seq, cap, ws_seq, pA_seq, group, bias_map,
                     clip_lo, alpha, snap_thr, snap_to, win=3):
    base = apply_bias_correction_v2(cf_seq * cap, ws_seq, group, bias_map)
    if win > 1:
        base = apply_rolling_smoothing(pd.Series(base), win, alpha).to_numpy().copy()
    base[pA_seq >= snap_thr] = np.maximum(base[pA_seq >= snap_thr], cap * snap_to)
    return np.clip(base, cap * clip_lo, cap)


def load_all():
    champ, champ_bias = load_gbm_cache()
    cache = champ.rename(columns={"gbm_cf": "cf_g132"})
    ext = pd.read_parquet(EXT_GBM_CACHE)
    with open(EXT_BIAS_CACHE, "rb") as f:
        ext_bias = pickle.load(f)
    for c in [c for c in ext.columns if c.startswith("cf_")]:
        m = cache[KEY].merge(ext[KEY + [c]], on=KEY, how="left")
        assert m[c].notna().all(), f"{c} 정렬 실패"
        cache[c] = m[c].to_numpy()
    dl = pd.read_parquet(DL_PRED_CACHE)
    for c in [c for c in dl.columns if c.startswith("dl_")]:
        m = cache[KEY].merge(dl[KEY + [c]], on=KEY, how="left")
        assert m[c].notna().all(), f"{c} 정렬 실패"
        cache[c] = m[c].to_numpy()
    return cache, champ_bias, ext_bias


def main():
    t0 = time.time()
    print("### exp-044: 새 후처리 아래 캐시 축 교차 스윕 (재학습 0) ###", flush=True)
    cache, champ_bias, ext_bias = load_all()
    print(f"캐시 결합: {cache.shape}", flush=True)

    fold_names = [fn for fn, _ in FOLD_LIST]
    hold = HOLD_FOLD["name"]
    cv_folds = [fn for fn in fold_names if fn != hold]
    slices = build_slices(cache)

    g132 = cache["cf_g132"].to_numpy()
    gspd = cache["cf_gspd"].to_numpy()
    q60  = cache["dl_q60"].to_numpy()
    aw   = cache["dl_aw"].to_numpy()
    hub  = cache["dl_hub"].to_numpy()
    groups_arr = cache["group"].to_numpy().astype(int)

    def blend(gbm_arr, dl_arr, w):
        """w: float(전역) 또는 {group:int → w} dict."""
        if isinstance(w, dict):
            wv = np.array([w[g] for g in groups_arr])
            return wv * gbm_arr + (1 - wv) * dl_arr
        return w * gbm_arr + (1 - w) * dl_arr

    def evaluate(blend_cf, pp):
        parts = []
        for (fname, group), sub in slices:
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            cf = blend_cf[sub.index]
            pred = postprocess_flex(cf, cap, sub["ws"].to_numpy(), sub["pA"].to_numpy(),
                                    group, champ_bias[fname], **pp)
            parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
        tidy = pd.concat(parts, ignore_index=True)
        return dict(cv=score_from_tidy(tidy[tidy["fold"] != hold], "pred"),
                    hold=score_from_tidy(tidy[tidy["fold"] == hold], "pred"),
                    folds={fn: score_from_tidy(tidy[tidy["fold"] == fn], "pred")[0] for fn in cv_folds})

    # ── 기준선 ──
    r_champ = evaluate(blend(g132, hub, 0.70), PP_CHAMP)
    r_ref   = evaluate(blend(gspd, q60, 0.70), PP_NEW)
    c_cv, c_hd = r_champ["cv"][0], r_champ["hold"][0]
    ref_cv, ref_hd = r_ref["cv"][0], r_ref["hold"][0]
    ref_n, ref_f = r_ref["cv"][1], r_ref["cv"][2]
    print(f"\n챔피언(g132·hub·w70·구PP): CV {c_cv:.4f} | HOLD {c_hd:.4f}  (기대 0.6448/0.6434)")
    print(f"ref(gspd·q60·w70·신PP):   CV {ref_cv:.4f} (1-NMAE {ref_n:.4f} FiCR {ref_f:.4f}) | "
          f"HOLD {ref_hd:.4f}  (기대 0.6481/0.6464)")

    # ── 변형 정의 ──
    V = {}  # name → (blend_cf, pp)

    # A. w 재스윕 (신PP)
    for w in [0.80, 0.75, 0.72, 0.68, 0.65, 0.60, 0.55]:
        V[f"A_w{int(w*100)}"] = (blend(gspd, q60, w), PP_NEW)

    # B. DL 가중 혼합 (w0.70, 신PP)
    DL_MIXES = {
        "B_q75a25": 0.75 * q60 + 0.25 * aw,
        "B_q50a50": 0.50 * q60 + 0.50 * aw,
        "B_q75h25": 0.75 * q60 + 0.25 * hub,
        "B_q50h50": 0.50 * q60 + 0.50 * hub,
        "B_qah":    (q60 + aw + hub) / 3.0,
    }
    for name, dl_arr in DL_MIXES.items():
        V[name] = (blend(gspd, dl_arr, 0.70), PP_NEW)

    # C. GBM 혼합 (q60·w0.70, 신PP)
    GBM_MIXES = {
        "C_gs75g25": 0.75 * gspd + 0.25 * g132,
        "C_gs50g50": 0.50 * gspd + 0.50 * g132,
    }
    for name, gbm_arr in GBM_MIXES.items():
        V[name] = (blend(gbm_arr, q60, 0.70), PP_NEW)

    # D. 그룹별 w OAT (기본 0.70, 한 그룹만 변경, 신PP)
    for g in [1, 2, 3]:
        for wg in [0.60, 0.80]:
            wmap = {1: 0.70, 2: 0.70, 3: 0.70}
            wmap[g] = wg
            V[f"D_g{g}w{int(wg*100)}"] = (blend(gspd, q60, wmap), PP_NEW)

    # E. 스무딩 윈도우 (신PP에서 win만 변경)
    V["E_win1"] = (blend(gspd, q60, 0.70), PP_NEW | {"win": 1})
    V["E_win5"] = (blend(gspd, q60, 0.70), PP_NEW | {"win": 5})

    # ── 채점 ──
    t1 = time.time()
    res = {name: evaluate(bc, pp) for name, (bc, pp) in V.items()}
    print(f"\n{len(V)}칸 채점 {time.time()-t1:.0f}s", flush=True)

    def report(res_dict, header):
        print("\n" + "=" * 110)
        print(header)
        print("=" * 110)
        print(f"{'변형':<14}{'CV':>8}{'ΔCVref':>9}{'HOLD':>8}{'Δhdref':>9}"
              f"{'1-NMAE':>9}{'Δn':>8}{'FiCR':>8}{'Δf':>8}{'ΔCVch':>9}{'Δhdch':>9}  판정")
        rows = []
        for name, r in sorted(res_dict.items(),
                              key=lambda kv: -min(kv[1]["cv"][0] - ref_cv, kv[1]["hold"][0] - ref_hd)):
            dcv, dhd = r["cv"][0] - ref_cv, r["hold"][0] - ref_hd
            dn, df = r["cv"][1] - ref_n, r["cv"][2] - ref_f
            fold_d = [r["folds"][fn] - r_ref["folds"][fn] for fn in cv_folds]
            mag = dcv > 0.001 and dhd > 0.001
            cons = all(x > -0.0002 for x in fold_d)
            both = dn > 0 and df > 0
            verdict = ("★3중충족" if (mag and cons and both) else
                       ("통과(매그니튜드)" if mag else ("양게이트" if (dcv > 0 and dhd > 0) else "")))
            print(f"{name:<14}{r['cv'][0]:>8.4f}{dcv:>+9.4f}{r['hold'][0]:>8.4f}{dhd:>+9.4f}"
                  f"{r['cv'][1]:>9.4f}{dn:>+8.4f}{r['cv'][2]:>8.4f}{df:>+8.4f}"
                  f"{r['cv'][0]-c_cv:>+9.4f}{r['hold'][0]-c_hd:>+9.4f}  {verdict}")
            rows.append(dict(name=name, dcv=dcv, dhd=dhd, dn=dn, df=df,
                             mag=mag, cons=cons, both=both, folds=fold_d))
        return rows

    rows = report(res, "블록 A~E (ref = gspd·q60·w0.70·신PP 대비)")

    # fold 분해 (양 게이트 통과분)
    top = sorted([r for r in rows if r["dcv"] > 0 and r["dhd"] > 0],
                 key=lambda r: -min(r["dcv"], r["dhd"]))[:8]
    if top:
        print(f"\n--- 양 게이트 통과 후보 fold별 ΔCV (ref 대비) ---")
        print(f"{'변형':<14}" + "".join(f"{fn.replace('fold_',''):>10}" for fn in cv_folds) + "  fold일관 두지표")
        for r in top:
            print(f"{r['name']:<14}" + "".join(f"{x:>+10.4f}" for x in r["folds"])
                  + f"{'  O' if r['cons'] else '  X':>8}{'  O' if r['both'] else '  X':>7}")

    # ── F. 사전등록 조합: 각 블록 승자(양 게이트 통과분만) 결합 ──
    def block_winner(prefix):
        cand = [r for r in rows if r["name"].startswith(prefix) and r["dcv"] > 0 and r["dhd"] > 0]
        return max(cand, key=lambda r: min(r["dcv"], r["dhd"]))["name"] if cand else None

    wA, wB, wC = block_winner("A_"), block_winner("B_"), block_winner("C_")
    print(f"\n블록 승자(양 게이트 통과분): A={wA} B={wB} C={wC}")
    combos = {}
    a_w = float(wA.split("w")[1]) / 100 if wA else 0.70
    b_dl = DL_MIXES[wB] if wB else q60
    c_gbm = GBM_MIXES[wC] if wC else gspd
    if wA and wB:
        combos[f"F_{wA[2:]}+{wB[2:]}"] = (blend(gspd, b_dl, a_w), PP_NEW)
    if wA and wC:
        combos[f"F_{wA[2:]}+{wC[2:]}"] = (blend(c_gbm, q60, a_w), PP_NEW)
    if wB and wC:
        combos[f"F_{wB[2:]}+{wC[2:]}"] = (blend(c_gbm, b_dl, 0.70), PP_NEW)
    if wA and wB and wC:
        combos[f"F_triple"] = (blend(c_gbm, b_dl, a_w), PP_NEW)
    if combos:
        res_f = {name: evaluate(bc, pp) for name, (bc, pp) in combos.items()}
        report(res_f, "블록 F: 사전등록 조합")

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
