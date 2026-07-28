# ═══════════════════════════════════════════════════════
# exp-055b: 터빈합 보정 비율의 시간 드리프트 진단 + 폴드별 보정 재채점 (재학습 0)
#
# exp-055 관찰: 전 변형이 Q1/Q2 양수·Q3/Q4 음수·홀드 음수 — 학습 데이터가 많을수록
#   악화되는 역전 패턴. 후보 원인: 라벨/터빈합 보정을 전 기간 전역 상수로 추정 →
#   비율이 시간에 따라 드리프트하면 후반 폴드에서 체계 편향.
# 보정은 그룹별 스칼라라 cf_tb에서 나눠 되돌리고 폴드별(train_end 이전 데이터만) 비율을
#   다시 곱하면 정확히 재현 — 재학습 0, 리크도 제거(전역 비율은 검증 분기를 포함했었음).
# 추가: EC⊕TB 멤버 혼합 비율 {0.75, 0.5, 0.25} 스윕.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp055b_calib.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import build_inputs, FOLD_LIST, CACHE_DIR, score_from_tidy
from exp039_dl_diverse import build_slices
from exp049_clean import (GBM_CLEAN_CACHE, DL_CLEAN_CACHE, KEY, PP_NEW,
                          postprocess_flex, load_eval_frame)
from exp050_lambda_sweep import merge_cached_cols
from exp054_gate import GBM_EC_CACHE
from exp055_turbine import TB_HOURS_CACHE, GBM_TB_CACHE, build_turbine_hours

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

GLOBAL_CALIB = {1: 0.9863, 2: 0.9902, 3: 0.9997}   # exp-055 로그의 전역 비율 (되돌리기용)


def calib_windowed(tbh, lab, end=None, start=None):
    """train 윈도우 한정 라벨/터빈합 비율 (전 터빈 생존·완전 시간만)"""
    out = {}
    for g in [1, 2, 3]:
        sub = tbh[tbh["group"] == g]
        piv_e = sub.pivot_table(index="forecast_kst_dtm", columns="tb", values="energy")
        piv_d = sub.pivot_table(index="forecast_kst_dtm", columns="tb", values="dead")
        ok = piv_e.notna().all(axis=1) & (piv_d < 0.5).all(axis=1)
        ssum = piv_e.loc[ok].sum(axis=1)
        m = lab[lab["group"] == g].set_index("forecast_kst_dtm")["target"].to_frame().join(
            ssum.rename("ss"), how="inner").dropna()
        m = m[m["ss"] > 0]
        if end is not None:
            m = m[m.index <= end]
        if start is not None:
            m = m[m.index >= start]
        out[g] = float((m["ss"] * m["target"]).sum() / (m["ss"] ** 2).sum()) if len(m) else np.nan
    return out


def main():
    t0 = time.time()
    train_df, cols, colsets = build_inputs(["spd"])
    lab = train_df.loc[train_df["target"].notna(), ["forecast_kst_dtm", "kpx_group", "target"]] \
                  .rename(columns={"kpx_group": "group"})
    tbh = build_turbine_hours()
    tbh["forecast_kst_dtm"] = pd.to_datetime(tbh["forecast_kst_dtm"])

    # ── 1. 드리프트 진단: 반기별 비율 ──
    print("--- 보정 비율 드리프트 (반기별) ---")
    for s, e in [("2022-01-01", "2022-06-30"), ("2022-07-01", "2022-12-31"),
                 ("2023-01-01", "2023-06-30"), ("2023-07-01", "2023-12-31"),
                 ("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31")]:
        r = calib_windowed(tbh, lab, end=pd.Timestamp(e), start=pd.Timestamp(s))
        print(f"  {s[:7]}~{e[:7]}: " + " ".join(f"g{g}={r[g]:.4f}" if not np.isnan(r[g]) else f"g{g}=--"
                                                for g in [1, 2, 3]), flush=True)

    # ── 2. 폴드별 보정 재채점 ──
    cache, champ_bias = load_eval_frame()
    assert merge_cached_cols(cache, GBM_CLEAN_CACHE, ["cf_cln03"])
    assert merge_cached_cols(cache, DL_CLEAN_CACHE, ["dl_cln03"])
    assert merge_cached_cols(cache, GBM_EC_CACHE, ["cf_ec"])
    assert merge_cached_cols(cache, GBM_TB_CACHE, ["cf_tb"])

    # 전역 보정 제거 → 폴드별 보정 적용
    fold_ends = {fn: fold["train_end"] for fn, fold in FOLD_LIST}
    uncal = cache["cf_tb"].to_numpy() / cache["group"].map(GLOBAL_CALIB).to_numpy()
    cf_tbf = np.full(len(cache), np.nan)
    print("\n--- 폴드별 보정 비율 (train_end 이전만, 리크 제거) ---")
    for fn, end in fold_ends.items():
        r = calib_windowed(tbh, lab, end=pd.Timestamp(end))
        sel = (cache["fold"] == fn).to_numpy()
        cf_tbf[sel] = uncal[sel] * cache.loc[sel, "group"].map(r).to_numpy()
        print(f"  {fn}: " + " ".join(f"g{g}={r[g]:.4f}" for g in [1, 2, 3]), flush=True)
    cache["cf_tbf"] = cf_tbf

    # ── 3. 채점 ──
    fold_names = [fn for fn, _ in FOLD_LIST]
    hold = HOLD_FOLD["name"]
    cv_folds = [fn for fn in fold_names if fn != hold]
    slices = build_slices(cache)

    def ev(gbm_arr, dl_col, w, pp):
        blend = gbm_arr if (dl_col is None or w >= 1.0) else \
            w * gbm_arr + (1 - w) * cache[dl_col].to_numpy()
        parts = []
        for (fname, group), sub in slices:
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            pred = postprocess_flex(blend[sub.index], cap, sub["ws"].to_numpy(),
                                    sub["pA"].to_numpy(), group, champ_bias[fname], **pp)
            parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
        tidy = pd.concat(parts, ignore_index=True)
        return dict(cv=score_from_tidy(tidy[tidy["fold"] != hold], "pred"),
                    hold=score_from_tidy(tidy[tidy["fold"] == hold], "pred"),
                    folds={fn: score_from_tidy(tidy[tidy["fold"] == fn], "pred")[0] for fn in cv_folds})

    ec = cache["cf_ec"].to_numpy()
    tb_g = cache["cf_tb"].to_numpy()     # 전역 보정 (exp-055 재현)
    tb_f = cache["cf_tbf"].to_numpy()    # 폴드별 보정
    r_ec = ev(ec, "dl_cln03", 0.65, PP_NEW)
    print(f"\n보드챔피언(ec×dC_w65): CV {r_ec['cv'][0]:.4f} | HOLD {r_ec['hold'][0]:.4f}  (기대 0.6541/0.6529)")

    res = {}
    for tag, tb in [("g", tb_g), ("f", tb_f)]:
        for mr in (0.75, 0.5, 0.25):
            mix = mr * ec + (1 - mr) * tb
            for w in (0.70, 0.65):
                res[f"mix{int(mr*100)}{tag}_w{int(w*100)}"] = ev(mix, "dl_cln03", w, PP_NEW)
    res["tbf_w100"] = ev(tb_f, None, 1.0, PP_NEW)

    ch_cv, ch_hd = r_ec["cv"][0], r_ec["hold"][0]
    ch_n, ch_f2 = r_ec["cv"][1], r_ec["cv"][2]
    print("\n" + "=" * 112)
    print(f"{'변형':<14}{'CV':>8}{'ΔCVec':>9}{'HOLD':>8}{'Δhdec':>9}"
          f"{'1-NMAE':>9}{'Δn':>8}{'FiCR':>8}{'Δf':>8}  판정 (mixNN = EC비중, g=전역/f=폴드별 보정)")
    rows = []
    for name, r in sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - ch_cv, kv[1]["hold"][0] - ch_hd)):
        dcv, dhd = r["cv"][0] - ch_cv, r["hold"][0] - ch_hd
        dn_, df_ = r["cv"][1] - ch_n, r["cv"][2] - ch_f2
        fold_d = [r["folds"][fn] - r_ec["folds"][fn] for fn in cv_folds]
        mag = dcv > 0.001 and dhd > 0.001
        cons = all(x > -0.0002 for x in fold_d)
        both = dn_ > 0 and df_ > 0
        verdict = ("★3중충족" if (mag and cons and both) else
                   ("통과(매그니튜드)" if mag else ("양게이트" if (dcv > 0 and dhd > 0) else "")))
        print(f"{name:<14}{r['cv'][0]:>8.4f}{dcv:>+9.4f}{r['hold'][0]:>8.4f}{dhd:>+9.4f}"
              f"{r['cv'][1]:>9.4f}{dn_:>+8.4f}{r['cv'][2]:>8.4f}{df_:>+8.4f}  {verdict}")
        rows.append(dict(name=name, folds=fold_d, cons=cons))
    print(f"\n--- fold별 ΔCV (ec_w65 대비) ---")
    print(f"{'변형':<14}" + "".join(f"{fn.replace('fold_',''):>10}" for fn in cv_folds) + "  fold일관")
    for r in rows:
        print(f"{r['name']:<14}" + "".join(f"{x:>+10.4f}" for x in r["folds"])
              + f"{'  O' if r['cons'] else '  X':>8}")
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
