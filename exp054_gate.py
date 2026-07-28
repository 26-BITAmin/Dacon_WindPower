# ═══════════════════════════════════════════════════════
# exp-054 게이트: ECMWF IFS 피처(4개)를 champ49 GBM base에 투입
#
# 근거:
#   - 역대 LB 최대 상승 3건 전부 데이터 축 (exp-020 JMA +0.0044 / exp-035 +0.0051 /
#     exp-042 +0.0052). exp-048에서 API 경로 소진 후 남은 "기관 원본 아카이브" 첫 투입.
#   - ECMWF IFS = 현 소스(KMA LDAPS·NCEP GFS·JMA GSM/MSM)와 독립인 4번째 기관 모델.
#   - 커버리지 2023-01-18+ (2022 결측) — exp-020 JMA MSM(2023+)이 같은 부분 커버로
#     전이 성공한 직접 선례. 결측은 prepare_matrices의 median impute로 처리(동일 경로).
#   - 피처는 exp-020 MSM 블록 미러링 4개만: ec_ws, ec_wdsin, ec_wdcos, ec_dis.
#     (exp-021에서 JMA 시간문맥·추세 확장이 전부 탈락했으므로 최소 블록 유지)
#
# 투입 위치: GBM base만 (348+4=352, cln03 가중 유지). DL 무변경 —
#   exp-020 선례(JMA는 base 경유로 성공) + exp-040 교훈(같은 정보를 양 멤버에 주면 상쇄).
# 평가: champ49(cln×cln_w70·신PP) 대비 3중 조건. GBM 단독 델타(gEC_w100 vs gC_w100) 병기.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp054_gate.py --jobs 8 --seeds 3
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, DATA_DIR
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import build_inputs, FOLD_LIST, CACHE_DIR, score_from_tidy
from exp039_dl_diverse import build_slices
from exp049_clean import (GBM_CLEAN_CACHE, DL_CLEAN_CACHE, KEY, PP_CHAMP, PP_NEW,
                          merge_avail, postprocess_flex, load_eval_frame, train_gbm_fold_clean)
from exp050_lambda_sweep import merge_cached_cols, save_cols

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EC_WIDE = DATA_DIR / "external" / "ecmwf_prevruns_wide.parquet"
EC925_WIDE = DATA_DIR / "external" / "ecmwf_prevruns_925_wide.parquet"
GBM_EC_CACHE = CACHE_DIR / "exp054_gbm_ec.parquet"
EC_COLS = ["ec_ws", "ec_wdsin", "ec_wdcos", "ec_dis"]
EC9_COLS = ["ec9_ws", "ec9_wdsin", "ec9_wdcos", "ec9_shear"]


def build_ec_frame(path, ucol, vcol, prefix):
    """(init_date, step, group, u, v) → 시간별 KST 피처 (u/v 벡터 보간)"""
    w = pd.read_parquet(path)
    w["init_date"] = pd.to_datetime(w["init_date"])
    frames = []
    for (init_d, g), sub in w.groupby(["init_date", "group"]):
        sub = sub.sort_values("step")
        if len(sub) < 2:
            continue
        # valid KST = init 12:00 UTC + step + 9h. 타깃일 D = init + 2일, D 00:00 ~ D+1 00:00 커버
        anchor_h = sub["step"].to_numpy(dtype=float)                     # 27~51
        target_h = np.arange(28, 52, dtype=float)                        # D 01:00 ~ D+1 00:00 (24개)
        u = np.interp(target_h, anchor_h, sub[ucol].to_numpy())
        v = np.interp(target_h, anchor_h, sub[vcol].to_numpy())
        base_utc = init_d + pd.Timedelta(hours=12)
        kst = base_utc + pd.to_timedelta(target_h, unit="h") + pd.Timedelta(hours=9)
        ws = np.hypot(u, v)
        wd = np.arctan2(u, v)  # 벡터 방향 (사인·코사인만 쓰므로 기상학 관례 오프셋 무관)
        frames.append(pd.DataFrame({"forecast_kst_dtm": kst, "kpx_group": g,
                                    f"{prefix}_ws": ws, f"{prefix}_wdsin": np.sin(wd),
                                    f"{prefix}_wdcos": np.cos(wd)}))
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["forecast_kst_dtm", "kpx_group"], keep="last")
    return out


def evaluate(cache, champ_bias):
    fold_names = [fn for fn, _ in FOLD_LIST]
    hold = HOLD_FOLD["name"]
    cv_folds = [fn for fn in fold_names if fn != hold]
    slices = build_slices(cache)

    def ev(gbm_col, dl_col, w, pp):
        blend = cache[gbm_col].to_numpy() if (dl_col is None or w >= 1.0) else \
            w * cache[gbm_col].to_numpy() + (1 - w) * cache[dl_col].to_numpy()
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

    r_old = ev("cf_g132", "dl_hub", 0.70, PP_CHAMP)
    r_ref = ev("cf_gspd", "dl_q60", 0.70, PP_NEW)
    r_ch = ev("cf_cln03", "dl_cln03", 0.70, PP_NEW)
    print(f"\n구챔피언(g132·hub·w70·구PP): CV {r_old['cv'][0]:.4f} | HOLD {r_old['hold'][0]:.4f}  (기대 0.6448/0.6434)")
    print(f"ref(gspd·q60·w70·신PP):     CV {r_ref['cv'][0]:.4f} | HOLD {r_ref['hold'][0]:.4f}  (기대 0.6481/0.6463)")
    print(f"champ49(cln×cln_w70·신PP):  CV {r_ch['cv'][0]:.4f} (1-NMAE {r_ch['cv'][1]:.4f} FiCR {r_ch['cv'][2]:.4f}) | "
          f"HOLD {r_ch['hold'][0]:.4f}  (기대 0.6534/0.6518)")

    res = {"gC_w100": ev("cf_cln03", None, 1.0, PP_NEW),
           "gEC_w100": ev("cf_ec", None, 1.0, PP_NEW)}
    if "cf_ec8" in cache.columns:
        res["gEC8_w100"] = ev("cf_ec8", None, 1.0, PP_NEW)
    for w in (0.75, 0.70, 0.65, 0.60):
        res[f"gEC×dC_w{int(w*100)}"] = ev("cf_ec", "dl_cln03", w, PP_NEW)
        if "cf_ec8" in cache.columns:
            res[f"gEC8×dC_w{int(w*100)}"] = ev("cf_ec8", "dl_cln03", w, PP_NEW)

    ch_cv, ch_hd = r_ch["cv"][0], r_ch["hold"][0]
    ch_n, ch_f = r_ch["cv"][1], r_ch["cv"][2]
    print("\n" + "=" * 112)
    print(f"{'변형':<14}{'CV':>8}{'ΔCVch':>9}{'HOLD':>8}{'Δhdch':>9}"
          f"{'1-NMAE':>9}{'Δn':>8}{'FiCR':>8}{'Δf':>8}  판정 (Δ=champ49 대비)")
    rows = []
    for name, r in sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - ch_cv, kv[1]["hold"][0] - ch_hd)):
        dcv, dhd = r["cv"][0] - ch_cv, r["hold"][0] - ch_hd
        dn_, df_ = r["cv"][1] - ch_n, r["cv"][2] - ch_f
        fold_d = [r["folds"][fn] - r_ch["folds"][fn] for fn in cv_folds]
        mag = dcv > 0.001 and dhd > 0.001
        cons = all(x > -0.0002 for x in fold_d)
        both = dn_ > 0 and df_ > 0
        verdict = ("★3중충족" if (mag and cons and both) else
                   ("통과(매그니튜드)" if mag else ("양게이트" if (dcv > 0 and dhd > 0) else "")))
        print(f"{name:<14}{r['cv'][0]:>8.4f}{dcv:>+9.4f}{r['hold'][0]:>8.4f}{dhd:>+9.4f}"
              f"{r['cv'][1]:>9.4f}{dn_:>+8.4f}{r['cv'][2]:>8.4f}{df_:>+8.4f}  {verdict}")
        rows.append(dict(name=name, folds=fold_d, cons=cons, both=both))
    # GBM 단독 델타 (base 개선 크기 — exp-040의 (a) 기준)
    d_solo_cv = res["gEC_w100"]["cv"][0] - res["gC_w100"]["cv"][0]
    d_solo_hd = res["gEC_w100"]["hold"][0] - res["gC_w100"]["hold"][0]
    print(f"\nGBM 단독 델타 (gEC−gC): ΔCV {d_solo_cv:+.4f} / Δhold {d_solo_hd:+.4f}")
    print(f"\n--- fold별 ΔCV (champ49 대비) ---")
    print(f"{'변형':<14}" + "".join(f"{fn.replace('fold_',''):>10}" for fn in cv_folds) + "  fold일관 두지표")
    for name, r in res.items():
        fold_d = [r["folds"][fn] - r_ch["folds"][fn] for fn in cv_folds]
        cons = all(x > -0.0002 for x in fold_d)
        both = (r["cv"][1] - ch_n) > 0 and (r["cv"][2] - ch_f) > 0
        print(f"{name:<14}" + "".join(f"{x:>+10.4f}" for x in fold_d)
              + f"{'  O' if cons else '  X':>8}{'  O' if both else '  X':>7}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-054 게이트: ECMWF 피처 → cln GBM base ###", flush=True)

    train_df, cols, colsets = build_inputs(["spd"])
    vcols = colsets["spd"]
    train_df = merge_avail(train_df)

    ec = build_ec_frame(EC_WIDE, "u10", "v10", "ec")
    train_df = train_df.merge(ec, on=["forecast_kst_dtm", "kpx_group"], how="left")
    train_df["ec_dis"] = (train_df["ec_ws"] - train_df["ldaps_ws_10m"]).abs()
    has925 = EC925_WIDE.exists()
    if has925:
        ec9 = build_ec_frame(EC925_WIDE, "u925", "v925", "ec9")
        train_df = train_df.merge(ec9, on=["forecast_kst_dtm", "kpx_group"], how="left")
        train_df["ec9_shear"] = train_df["ec9_ws"] - train_df["ec_ws"]
    cov = train_df.groupby(train_df["forecast_kst_dtm"].dt.year)["ec_ws"].apply(lambda d: d.notna().mean().round(3))
    print("연도별 ECMWF 커버리지:\n" + cov.to_string(), flush=True)
    # 신호 확인: ws ↔ 발전량 corr (라벨 있는 행)
    for wcol in (["ec_ws", "ec9_ws"] if has925 else ["ec_ws"]):
        lab = train_df["target"].notna() & train_df[wcol].notna()
        for g in [1, 2, 3]:
            m = lab & (train_df["kpx_group"] == g)
            cap = CAPACITY_KWH[f"kpx_group_{g}"]
            c = np.corrcoef(train_df.loc[m, wcol], train_df.loc[m, "target"] / cap)[0, 1]
            cl = np.corrcoef(train_df.loc[m, wcol], train_df.loc[m, "ldaps_ws_10m"])[0, 1]
            print(f"  그룹{g}: corr({wcol}, CF) {c:.3f} | corr({wcol}, ldaps_ws) {cl:.3f}", flush=True)

    cache, champ_bias = load_eval_frame()
    assert merge_cached_cols(cache, GBM_CLEAN_CACHE, ["cf_cln03"]), "exp-049 GBM 캐시 필요"
    assert merge_cached_cols(cache, DL_CLEAN_CACHE, ["dl_cln03"]), "exp-049 DL 캐시 필요"

    want = {"cf_ec": vcols + EC_COLS}
    if has925:
        want["cf_ec8"] = vcols + EC_COLS + EC9_COLS
    have = merge_cached_cols(cache, GBM_EC_CACHE, list(want))
    for col, use_cols in want.items():
        if col in have:
            print(f"[{col}] 캐시 재사용", flush=True)
            continue
        preds = []
        for fname, fold in FOLD_LIST:
            tf = time.time()
            p = train_gbm_fold_clean(train_df, use_cols, fold, args.seeds, args.jobs, "cln03")
            p = p.rename(columns={"cf_cln03": col})
            preds.append(p)
            print(f"  [gbm:{col}:{fname}] {time.time()-tf:.0f}s", flush=True)
        p = pd.concat(preds, ignore_index=True)
        m = cache[KEY].merge(p, on=KEY, how="left")
        assert m[col].notna().all(), "정렬 실패"
        cache[col] = m[col].to_numpy()
        have.append(col)
        save_cols(cache, GBM_EC_CACHE, have)
        print(f"  캐시 저장 → {GBM_EC_CACHE.name}", flush=True)

    evaluate(cache, champ_bias)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
