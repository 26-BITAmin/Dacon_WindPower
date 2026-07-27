# ═══════════════════════════════════════════════════════
# exp-045: q60 비대칭 계수(τ) 스윕 — 현 최선 후보의 유일한 미스윕 하이퍼파라미터
#
# 배경: 현 세션 최선 후보(exp-043: gspd·q60·w0.70·c16·al40·sn93·pt27, ΔCV +0.0033/
#   Δhold +0.0030)의 핵심 성분 q60은 exp-039에서 τ=0.60(과소 1.2×/과대 0.8×) **단일
#   추측값**으로 도입됐고 스윕된 적이 없다. exp-044에서 캐시 축(w·DL혼합·GBM혼합·
#   그룹별w·윈도우)은 전부 소진 확인 → 남은 인접 축은 q60 자체의 τ.
#   exp-039 관찰: aw(signed +0.0210)는 실패 프로파일, q60(signed +0.0075)은 성공
#   프로파일 → τ↑는 signed 상향(aw 방향), τ↓는 hub 방향. 0.60이 능선인지 확인 필요.
#
# 변형: τ ∈ {0.55, 0.65, 0.70} (q55/q65/q70). 입력 spd 348·MTMLP 384×3 ep120·
#   3-seed 전부 exp-039와 동일 — 손실의 τ만 변경. q60은 exp-039 캐시 재사용(재학습 0).
#
# 평가: gspd base(exp-040 캐시) + 새 후처리(c16·al40·sn93·pt27) + w{0.75,0.70,0.65}.
#   ref = gspd·q60·w0.70·신PP (동일 런 재현 기대 CV 0.6481 / HOLD 0.6463).
#   인접 τ 혼합 {q55+q60, q60+q65, q60+q70} (평균, w0.70)도 무비용 채점.
# 판정: 3중 조건 (양 게이트 ΔCV·Δhold > +0.001 / fold 일관 / 두 지표 동반 상승).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp045_tau_sweep.py
# ═══════════════════════════════════════════════════════
import argparse
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from exp005_qopt import CAPACITY_KWH, apply_rolling_smoothing, apply_bias_correction_v2
from exp019_multigrid import HOLD_FOLD
from exp033_dl import predict, GIDX
from exp035_raw_dl import (build_inputs, load_gbm_cache, score_from_tidy,
                           FOLD_LIST, DL_CFG, CACHE_DIR)
from exp039_dl_diverse import (_huber, train_one, DL_PRED_CACHE as E39_CACHE,
                               build_slices)
from exp040_gbm_spd import GBM_CACHE as EXT_GBM_CACHE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

KEY = ["fold", "group", "forecast_kst_dtm"]
DL_PRED_CACHE = CACHE_DIR / "exp045_dl_preds.parquet"

TAUS = {"q55": 0.55, "q65": 0.65, "q70": 0.70}
MIXES = [("q55", "q60"), ("q60", "q65"), ("q60", "q70")]
BLEND_WS = [0.75, 0.70, 0.65]

PP_CHAMP = dict(clip_lo=0.16, alpha=0.50, snap_thr=0.30, snap_to=0.94)
PP_NEW   = dict(clip_lo=0.16, alpha=0.40, snap_thr=0.27, snap_to=0.93)


def make_loss_tau(tau):
    def f(pred, tgt, w):
        e = pred - tgt
        asym = torch.where(e < 0, 2.0 * tau, 2.0 * (1.0 - tau))
        return (_huber(e) * asym * w).sum() / w.sum()
    return f


def make_xy(df, cols, imputer, scaler, fit=False):
    """exp-039 make_xy의 std 가중 버전 (q계열은 전부 std)."""
    sub = df.loc[df["target"].notna()]
    caps = sub["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    Xraw = sub[cols].to_numpy(dtype=np.float32)
    Xs = scaler.fit_transform(imputer.fit_transform(Xraw)) if fit else scaler.transform(imputer.transform(Xraw))
    g = sub["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    cf = (sub["target"].to_numpy() / caps).astype(np.float32)
    w = np.where(cf < 0.10, 0.05, 1.0).astype(np.float32)
    return Xs.astype(np.float32), g, cf, w


def dl_fold_predict_tau(tr_df, va_df, cols, seeds, tau):
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sca = StandardScaler()
    Xtr, gtr, ytr, wtr = make_xy(tr_df, cols, imp, sca, fit=True)
    Xva = sca.transform(imp.transform(va_df[cols].to_numpy(dtype=np.float32))).astype(np.float32)
    gva = va_df["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    loss_fn = make_loss_tau(tau)
    preds = []
    for s in range(seeds):
        m = train_one(Xtr, gtr, ytr, wtr, DL_CFG, loss_fn, s)
        preds.append(predict(m, Xva, gva))
        del m
        torch.cuda.empty_cache()
    return np.mean(preds, axis=0)


def postprocess_flex(cf_seq, cap, ws_seq, pA_seq, group, bias_map,
                     clip_lo, alpha, snap_thr, snap_to):
    base = apply_bias_correction_v2(cf_seq * cap, ws_seq, group, bias_map)
    base = apply_rolling_smoothing(pd.Series(base), 3, alpha).to_numpy().copy()
    base[pA_seq >= snap_thr] = np.maximum(base[pA_seq >= snap_thr], cap * snap_to)
    return np.clip(base, cap * clip_lo, cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-045: q60 비대칭 계수 τ 스윕 ###", flush=True)

    train_df, cols, colsets = build_inputs(["spd"])
    vcols = colsets["spd"]
    cache, champ_bias = load_gbm_cache()
    cache = cache.rename(columns={"gbm_cf": "cf_g132"})
    print(f"입력 spd={len(vcols)}개 / GBM 캐시 {cache.shape}", flush=True)

    # gspd base (exp-040 캐시)
    ext = pd.read_parquet(EXT_GBM_CACHE)
    m = cache[KEY].merge(ext[KEY + ["cf_gspd"]], on=KEY, how="left")
    assert m["cf_gspd"].notna().all(), "gspd 정렬 실패"
    cache["cf_gspd"] = m["cf_gspd"].to_numpy()

    # 기존 DL 캐시 (q60·hub — exp-039)
    prev = pd.read_parquet(E39_CACHE)
    for c in ["dl_q60", "dl_hub"]:
        mm = cache[KEY].merge(prev[KEY + [c]], on=KEY, how="left")
        assert mm[c].notna().all(), f"{c} 정렬 실패"
        cache[c] = mm[c].to_numpy()

    folds = FOLD_LIST
    fold_names = [fn for fn, _ in folds]
    hold = HOLD_FOLD["name"]
    cv_folds = [fn for fn in fold_names if fn != hold]

    # ── 신규 τ 변형 학습 (변형별 parquet 캐시) ──
    have = {}
    if DL_PRED_CACHE.exists() and not args.refresh:
        prev45 = pd.read_parquet(DL_PRED_CACHE)
        for c in [c for c in prev45.columns if c.startswith("dl_")]:
            merged = cache[KEY].merge(prev45[KEY + [c]], on=KEY, how="left")
            if merged[c].notna().all():
                have[c[3:]] = merged[c].to_numpy()
        if have:
            print(f"exp-045 DL 캐시 로드: {sorted(have)}", flush=True)

    for v, tau in TAUS.items():
        if v in have:
            continue
        out = np.full(len(cache), np.nan)
        for fname, fold in folds:
            tf = time.time()
            tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
            sel = (cache["fold"] == fname).to_numpy()
            keys = cache.loc[sel, ["group", "forecast_kst_dtm"]]
            va = keys.merge(train_df, left_on=["group", "forecast_kst_dtm"],
                            right_on=["kpx_group", "forecast_kst_dtm"], how="left")
            assert len(va) == len(keys), f"{v}/{fname} 검증행 정렬 실패"
            out[sel] = dl_fold_predict_tau(tr_all, va, vcols, args.seeds, tau)
            print(f"  [dl:{v}(tau={tau}):{fname}] {time.time()-tf:.0f}s", flush=True)
        have[v] = out
        cache[f"dl_{v}"] = out
        keep = KEY + [f"dl_{k}" for k in have]
        cache[keep].to_parquet(DL_PRED_CACHE, index=False)
        print(f"  캐시 저장 ({sorted(have)}) → {DL_PRED_CACHE.name}", flush=True)

    for v in have:
        cache[f"dl_{v}"] = have[v]
    for a, b in MIXES:
        cache[f"dl_{a}+{b}"] = 0.5 * (cache[f"dl_{a}"].to_numpy() + cache[f"dl_{b}"].to_numpy())

    # ── 평가 (gspd base + 신PP) ──
    slices = build_slices(cache)
    gspd = cache["cf_gspd"].to_numpy()

    def evaluate(dl_col, w, pp, gbm_arr=None, gbm_col=None):
        gbm = gbm_arr if gbm_arr is not None else (cache[gbm_col].to_numpy() if gbm_col else gspd)
        blend = gbm if dl_col is None else w * gbm + (1 - w) * cache[dl_col].to_numpy()
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

    r_champ = evaluate("dl_hub", 0.70, PP_CHAMP, gbm_col="cf_g132")
    r_ref   = evaluate("dl_q60", 0.70, PP_NEW)
    c_cv, c_hd = r_champ["cv"][0], r_champ["hold"][0]
    ref_cv, ref_hd = r_ref["cv"][0], r_ref["hold"][0]
    ref_n, ref_f = r_ref["cv"][1], r_ref["cv"][2]
    print(f"\n챔피언(g132·hub·w70·구PP): CV {c_cv:.4f} | HOLD {c_hd:.4f}  (기대 0.6448/0.6434)")
    print(f"ref(gspd·q60·w70·신PP):   CV {ref_cv:.4f} (1-NMAE {ref_n:.4f} FiCR {ref_f:.4f}) | "
          f"HOLD {ref_hd:.4f}  (기대 0.6481/0.6463)")

    cand = list(TAUS) + [f"{a}+{b}" for a, b in MIXES]
    res = {}
    for cnd in cand:
        for w in BLEND_WS:
            if "+" in cnd and w != 0.70:
                continue
            res[f"{cnd}_w{int(w*100)}"] = evaluate(f"dl_{cnd}", w, PP_NEW)

    # ── 리포트 ──
    print("\n" + "=" * 108)
    print(f"{'변형':<14}{'CV':>8}{'ΔCVref':>9}{'HOLD':>8}{'Δhdref':>9}"
          f"{'1-NMAE':>9}{'Δn':>8}{'FiCR':>8}{'Δf':>8}{'ΔCVch':>9}{'Δhdch':>9}  판정")
    rows = []
    for name, r in sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - ref_cv, kv[1]["hold"][0] - ref_hd)):
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
        rows.append(dict(name=name, folds=fold_d, cons=cons, both=both, dcv=dcv, dhd=dhd))

    print(f"\n--- fold별 ΔCV (ref 대비) ---")
    print(f"{'변형':<14}" + "".join(f"{fn.replace('fold_',''):>10}" for fn in cv_folds) + "  fold일관 두지표")
    for r in sorted(rows, key=lambda r: -min(r["dcv"], r["dhd"]))[:8]:
        print(f"{r['name']:<14}" + "".join(f"{x:>+10.4f}" for x in r["folds"])
              + f"{'  O' if r['cons'] else '  X':>8}{'  O' if r['both'] else '  X':>7}")

    # ── 진단: signed 오차·GBM 오차 상관 (평가대상 시간) ──
    caps = cache["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    acf = cache["target"].to_numpy() / caps
    ev = (cache["target"].to_numpy() >= caps * 0.10) & np.isfinite(acf)
    eg = gspd - acf
    diag = []
    for v in ["hub", "q55", "q60", "q65", "q70"]:
        e = cache[f"dl_{v}"].to_numpy() - acf
        diag.append({"variant": v, "corr_vs_gspd": float(np.corrcoef(eg[ev], e[ev])[0, 1]),
                     "dl_mae": float(np.abs(e[ev]).mean()), "signed": float(e[ev].mean())})
    print("\n--- DL 단독 품질 / gspd 오차 상관 ---")
    print(pd.DataFrame(diag).round(4).to_string(index=False))

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
