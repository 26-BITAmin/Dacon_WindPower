# ═══════════════════════════════════════════════════════
# PC-18b: 앙상블 풍속을 v-공간 **직접 멤버**로 — 트랙 고유 데이터 채널
#
# 근거: PC-13이 실증한 원리 — 같은 데이터도 피처 채널은 벽(corr 0.999+)이지만
#   트랙 고유 채널(커브·v-공간)은 통한다. ens_ws(ECMWF 앙상블 평균)·gefs_ws(NOAA)는
#   **물리 풍속이라 v-공간의 네이티브 양** — GBM을 거치지 않고 심플렉스에 직접
#   들어갈 수 있다. 10m 풍속 ↔ 유효 허브 풍속의 스케일 차이는 폴드-세이프
#   선형 캘리브레이션(학습창 라벨의 v_t에 회귀)으로 맞춘다.
#   평가행은 전부 2024라 ens(2023+)·gefs(2021+) 커버리지 1.0.
#
# 판정: Δ = v1. PC-16 정점 (0.775 pcden, 0.15 dlvc, 0.075 dC)에
#   지분 δ ∈ {0.05, 0.10}로 삽입 (기존 가중 ×(1−δ)).
# 재학습 0. 실행: C:\miniconda3\python.exe -X utf8 pc18b_rawwind.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np

from exp005_qopt import CAPACITY_KWH
from exp015_scada_pc import build_scada_hourly
from exp035_raw_dl import CACHE_DIR, FOLD_LIST
from exp049_clean import PP_NEW
from exp050_lambda_sweep import merge_cached_cols
from exp060_pergroup_hpo import load_cache_full, make_ev
from exp065_gbm3 import build_train_frame, slice_fold
from exp073_gate import build_ens_frame
from exp089_gefs_features import build_gefs_frame
from pc01_pcinv import build_curve, make_transform

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PEAK = dict(cf_pcden=0.775, dl_dlvc=0.15, dl_cln03=0.075)
DELTAS = (0.05, 0.10)


def main():
    t0 = time.time()
    print("### PC-18b: 앙상블 풍속 v-공간 직접 멤버 (재학습 0) ###", flush=True)
    cache, champ_bias = load_cache_full()
    for f, cc in [("exp082_gbm_pcinv.parquet", ["cf_pcinv"]),
                  ("exp082_gbm_pcden.parquet", ["cf_pcden"]),
                  ("pc05_dlvc.parquet", ["dl_dlvc"])]:
        merge_cached_cols(cache, CACHE_DIR / f, [c for c in cc if c not in cache.columns])

    train_df, _ = build_train_frame()
    scada_h = build_scada_hourly()
    ens = build_ens_frame()
    gefs = build_gefs_frame()
    # 캐시 행에 원시 풍속 병합
    for name, frame, col in (("ens", ens, "ens_ws"), ("gefs", gefs, "gefs_ws")):
        m = cache[["group", "forecast_kst_dtm"]].merge(
            frame.rename(columns={"kpx_group": "group"})[["group", "forecast_kst_dtm", col]],
            on=["group", "forecast_kst_dtm"], how="left")
        cache[col] = m[col].to_numpy()
        print(f"  {col} 평가행 커버리지 {cache[col].notna().mean():.3f}", flush=True)
    # 학습 프레임에도 (캘리브레이션용)
    train_df = train_df.merge(ens[["forecast_kst_dtm", "kpx_group", "ens_ws"]],
                              on=["forecast_kst_dtm", "kpx_group"], how="left")
    train_df = train_df.merge(gefs, on=["forecast_kst_dtm", "kpx_group"], how="left")

    ev = make_ev(cache, champ_bias)
    v1 = ev("cf_pcinv", "dl_cln03", 0.65, PP_NEW)
    champ = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    v1_cv, v1_hd = v1["cv"][0], v1["hold"][0]
    cv_folds = list(v1["folds"].keys())
    print(f"v1 {v1_cv:.4f}/{v1_hd:.4f} | 챔피언 {champ['cv'][0]:.4f}/{champ['hold'][0]:.4f}", flush=True)

    # fold×그룹: 커브 + 캘리브레이션 (학습창 라벨 v_t ~ a·ws_src + b) + 심플렉스 조립
    res_cols = {}
    for src in ("ens_ws", "gefs_ws"):
        for d in DELTAS:
            res_cols[(src, d)] = np.full(len(cache), np.nan)
    calib_log = []
    for fname, fold in FOLD_LIST:
        tr_all, _ = slice_fold(train_df, fold)
        for group in (1, 2, 3):
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()]
            curve = build_curve(scada_h[group], tr_g, fold["train_end"], None)
            if curve is None:
                continue
            fwd, inv, _, _ = make_transform(curve)
            sel = ((cache["fold"] == fname) & (cache["group"] == group)).to_numpy()
            v_mem = {c: inv(cache.loc[sel, c].to_numpy()) for c in PEAK}
            v_base = sum(w * v_mem[c] for c, w in PEAK.items()) / sum(PEAK.values())
            v_t_tr = inv((tr_g["target"] / cap).to_numpy())
            for src in ("ens_ws", "gefs_ws"):
                ws_tr = tr_g[src].to_numpy() if src in tr_g.columns else np.full(len(tr_g), np.nan)
                ok = ~np.isnan(ws_tr) & ~np.isnan(v_t_tr)
                if ok.sum() < 200:
                    a, b = np.nan, np.nan
                else:
                    a, b = np.polyfit(ws_tr[ok], v_t_tr[ok], 1)
                ws_ev = cache.loc[sel, src].to_numpy()
                v_cal = a * ws_ev + b if not np.isnan(a) else np.full(sel.sum(), np.nan)
                # 평가행 결측(캘리브레이션 불가 포함)은 v_base로 대체 (지분 효과 소멸 = 중립)
                v_cal = np.where(np.isnan(v_cal), v_base, v_cal)
                if fname == "fold_2024Q1":
                    calib_log.append(f"    {fname} g{group} {src}: a={a:.3f} b={b:+.2f} "
                                     f"(n={ok.sum()})")
                for d in DELTAS:
                    vmix = (1 - d) * sum(w * v_mem[c] for c, w in PEAK.items()) + d * v_cal
                    res_cols[(src, d)][sel] = fwd(vmix)
    print("\n[캘리브레이션 예시 (Q1)]", flush=True)
    for line in calib_log:
        print(line, flush=True)

    res = {}
    for (src, d), vals in res_cols.items():
        assert not np.isnan(vals).any()
        label = f"{src.split('_')[0]}_d{int(d*100)}"
        cache[label] = vals
        res[label] = ev(label, None, 1.0, PP_NEW)
    # 기준: PEAK 자체 (지분 0)
    vals = np.full(len(cache), np.nan)
    for fname, fold in FOLD_LIST:
        tr_all, _ = slice_fold(train_df, fold)
        for group in (1, 2, 3):
            tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & (tr_all["target"].notna())]
            curve = build_curve(scada_h[group], tr_g, fold["train_end"], None)
            if curve is None:
                continue
            fwd, inv, _, _ = make_transform(curve)
            sel = ((cache["fold"] == fname) & (cache["group"] == group)).to_numpy()
            v_mem = {c: inv(cache.loc[sel, c].to_numpy()) for c in PEAK}
            vals[sel] = fwd(sum(w * v_mem[c] for c, w in PEAK.items()))
    cache["peak0"] = vals
    res["peak(δ=0)"] = ev("peak0", None, 1.0, PP_NEW)

    print(f"\n{'셀':<14}{'CV':>8}{'Δv1':>9}{'HOLD':>8}{'Δv1':>9}{'Δn':>9}{'Δf':>9}  fold(Δv1)", flush=True)
    for name, r in sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - v1_cv,
                                                           kv[1]["hold"][0] - v1_hd)):
        fd = " ".join(f"{r['folds'][fn]-v1['folds'][fn]:+.4f}" for fn in cv_folds)
        print(f"{name:<14}{r['cv'][0]:>8.4f}{r['cv'][0]-v1_cv:>+9.4f}"
              f"{r['hold'][0]:>8.4f}{r['hold'][0]-v1_hd:>+9.4f}"
              f"{r['cv'][1]-v1['cv'][1]:>+9.4f}{r['cv'][2]-v1['cv'][2]:>+9.4f}  {fd}", flush=True)
    print(f"총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
