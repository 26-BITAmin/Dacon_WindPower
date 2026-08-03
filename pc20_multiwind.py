# ═══════════════════════════════════════════════════════
# PC-20 (R20): 다중 소스 풍속 v-멤버 — ldaps·ec·gefs를 ens와 병행 + 스프레드 캘리브레이션
#
# 근거: PC-19까지의 관찰 — 같은 데이터도 피처 채널(PC-10 ens3 중립)과
#   v-공간 채널(PC-18b ens_ws +0.0004)의 결과가 갈린다 = 채널이 관건.
#   그렇다면 이미 피처에 있는 ldaps_ws_10m·ec_ws·gfs_ws_10m도 v-멤버 후보다.
#   추가: ens 캘리브레이션에 스프레드 항 (v_t ~ a·ws + c·spread + b).
#
# 판정: Δ = v1 · 챔피언 병기. 기저 = PC-19 정점 (0.75, 0.15, 0.025, 0.075 ens).
# 재학습 0. 실행: C:\miniconda3\python.exe -X utf8 pc20_multiwind.py
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

BASE = dict(cf_pcden=0.75, dl_dlvc=0.15, dl_cln03=0.025)   # + ens 0.075 = PC-19 정점
RAWS = ["ens_ws", "ldaps_ws_10m", "ec_ws", "gfs_ws_10m", "gefs_ws"]


def main():
    t0 = time.time()
    print("### PC-20: 다중 소스 풍속 v-멤버 (재학습 0) ###", flush=True)
    cache, champ_bias = load_cache_full()
    for f, cc in [("exp082_gbm_pcinv.parquet", ["cf_pcinv"]),
                  ("exp082_gbm_pcden.parquet", ["cf_pcden"]),
                  ("pc05_dlvc.parquet", ["dl_dlvc"])]:
        merge_cached_cols(cache, CACHE_DIR / f, [c for c in cc if c not in cache.columns])
    train_df, _ = build_train_frame()
    scada_h = build_scada_hourly()
    ens = build_ens_frame()
    gefs = build_gefs_frame()
    train_df = train_df.merge(ens, on=["forecast_kst_dtm", "kpx_group"], how="left")
    train_df = train_df.merge(gefs, on=["forecast_kst_dtm", "kpx_group"], how="left")
    # 평가행에 원시 풍속·스프레드 병합 (train_df에서 끌어옴 — 평가행도 train_df에 존재)
    src = train_df[["kpx_group", "forecast_kst_dtm"] + RAWS + ["ens_spread"]] \
        .rename(columns={"kpx_group": "group"})
    m = cache[["group", "forecast_kst_dtm"]].merge(src, on=["group", "forecast_kst_dtm"], how="left")
    for c in RAWS + ["ens_spread"]:
        cache[c] = m[c].to_numpy()
        print(f"  {c} 평가행 커버리지 {cache[c].notna().mean():.3f}", flush=True)

    # fold×그룹 사전 계산
    reg = {}
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
            v = {c: inv(cache.loc[sel, c].to_numpy()) for c in BASE}
            v_base = sum(vv for vv in v.values()) / len(v)
            v_t_tr = inv((tr_g["target"] / cap).to_numpy())
            cal = {}
            for c in RAWS:
                ws_tr = tr_g[c].to_numpy() if c in tr_g.columns else np.full(len(tr_g), np.nan)
                ok = ~np.isnan(ws_tr) & ~np.isnan(v_t_tr)
                if ok.sum() < 200:
                    cal[c] = np.full(int(sel.sum()), np.nan)
                    continue
                a, b = np.polyfit(ws_tr[ok], v_t_tr[ok], 1)
                ws_ev = cache.loc[sel, c].to_numpy()
                cal[c] = np.where(np.isnan(ws_ev), np.nan, a * ws_ev + b)
            # ens 2D (ws + spread)
            ws_tr = tr_g["ens_ws"].to_numpy(); sp_tr = tr_g["ens_spread"].to_numpy()
            ok = ~np.isnan(ws_tr) & ~np.isnan(sp_tr) & ~np.isnan(v_t_tr)
            A = np.column_stack([ws_tr[ok], sp_tr[ok], np.ones(ok.sum())])
            coef, *_ = np.linalg.lstsq(A, v_t_tr[ok], rcond=None)
            ws_ev = cache.loc[sel, "ens_ws"].to_numpy(); sp_ev = cache.loc[sel, "ens_spread"].to_numpy()
            cal["ens2d"] = np.where(np.isnan(ws_ev) | np.isnan(sp_ev), np.nan,
                                    coef[0] * ws_ev + coef[1] * sp_ev + coef[2])
            for c in cal:
                cal[c] = np.where(np.isnan(cal[c]), v_base, cal[c])
            reg[(fname, group)] = (fwd, sel, v, cal)
    print(f"준비 완료 ({time.time()-t0:.0f}s)", flush=True)

    ev = make_ev(cache, champ_bias)
    v1 = ev("cf_pcinv", "dl_cln03", 0.65, PP_NEW)
    champ = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    v1_cv, v1_hd = v1["cv"][0], v1["hold"][0]
    ch_cv, ch_hd = champ["cv"][0], champ["hold"][0]
    cv_folds = list(v1["folds"].keys())
    print(f"v1 {v1_cv:.4f}/{v1_hd:.4f} | 챔피언 {ch_cv:.4f}/{ch_hd:.4f}", flush=True)

    res = {}
    def score(raw_mix, label):
        """raw_mix: {cal이름: 지분}. 기저 3멤버는 (1-Σ지분) 비례 축소."""
        tot = sum(raw_mix.values())
        vals = np.full(len(cache), np.nan)
        for (fname, group), (fwd, sel, v, cal) in reg.items():
            vmix = (1 - tot) / sum(BASE.values()) * sum(w * v[c] for c, w in BASE.items())
            for c, w in raw_mix.items():
                vmix = vmix + w * cal[c]
            vals[sel] = fwd(vmix)
        assert not np.isnan(vals).any()
        cache[label] = vals
        res[label] = ev(label, None, 1.0, PP_NEW)

    score({"ens_ws": 0.075}, "ens075(기준)")
    score({"ens2d": 0.075}, "ens2d075")
    for c in ("ldaps_ws_10m", "ec_ws", "gfs_ws_10m", "gefs_ws"):
        score({"ens_ws": 0.075, c: 0.025}, f"ens+{c.split('_')[0]}25")
        score({"ens_ws": 0.05, c: 0.05}, f"ens50+{c.split('_')[0]}50")
    score({"ens_ws": 0.05, "ldaps_ws_10m": 0.025, "ec_ws": 0.025}, "ens+ld+ec")
    score({"ens_ws": 0.10}, "ens100")

    print(f"\n{'셀':<16}{'CV':>8}{'Δv1':>9}{'HOLD':>8}{'Δv1':>9}{'Δn':>9}{'Δf':>9}"
          f"{'Δ챔CV':>9}{'Δ챔hd':>9}  fold(Δv1)", flush=True)
    for name, r in sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - v1_cv,
                                                           kv[1]["hold"][0] - v1_hd)):
        fd = " ".join(f"{r['folds'][fn]-v1['folds'][fn]:+.4f}" for fn in cv_folds)
        print(f"{name:<16}{r['cv'][0]:>8.4f}{r['cv'][0]-v1_cv:>+9.4f}"
              f"{r['hold'][0]:>8.4f}{r['hold'][0]-v1_hd:>+9.4f}"
              f"{r['cv'][1]-v1['cv'][1]:>+9.4f}{r['cv'][2]-v1['cv'][2]:>+9.4f}"
              f"{r['cv'][0]-ch_cv:>+9.4f}{r['hold'][0]-ch_hd:>+9.4f}  {fd}", flush=True)
    print(f"총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
