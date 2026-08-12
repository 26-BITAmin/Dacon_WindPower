# ═══════════════════════════════════════════════════════
# wi01: DL 입력 풍속→BIV 분포 교체 — 6변형 게이트 (wind_input ablation 2/3)
#
# 사용자 지시 (2026-08-05): 조작 변인 = DL 입력만. GBM(cf_ens3)·w0.65·PP(스냅OFF)·
#   편향맵·clip·평활·그룹스케일 전부 동결. 제거 범위와 분포 형태를 ablation:
#
#   제거 범위 3종 (DL 348피처 기준):
#     ws  : 풍속 크기 계열만 제거 (~30) — ws·시어·거스트·ws파생·tmp·spa.
#           풍향 sin/cos·u/v 성분·격자편차는 유지
#     lvl : 레벨 바람 전부 제거 (~76) — ws + 풍향 + u/v 성분 + XBLWS 등.
#           격자편차 216은 유지 (exp-035 +0.0051의 원천 보존)
#     full: 바람 전부 제거 (~231) — lvl + 격자편차의 바람 채널 155
#   분포 형태 2종: b5 = (μu,μv,σu,σv,ρ) / b9 = b5 + (μws, 방향sin/cos, σws)
#   → 6변형: ws_b5, ws_b9, lvl_b5, lvl_b9, full_b5, full_b9
#
# 판정 (t03·exp-122 관례):
#   기준 = 챔피언 dl_cln03 (캐시, 재학습 없음) × cf_ens3 w0.65, 스냅 OFF
#   작동점 2개: 미스케일 / 제출 스케일 ×.98/.98/.97 (exp-123 규칙)
#   Δn/Δf 분리 · 평균 이동(%cap, 역전군 판별) · w {0.55~0.70} 스윕 · noQ4 근사
#   노이즈: 게이트 1σ ≈ 0.0003 (PC-30)
#
# 실행(루트에서):
#   C:\miniconda3\python.exe -X utf8 "open (1)/testing/wind_input/wi01_bivgate.py" --stage list
#   C:\miniconda3\python.exe -X utf8 "open (1)/testing/wind_input/wi01_bivgate.py" --stage full
#   병렬 분할(사용자 "GPU 많이 활용" 지시, 2026-08-05): 변형 부분집합 × 전용 캐시 태그
#     ... --stage full --variants ws_b5,ws_b9 [--tag ws]   (프로세스별 캐시 분리 = 경합 방지)
#     ... --stage eval                                     (전 캐시 취합 후 채점만)
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import CACHE_DIR, FOLD_LIST, build_inputs, score_from_tidy
from exp039_dl_diverse import build_slices
from exp049_clean import PP_NEW, dl_fold_predict_clean, merge_avail, postprocess_flex
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp060_pergroup_hpo import load_cache_full
from exp073_gate import ENS_CACHE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BIV_PATH = HERE.parent / "wi00_biv_feats.parquet"
CSV_OUT = HERE.parent / "wi01_results.csv"
ALL_VARIANTS = ("ws_b5", "ws_b9", "lvl_b5", "lvl_b9", "full_b5", "full_b9")
HOLD = HOLD_FOLD["name"]
SCALES = {1: 0.98, 2: 0.98, 3: 0.97}
WS_SWEEP = (0.55, 0.60, 0.65, 0.70)
NOSNAP = dict(PP_NEW, snap_thr=1.1)                  # 현 챔피언 = 스냅 OFF (exp-126)
BIV5 = ["biv_mu_u", "biv_mu_v", "biv_sig_u", "biv_sig_v", "biv_rho"]
BIV9 = BIV5 + ["biv_ws", "biv_dsin", "biv_dcos", "biv_sigws"]

# ── 제거 목록 (레벨 132 안) — 전 항목 존재를 assert로 검수 ──
WS_MAG = [
    "ldaps_ws_10m", "gfs_ws_10m", "gfs_ws_80m", "gfs_ws_100m", "gfs_ws_pbl",
    "gfs_wind_shear_100_10", "gfs_wind_shear_80_10", "gfs_surface_0_gust",
    "ws_disagreement", "ws_x_season", "disagreement_x_season",
    "phy_ws10_cu", "phy_ws100_cu", "phy_rho_ws100_cu", "phy_ws50max",
    "phy_gust_range", "phy_gust_over_ws", "phy_gfs_gust_factor",
    "phy_ws850", "phy_shear_850_100", "phy_pc100",
    "jgsm_ws", "jmsm_ws", "jgsm_dis", "jmsm_dis",
]  # + tmp_* · spa_* (프리픽스 매칭)
WS_MAG_PREFIX = ("tmp_", "spa_")

LVL_EXTRA = [
    "ldaps_wd_sin_10m", "ldaps_wd_cos_10m",
    "gfs_wd_sin_10m", "gfs_wd_cos_10m", "gfs_wd_sin_80m", "gfs_wd_cos_80m",
    "gfs_wd_sin_100m", "gfs_wd_cos_100m", "gfs_wd_sin_pbl", "gfs_wd_cos_pbl",
    "wd_agreement", "jgsm_wdsin", "jgsm_wdcos", "jmsm_wdsin", "jmsm_wdcos",
    "ldaps_heightAboveGround_10_10u", "ldaps_heightAboveGround_10_10v",
    "ldaps_heightAboveGround_50_50MUmax", "ldaps_heightAboveGround_50_50MUmin",
    "ldaps_heightAboveGround_50_50MVmax", "ldaps_heightAboveGround_50_50MVmin",
    "ldaps_heightAboveGround_5_XBLWS", "ldaps_heightAboveGround_5_YBLWS",
    "gfs_heightAboveGround_10_10u", "gfs_heightAboveGround_10_10v",
    "gfs_heightAboveGround_80_u", "gfs_heightAboveGround_80_v",
    "gfs_heightAboveGround_100_100u", "gfs_heightAboveGround_100_100v",
    "gfs_planetaryBoundaryLayer_0_u", "gfs_planetaryBoundaryLayer_0_v",
    "gfs_isobaricInhPa_850_u", "gfs_isobaricInhPa_850_v",
    "gfs_isobaricInhPa_700_u", "gfs_isobaricInhPa_700_v",
    "gfs_isobaricInhPa_500_u", "gfs_isobaricInhPa_500_v",
]

# 격자편차 216 안의 바람 채널 변수 토큰 (sdL_/sdG_ 프리픽스 뒤)
SPD_WIND_TOKENS = ("u10", "v10", "u100", "v100", "ws10", "ws100",
                   "ws850", "wsbl", "ws50x", "gust")


def spd_wind_cols(vcols):
    out = []
    for c in vcols:
        if c.startswith(("sdL_", "sdG_")):
            token = c.split("_", 2)[1]
            if token in SPD_WIND_TOKENS:
                out.append(c)
    return out


def make_colsets(vcols):
    """{범위: (제거목록, 잔여목록)} — 존재 검수 포함."""
    ws_mag = list(WS_MAG) + [c for c in vcols if c.startswith(WS_MAG_PREFIX)]
    missing = [c for c in WS_MAG + LVL_EXTRA if c not in vcols]
    assert not missing, f"제거 대상 부재: {missing}"
    lvl = ws_mag + LVL_EXTRA
    full = lvl + spd_wind_cols(vcols)
    sets = {}
    for name, drop in (("ws", ws_mag), ("lvl", lvl), ("full", full)):
        drop_s = set(drop)
        assert len(drop_s) == len(drop), f"{name}: 제거 목록 중복"
        keep = [c for c in vcols if c not in drop_s]
        sets[name] = (drop, keep)
    return sets


def merge_biv(df, biv_f):
    """fold별 BIV 9열을 (kpx_group, dtm) 키로 병합."""
    m = df.merge(biv_f[["kpx_group", "forecast_kst_dtm"] + BIV9],
                 on=["kpx_group", "forecast_kst_dtm"], how="left")
    assert len(m) == len(df), "BIV 병합 행 증식"
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["list", "full", "eval"], default="list")
    ap.add_argument("--variants", type=str, default=",".join(ALL_VARIANTS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", type=str, default="",
                    help="병렬 프로세스별 캐시 접미사 (파일 경합 방지)")
    args = ap.parse_args()
    variants = [v for v in args.variants.split(",") if v]
    suffix = f"_{args.tag}" if args.tag else ""
    dl_cache = HERE.parent / f"wi01_dl_biv{suffix}.parquet"
    partial = HERE.parent / f"wi01_dl_biv{suffix}_partial.parquet"
    t0 = time.time()
    print("### wi01: DL 풍속→BIV 분포 교체 6변형 게이트 ###", flush=True)

    train_df, _, colsets = build_inputs(["spd"])
    vcols = colsets["spd"]
    train_df = merge_avail(train_df)
    sets = make_colsets(vcols)
    for name, (drop, keep) in sets.items():
        spd_part = sum(c.startswith(("sdL_", "sdG_")) for c in drop)
        print(f"  [{name}] 제거 {len(drop)} (레벨 {len(drop)-spd_part} + 격자 {spd_part}) "
              f"→ 잔여 {len(keep)} | +BIV 5/9", flush=True)
    if args.stage == "list":
        for name, (drop, _) in sets.items():
            lvl_drop = [c for c in drop if not c.startswith(("sdL_", "sdG_"))]
            print(f"\n[{name}] 레벨 제거 {len(lvl_drop)}개:", flush=True)
            for c in lvl_drop:
                print(f"    - {c}", flush=True)
        print(f"\n### --stage list 종료 ({(time.time()-t0)/60:.1f}분) ###")
        return

    biv = pd.read_parquet(BIV_PATH)
    print(f"BIV 피처 로드: {biv.shape}", flush=True)

    # ── DL 재학습 (fold 단위 partial — exp-122 관례) ──
    cache, champ_bias = load_cache_full()
    assert merge_cached_cols(cache, ENS_CACHE, ["cf_ens3"])

    if args.stage == "eval":
        # 전 캐시 취합: 병렬 프로세스가 남긴 wi01_dl_biv*.parquet 전부
        variants = list(ALL_VARIANTS)
        cols_all = [f"dl_{v}" for v in variants]
        for f in sorted(HERE.parent.glob("wi01_dl_biv*.parquet")):
            merge_cached_cols(cache, f, [c for c in cols_all if c not in cache.columns
                                         or cache[c].isna().any()])
        miss = [c for c in cols_all if c not in cache.columns or cache[c].isna().any()]
        assert not miss, f"미완성 변형: {miss}"
        cols_new = cols_all
    else:
        cols_new = [f"dl_{v}" for v in variants]
        if dl_cache.exists():
            merge_cached_cols(cache, dl_cache, cols_new)
        if partial.exists():
            merge_cached_cols(cache, partial, cols_new)

    for vname in (variants if args.stage == "full" else []):
        col = f"dl_{vname}"
        if col in cache.columns and cache[col].notna().all():
            print(f"\n[{col}] 캐시 재사용", flush=True)
            continue
        scope, form = vname.rsplit("_", 1)
        keep = sets[scope][1] + (BIV5 if form == "b5" else BIV9)
        print(f"\n[{col}] {len(keep)}피처로 DL 재학습", flush=True)
        out = cache[col].to_numpy().copy() if col in cache.columns else np.full(len(cache), np.nan)
        for fname, fold in FOLD_LIST:
            sel = (cache["fold"] == fname).to_numpy()
            if not np.isnan(out[sel]).any():
                continue
            tf = time.time()
            biv_f = biv[biv["fold"] == fname]
            tr_all = merge_biv(train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]],
                               biv_f)
            keys = cache.loc[sel, ["group", "forecast_kst_dtm"]]
            va = keys.merge(merge_biv(train_df, biv_f),
                            left_on=["group", "forecast_kst_dtm"],
                            right_on=["kpx_group", "forecast_kst_dtm"], how="left")
            assert len(va) == len(keys), "정렬 실패"
            cov_tr = tr_all.loc[tr_all["target"].notna(), "biv_mu_u"].notna().mean()
            cov_va = va["biv_mu_u"].notna().mean()
            assert cov_va > 0.99, f"검증 BIV 커버 {cov_va:.3f}"
            out[sel] = dl_fold_predict_clean(tr_all, va, keep, args.seeds, "cln03")
            cache[col] = out
            save_cols(cache, partial, [c for c in cols_new if c in cache.columns])
            print(f"  [{col}:{fname}] {time.time()-tf:.0f}s "
                  f"(BIV 커버 tr {cov_tr:.3f} / va {cov_va:.3f})", flush=True)
        cache[col] = out
    if args.stage == "full":
        save_cols(cache, dl_cache, cols_new)
        if set(variants) != set(ALL_VARIANTS):
            print(f"\n부분 실행 완료 ({variants}) — 전 변형 취합 채점은 --stage eval "
                  f"| 총 소요: {(time.time()-t0)/60:.1f}분", flush=True)
            return

    # ── 멤버 프로파일 ──
    mm = cache["target"].notna().to_numpy()
    print("\n[멤버 프로파일]", flush=True)
    print(f"  {'멤버':<12}{'corr(cf_ens3)':>15}{'corr(dl_cln03)':>16}{'평균이동 vs dl_cln03':>22}",
          flush=True)
    for c in ["dl_cln03"] + cols_new:
        r = np.corrcoef(cache.loc[mm, c], cache.loc[mm, "cf_ens3"])[0, 1]
        rd = np.corrcoef(cache.loc[mm, c], cache.loc[mm, "dl_cln03"])[0, 1]
        sh = (cache[c].to_numpy() - cache["dl_cln03"].to_numpy())[mm].mean() * 100
        print(f"  {c:<12}{r:>15.4f}{rd:>16.4f}{sh:>+21.3f}%cap", flush=True)

    # ── 게이트: 스냅 OFF 고정 × 작동점 2개 (t03 관례) ──
    slices = build_slices(cache)
    cap_arr = cache["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{g}"]).to_numpy(float)

    def pp_of(raw, scaled):
        out = np.full(len(cache), np.nan)
        for (fname, group), sub in slices:
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            p = postprocess_flex(raw[sub.index], cap, sub["ws"].to_numpy(),
                                 sub["pA"].to_numpy(), group, champ_bias[fname], **NOSNAP)
            out[sub.index] = np.clip(p * SCALES[group], 0, cap) if scaled else p
        return out

    def sc_of(pred):
        t = pd.DataFrame({"fold": cache["fold"], "group": cache["group"],
                          "forecast_kst_dtm": cache["forecast_kst_dtm"],
                          "target": cache["target"], "pred": pred})
        c = t[t["fold"] != HOLD]
        return (score_from_tidy(c, "pred"), score_from_tidy(t[t["fold"] == HOLD], "pred"),
                score_from_tidy(c[c["fold"] != "fold_2024Q4"], "pred"))

    rows = []
    for scaled in (False, True):
        tag = "제출 작동점 ×.98/.98/.97" if scaled else "미스케일"
        print("\n" + "=" * 110, flush=True)
        print(f"═══ 스냅 OFF | {tag} ═══", flush=True)
        res = {}
        for dl in ["dl_cln03"] + cols_new:
            for w in WS_SWEEP:
                p = pp_of(w * cache["cf_ens3"].to_numpy() + (1 - w) * cache[dl].to_numpy(),
                          scaled)
                res[(dl, w)] = (sc_of(p), p)
        ref, refp = res[("dl_cln03", 0.65)]
        print(f"  {'DL 멤버':<11}{'w':>6}{'CV':>9}{'ΔCV':>9}{'Δhd':>9}{'ΔnoQ4':>9}"
              f"{'Δn':>9}{'Δf':>9}{'이동%cap':>10}", flush=True)
        for dl in ["dl_cln03"] + cols_new:
            for w in WS_SWEEP:
                s, p = res[(dl, w)]
                mark = "  <-기준" if (dl == "dl_cln03" and w == 0.65) else ""
                print(f"  {dl:<11}{w:>6.2f}{s[0][0]:>9.4f}{s[0][0]-ref[0][0]:>+9.4f}"
                      f"{s[1][0]-ref[1][0]:>+9.4f}{s[2][0]-ref[2][0]:>+9.4f}"
                      f"{s[0][1]-ref[0][1]:>+9.4f}{s[0][2]-ref[0][2]:>+9.4f}"
                      f"{np.nanmean((p-refp)/cap_arr)*100:>+10.3f}{mark}", flush=True)
                rows.append(dict(scaled=scaled, dl=dl, w=w, cv=s[0][0],
                                 d_cv=s[0][0]-ref[0][0], d_hold=s[1][0]-ref[1][0],
                                 d_noq4=s[2][0]-ref[2][0], d_n=s[0][1]-ref[0][1],
                                 d_f=s[0][2]-ref[0][2],
                                 shift_pcap=float(np.nanmean((p-refp)/cap_arr)*100)))

    pd.DataFrame(rows).to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print(f"\n상세 → {CSV_OUT.name} | 총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
