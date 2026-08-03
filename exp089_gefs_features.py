# ═══════════════════════════════════════════════════════
# exp-089 피처·게이트: NOAA GEFS 앙상블 10m 바람 (평균·스프레드)
#
# 수집은 exp089_fetch_gefs.py (WSL). 여기서는 시간보간 → 피처 3개 → 이중 게이트.
#
# 피처 (exp-073 ENS3와 같은 최소 블록 — 검증된 형태를 그대로 미러):
#   gefs_ws     = hypot(u_avg, v_avg)        앙상블 평균 풍속
#   gefs_spread = hypot(u_spr, v_spr)        앙상블 스프레드(불확실성)
#   gefs_gap    = gefs_ws − gfs_ws_10m       앙상블 − 결정론(0.25° GFS, 기존 피처)
#   ※ exp-073은 멤버별 ws를 평균했으나 NOAA는 성분(u,v)을 미리 평균해 올리므로
#     여기서는 벡터평균 풍속이다 (스칼라평균 ≥ 벡터평균 — 동일량 아님, 기록만).
#
# 게이트 변형 2개:
#   gf3    = 352 + GEFS3          → 판정 Δ = cf_hpo   (GEFS 단독 효과)
#   gfens  = 352 + ENS3 + GEFS3   → 판정 Δ = cf_ens3  (**ECMWF 앙상블을 넘어서는가** —
#            보드 챔피언이 이미 ens3를 갖고 있으므로 이쪽이 의사결정용)
#
# 실행: C:\miniconda3\python.exe -X utf8 exp089_gefs_features.py --stage signal
#       C:\miniconda3\python.exe -X utf8 exp089_gefs_features.py --stage gate
# ═══════════════════════════════════════════════════════
import argparse
import glob
import json
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, DATA_DIR
from exp035_raw_dl import FOLD_LIST, CACHE_DIR
from exp049_clean import KEY, PP_NEW
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp056_hpo import RESULT_JSON as E56_JSON, train_gate_fold
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp065_gbm3 import build_train_frame
from exp073_gate import build_ens_frame, ENS3

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEC_DIR = DATA_DIR / "external" / "gefs_decoded"
GEFS3 = ["gefs_ws", "gefs_spread", "gefs_gap"]
VARIANTS = {"gf3": ("cf_hpo", []), "gfens": ("cf_ens3", ENS3)}


def build_gefs_frame():
    """gefs_decoded/*.csv → 시간별 KST 피처 프레임 (exp-073 보간 규약과 동일)."""
    files = sorted(glob.glob(str(DEC_DIR / "*.csv")))
    assert files, "수집 CSV 없음"
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    target_h = np.arange(28, 52, dtype=float)          # D 01:00 ~ D+1 00:00 KST
    frames = []
    for (init_d, g), sub in raw.groupby(["init", "group"]):
        sub = sub.sort_values("step")
        if len(sub) < 2:
            continue
        h = sub["step"].to_numpy(dtype=float)
        u = np.interp(target_h, h, sub["u_avg"].to_numpy())
        v = np.interp(target_h, h, sub["v_avg"].to_numpy())
        su = np.interp(target_h, h, sub["u_spr"].to_numpy())
        sv = np.interp(target_h, h, sub["v_spr"].to_numpy())
        kst = (pd.Timestamp(str(init_d)) + pd.Timedelta(hours=12 + 9)
               + pd.to_timedelta(target_h, unit="h"))
        frames.append(pd.DataFrame({"forecast_kst_dtm": kst, "kpx_group": int(g),
                                    "gefs_ws": np.hypot(u, v),
                                    "gefs_spread": np.hypot(su, sv)}))
    out = (pd.concat(frames, ignore_index=True)
             .drop_duplicates(["forecast_kst_dtm", "kpx_group"], keep="last"))
    print(f"GEFS frame: {len(out)}행 | 수집일 {len(files)}", flush=True)
    return out


def prepare(with_ens):
    train_df, use_cols = build_train_frame()           # 352 (spd + EC)
    gf = build_gefs_frame()
    train_df = train_df.merge(gf, on=["forecast_kst_dtm", "kpx_group"], how="left")
    train_df["gefs_gap"] = train_df["gefs_ws"] - train_df["gfs_ws_10m"]
    cols = list(use_cols)
    if with_ens:
        ens = build_ens_frame()
        train_df = train_df.merge(ens, on=["forecast_kst_dtm", "kpx_group"], how="left")
        train_df["ens_gap"] = train_df["ens_ws"] - train_df["ec_ws"]
        cols += ENS3
    cov = train_df.groupby(train_df["forecast_kst_dtm"].dt.year)["gefs_ws"] \
                  .apply(lambda d: d.notna().mean().round(3))
    print("연도별 GEFS 커버리지:\n" + cov.to_string(), flush=True)
    return train_df, cols + GEFS3


def stage_signal():
    """게이트 전 저비용 go/no-go — 라벨 상관이 기존 소스를 넘는가."""
    train_df, _ = prepare(with_ens=True)
    for g in [1, 2, 3]:
        m = (train_df["target"].notna() & train_df["gefs_ws"].notna()
             & (train_df["kpx_group"] == g))
        cf = train_df.loc[m, "target"] / CAPACITY_KWH[f"kpx_group_{g}"]
        r = {c: np.corrcoef(train_df.loc[m, c], cf)[0, 1]
             for c in ("gefs_ws", "ens_ws", "ec_ws", "gfs_ws_10m", "ldaps_ws_10m")}
        print(f"  그룹{g} (라벨행 {m.sum()}): " +
              " | ".join(f"{k} {v:.4f}" for k, v in r.items()), flush=True)
    m = train_df["gefs_ws"].notna() & train_df["ens_ws"].notna()
    print(f"\n  corr(gefs_ws, ens_ws) = "
          f"{np.corrcoef(train_df.loc[m,'gefs_ws'], train_df.loc[m,'ens_ws'])[0,1]:.4f}"
          f"  ← 1에 가까울수록 ECMWF 앙상블과 중복", flush=True)
    print(f"  스프레드 평균 GEFS {train_df['gefs_spread'].mean():.2f} m/s", flush=True)


def stage_gate(tags, seeds, jobs):
    e56 = json.loads(E56_JSON.read_text())
    cache, champ_bias = load_cache_full()
    assert "cf_ens3" in merge_cached_cols(
        cache, CACHE_DIR / "exp073_gbm_ens.parquet", ["cf_ens3"]), "exp-073 ens 캐시 없음"

    for tag in tags:
        ref_col, extra = VARIANTS[tag]
        col = f"cf_{tag}"
        cpath = CACHE_DIR / f"exp089_gbm_{tag}.parquet"
        ppath = CACHE_DIR / f"exp089_gbm_{tag}_partial.parquet"
        if col in merge_cached_cols(cache, cpath, [col]):
            continue
        train_df, cols = prepare(with_ens=bool(extra))
        print(f"\n[{tag}] {len(cols)}피처 | 판정 기준 {ref_col}", flush=True)
        out = np.full(len(cache), np.nan)
        if ppath.exists():
            prev = pd.read_parquet(ppath)
            out = cache[KEY].merge(prev[KEY + [col]], on=KEY, how="left")[col].to_numpy()
        for fname, fold in FOLD_LIST:
            sel = (cache["fold"] == fname).to_numpy()
            if not np.isnan(out[sel]).any():
                continue
            tf = time.time()
            p = train_gate_fold(train_df, cols, fold, seeds, jobs,
                                e56["lgb"], e56["lgb_n"], e56["xgb"], e56["xgb_n"])
            m = cache.loc[sel, KEY].merge(p.rename(columns={"cf_hpo": col}), on=KEY, how="left")
            assert m[col].notna().all(), f"정렬 실패 {fname}"
            out[sel] = m[col].to_numpy()
            cache[col] = out
            cache[KEY + [col]].to_parquet(ppath, index=False)
            print(f"  [{tag}:{fname}] {time.time()-tf:.0f}s", flush=True)
        cache[col] = out
        save_cols(cache, cpath, [col])

    mm = cache["target"].notna().to_numpy()
    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} | HOLD {ref['hold'][0]:.4f}"
          f"  (기대 0.6556/0.6535)", flush=True)

    res = {"hpo_solo": ev("cf_hpo", None, 1.0, PP_NEW),
           "ens3_w65": ev("cf_ens3", "dl_cln03", 0.65, PP_NEW)}
    for tag in tags:
        ref_col, _ = VARIANTS[tag]
        col = f"cf_{tag}"
        v, b = cache[col].to_numpy(), cache[ref_col].to_numpy()
        print(f"  corr({col}, {ref_col}) = {np.corrcoef(v[mm], b[mm])[0,1]:.4f} | "
              f"평균 이동 {(v-b).mean()*100:+.3f}%cap", flush=True)
        res[f"{tag}_solo"] = ev(col, None, 1.0, PP_NEW)
        for w in (0.65, 0.70):
            res[f"{tag}_w{int(w*100)}"] = ev(col, "dl_cln03", w, PP_NEW)
    print_table(res, ref)
    print("\n※ gf3는 cf_hpo 대비, gfens는 cf_ens3(= ens3_w65 행) 대비로 읽을 것 —"
          " 표의 Δ는 전부 챔피언(hpo×dC_w65) 기준이다.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["signal", "gate"], default="signal")
    ap.add_argument("--variants", default="gf3,gfens")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-089 GEFS 앙상블 피처 ###", flush=True)
    if args.stage == "signal":
        stage_signal()
    else:
        stage_gate([t.strip() for t in args.variants.split(",") if t.strip()],
                   args.seeds, args.jobs)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
