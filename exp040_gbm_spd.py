# ═══════════════════════════════════════════════════════
# exp-040: 격자 편차(spd)를 **GBM base**에 투입 — 0.70 지분 쪽 정보 주입
#
# 배경: exp-035에서 spd(격자별 값−공간평균 편차 216개)가 DL 입력으로 LB +0.00508(세션 최대)을
#   냈다. 그런데 그 정보는 **DL 지분 0.30에만** 들어가 있다. GBM(지분 0.70)은 여전히
#   LDAPS 16격자 중 1개·GFS 9격자 중 1개만 보는 132피처다.
#   exp-019에서 GBM에 다격자를 넣어 실패했으나, 그때 표현은 **레벨값**(상호 corr 0.95+)이라
#   경사 성분이 분리돼 있지 않았다 — exp-035가 명시한 실패 원인. **편차 표현은 GBM에 미투입.**
#   exp-032의 "base 트랙 소진"도 132피처 조건부 결론이고, exp-035가 그 한정을 이미 실증했다.
#
# 변경점 분리: 후처리 장치(정격 분류기 pA·스냅·clip·α)는 챔피언 것을 그대로 재사용하고
#   **GBM base 예측만** 교체한다. 편향보정 맵만 두 방식 병기 —
#     champ = 챔피언(132) 맵 재사용 / refit = 확장 피처로 재추정 (맵 불일치 교란 제거)
#
# 변형:
#   g132  : 챔피언 (exp-035 GBM 캐시 재사용 — 결정적이라 재학습 불요, CV 0.6408/HOLD 0.6391)
#   gspd  : 132 + spd 편차 216 = 348
#   gspdG : 132 + GFS 편차 80 = 212   (exp-035에서 이득 대부분이 GFS 광역이었음)
#
# 평가: GBM 단독(w=1.0) + DL 블렌드. DL 예측은 exp-039 캐시(dl_hub = exp-035 spd DL) 재사용.
#   GBM도 spd를 보면 DL의 고유 기여가 줄어 최적 w가 올라갈 수 있으므로 w ∈ {0.80~0.60} 확장.
# 판정: 이중 게이트(4-fold CV + 1년 홀드아웃), 동일 캐시 런 Δ만 유효.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp040_gbm_spd.py --variants gspd,gspdG
#       C:\miniconda3\python.exe -X utf8 exp040_gbm_spd.py --variants gspd --quick
# ═══════════════════════════════════════════════════════
import argparse
import pickle
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import (CAPACITY_KWH, make_sample_weight, lgbm_params, prepare_matrices)
from exp009_rated_snap import estimate_bias_data, bias_map_from_data, COARSE_BINS
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import (build_inputs, load_gbm_cache, postprocess, score_from_tidy,
                           FOLD_LIST, CACHE_DIR)
from exp039_dl_diverse import DL_PRED_CACHE, build_slices

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

GBM_CACHE = CACHE_DIR / "exp040_gbm_ext.parquet"
BIAS_CACHE = CACHE_DIR / "exp040_bias_maps.pkl"

# 변형 → build_inputs가 만드는 colset 키
GBM_VARIANTS = {"gspd": "spd", "gspdG": "spdG", "gspdL": "spdL"}
BLEND_WS = [0.80, 0.75, 0.70, 0.65, 0.60]
DL_COL = "dl_hub"          # exp-039 캐시의 exp-035 spd DL 재현분


def train_gbm_fold(train_df, cols, fold, seeds, jobs):
    """fold별 그룹 GBM base(LGBM/XGB 50:50) 예측 + 확장 피처로 재추정한 편향맵."""
    import lightgbm as lgb
    import xgboost as xgbm

    tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
    va_all = train_df.loc[(train_df["forecast_kst_dtm"] >= fold["valid_start"]) &
                          (train_df["forecast_kst_dtm"] <= fold["valid_end"])]
    bias_data = estimate_bias_data(train_df, cols, fold["train_end"], 500, jobs)
    bias_map = bias_map_from_data(bias_data, {g: COARSE_BINS for g in bias_data})

    rows = []
    for group in [1, 2, 3]:
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
        va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
        if len(tr_g) == 0 or len(va_g) == 0:
            continue
        X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
        y_tr = tr_g["target"]
        w_tr = make_sample_weight(y_tr.to_numpy(), cap, low_weight=0.05)
        pl = np.mean([lgb.LGBMRegressor(**lgbm_params(500, jobs, s))
                      .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        px = np.mean([xgbm.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                      min_child_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=s,
                      n_jobs=jobs, verbosity=0, objective="reg:absoluteerror")
                      .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        rows.append(pd.DataFrame({"fold": fold["name"], "group": group,
                                  "forecast_kst_dtm": va_g["forecast_kst_dtm"].to_numpy(),
                                  "gbm_cf": (0.5 * pl + 0.5 * px) / cap}))
    return pd.concat(rows, ignore_index=True), bias_map


def run(args):
    train_df, cols, colsets = build_inputs(["spd"])
    champ, champ_bias = load_gbm_cache()          # gbm_cf(132)·pA·편향맵 = 챔피언 장치
    folds = FOLD_LIST if not args.quick else [FOLD_LIST[3], FOLD_LIST[4]]
    fold_names = [fn for fn, _ in folds]
    cache = champ[champ["fold"].isin(fold_names)].reset_index(drop=True)
    key = ["fold", "group", "forecast_kst_dtm"]
    cache = cache.rename(columns={"gbm_cf": "cf_g132"})

    # DL 예측 (exp-039 캐시) — 없으면 GBM 단독 평가만
    have_dl = False
    if DL_PRED_CACHE.exists():
        dl = pd.read_parquet(DL_PRED_CACHE)
        if DL_COL in dl.columns:
            m = cache[key].merge(dl[key + [DL_COL]], on=key, how="left")
            if m[DL_COL].notna().all():
                cache[DL_COL] = m[DL_COL].to_numpy(); have_dl = True
    print(f"입력: g132={len(cols)} / " + " / ".join(f"{v}={len(colsets[GBM_VARIANTS[v]])}" for v in args.variants)
          + f" | DL 캐시 {'있음' if have_dl else '없음(GBM 단독만 평가)'}", flush=True)

    # ── 확장 GBM 학습 (변형별 parquet 캐시) ──
    ext_bias = {}
    if GBM_CACHE.exists() and BIAS_CACHE.exists() and not args.refresh and not args.quick:
        prev = pd.read_parquet(GBM_CACHE)
        with open(BIAS_CACHE, "rb") as f:
            ext_bias = pickle.load(f)
        for c in [c for c in prev.columns if c.startswith("cf_")]:
            m = cache[key].merge(prev[key + [c]], on=key, how="left")
            if m[c].notna().all():
                cache[c] = m[c].to_numpy()
        print(f"확장 GBM 캐시 로드: {[c for c in cache.columns if c.startswith('cf_')]}", flush=True)

    for v in args.variants:
        if f"cf_{v}" in cache.columns:
            continue
        vcols = colsets[GBM_VARIANTS[v]]
        preds, bmaps = [], {}
        for fname, fold in folds:
            tf = time.time()
            p, bm = train_gbm_fold(train_df, vcols, fold, args.gbm_seeds, args.jobs)
            preds.append(p); bmaps[fname] = bm
            print(f"  [gbm:{v}:{fname}] {time.time()-tf:.0f}s", flush=True)
        p = pd.concat(preds, ignore_index=True).rename(columns={"gbm_cf": f"cf_{v}"})
        m = cache[key].merge(p, on=key, how="left")
        assert m[f"cf_{v}"].notna().all(), f"{v} 정렬 실패"
        cache[f"cf_{v}"] = m[f"cf_{v}"].to_numpy()
        ext_bias[v] = bmaps
        if not args.quick:
            cache[key + [c for c in cache.columns if c.startswith("cf_") and c != "cf_g132"]] \
                .to_parquet(GBM_CACHE, index=False)
            with open(BIAS_CACHE, "wb") as f:
                pickle.dump(ext_bias, f)

    # ── 평가 ──
    cv_folds = [fn for fn in fold_names if fn != HOLD_FOLD["name"]]
    slices = build_slices(cache)
    hold = HOLD_FOLD["name"]

    def score(cf_col, w, bias_mode):
        parts = []
        for (fname, group), sub in slices:
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            gbm = sub[cf_col].to_numpy()
            cf = gbm if w >= 1.0 else w * gbm + (1 - w) * sub[DL_COL].to_numpy()
            bmap = champ_bias[fname] if bias_mode == "champ" else ext_bias[cf_col[3:]][fname]
            pred = postprocess(cf, cap, sub["ws"].to_numpy(), sub["pA"].to_numpy(), group, bmap)
            parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
        tidy = pd.concat(parts, ignore_index=True)
        return dict(cv=score_from_tidy(tidy[tidy["fold"] != hold], "pred"),
                    hold=score_from_tidy(tidy[tidy["fold"] == hold], "pred"),
                    folds={fn: score_from_tidy(tidy[tidy["fold"] == fn], "pred")[0] for fn in cv_folds})

    res = {"g132_w100": score("cf_g132", 1.0, "champ")}
    if have_dl:
        res["g132_w70"] = score("cf_g132", 0.70, "champ")      # = 현 LB 챔피언 구성(spd_w70)
    for v in args.variants:
        for bm in (["champ", "refit"] if not args.quick else ["refit"]):
            tag = "" if bm == "refit" else "·cB"
            res[f"{v}_w100{tag}"] = score(f"cf_{v}", 1.0, bm)
            if have_dl:
                for w in BLEND_WS:
                    res[f"{v}_w{int(w*100):02d}{tag}"] = score(f"cf_{v}", w, bm)

    # ── 리포트 ──
    b_cv, b_hd = res["g132_w100"]["cv"][0], res["g132_w100"]["hold"][0]
    print("\n" + "=" * 96)
    print(f"기준A gbm132 단독: CV {b_cv:.4f} (1-NMAE {res['g132_w100']['cv'][1]:.4f} "
          f"FiCR {res['g132_w100']['cv'][2]:.4f}) | HOLD {b_hd:.4f}   ※ exp-035 로그 0.6408/0.6391")
    if have_dl:
        c_cv, c_hd = res["g132_w70"]["cv"][0], res["g132_w70"]["hold"][0]
        print(f"기준B 챔피언(gbm132+DLspd w0.70): CV {c_cv:.4f} | HOLD {c_hd:.4f}  "
              f"(Δgbm {c_cv-b_cv:+.4f}/{c_hd-b_hd:+.4f} — 로그 +0.0040/+0.0042와 일치해야 재현 OK)")
    else:
        c_cv = c_hd = None
    print("=" * 96)
    print(f"{'변형':<18}{'CV':>8}{'HOLD':>8}{'1-NMAE':>9}{'FiCR':>8}{'ΔA_cv':>9}{'ΔA_hold':>9}"
          + (f"{'ΔB_cv':>9}{'ΔB_hold':>9}  게이트B" if c_cv is not None else ""))
    for name, r in res.items():
        line = (f"{name:<18}{r['cv'][0]:>8.4f}{r['hold'][0]:>8.4f}{r['cv'][1]:>9.4f}{r['cv'][2]:>8.4f}"
                f"{r['cv'][0]-b_cv:>+9.4f}{r['hold'][0]-b_hd:>+9.4f}")
        if c_cv is not None:
            dc, dh = r["cv"][0] - c_cv, r["hold"][0] - c_hd
            line += f"{dc:>+9.4f}{dh:>+9.4f}  " + ("통과" if (dc > 0 and dh > 0) else ("  ·" if max(dc, dh) > 0 else ""))
        print(line)

    # GBM 단독끼리의 오차 상관·MAE (spd 정보가 GBM 오차를 바꾸는지)
    caps = cache["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    acf = cache["target"].to_numpy() / caps
    ev = cache["target"].to_numpy() >= caps * 0.10
    print("\n--- GBM 오차 진단 (평가대상 시간, 보정 전 원시 CF) ---")
    rows = []
    e132 = cache["cf_g132"].to_numpy() - acf
    for v in ["g132"] + list(args.variants):
        e = cache[f"cf_{v}"].to_numpy() - acf
        row = {"gbm": v, "mae": float(np.abs(e[ev]).mean()), "signed": float(e[ev].mean()),
               "corr_vs_g132": float(np.corrcoef(e132[ev], e[ev])[0, 1])}
        if have_dl:
            ed = cache[DL_COL].to_numpy() - acf
            row["corr_vs_DL"] = float(np.corrcoef(e[ev], ed[ev])[0, 1])
        rows.append(row)
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    if c_cv is not None:
        print("\n--- fold별 ΔCV (기준B 챔피언 대비) ---")
        print(f"{'변형':<18}" + "".join(f"{fn.replace('fold_',''):>10}" for fn in cv_folds))
        for name, r in res.items():
            if name == "g132_w70":
                continue
            print(f"{name:<18}" + "".join(
                f"{r['folds'][fn]-res['g132_w70']['folds'][fn]:>+10.4f}" for fn in cv_folds))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=str, default="gspd,gspdG")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--gbm-seeds", type=int, default=3)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--quick", action="store_true", help="Q4+홀드아웃·1시드 배관/시간 점검")
    args = ap.parse_args()
    args.variants = [v for v in args.variants.split(",") if v]
    unknown = [v for v in args.variants if v not in GBM_VARIANTS]
    assert not unknown, f"미정의 변형: {unknown}"
    if args.quick:
        args.gbm_seeds = 1
        print("### QUICK 배관/시간 점검 모드 (판정용 아님) ###")
    t0 = time.time()
    print(f"### exp-040 (변형={args.variants}, gbm_seeds={args.gbm_seeds}) ###", flush=True)
    run(args)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
