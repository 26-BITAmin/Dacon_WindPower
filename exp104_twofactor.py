# ═══════════════════════════════════════════════════════
# exp-104: 2-요인 분해 파이프라인 — 발전량 = 잠재발전량(날씨) × 가동률
#
# 사용자 제안: "곁다리로 쓰지 말고 차라리 가동률을 회귀해야 하는 것 아닌가."
#
# 지금까지 eff는 전부 **y를 직접 회귀하는 구조를 유지한 채** 곁다리로만 썼다:
#   exp-110 학습 가중 / exp-111B 피처 / exp-111C 사후 배율 — 셋 다 기각·중립.
# 여기서는 **구조 자체를 쪼갠다**:
#     ŷ = ŷ_potential(날씨) × êff(날씨)
#
# exp-053과의 차이 (그쪽은 절반만 했다):
#   - exp-053: y' = min(y/avail, cap)로 학습만 하고 **예측 시 되돌려 곱하지 않음**
#     → 구조적 과대예측(저가용 구간 signed +0.0697 → +0.0805), 복원 강도에 단조 악화.
#   - 여기: **êff를 예측해 곱해서 되돌린다** + 가동률 지표가 `avail_strict`(g3 오라벨,
#     EDA 12절)가 아니라 **`eff`**(유효출력률)이고, exp-111A에서 그룹 내부 AUC 0.758~0.798로
#     실제로 예측된다는 것이 확인됐다.
#
# 규약:
#   - `eff`는 ws≥5 슬롯에서만 정의된다. 그 밖(저풍속)은 "측정 가능한 열화 없음"으로 보고
#     **eff = 1.0** 으로 둔다 — 어차피 대부분 실측 <10%cap이라 채점 제외 구간이다.
#   - 잠재 타깃 y' = min(y / max(eff, 0.4), cap). floor 0.4는 복원 배율 2.5배 상한(exp-053 선례).
#   - 잠재 모델은 라벨에서 가용성을 이미 제거했으므로 **cln03 가용성 가중을 쓰지 않는다**
#     (저발전 다운웨이트만 유지). 이중 차감 방지.
#   - eff 예측은 exp-111의 규약 그대로 — 학습창은 시간블록 4겹 교차적합, 검증창은 전체 적합.
#
# 셀:
#   pot_x_eff : ŷ_potential × êff        ← 본 제안
#   pot_only  : ŷ_potential × 1.0        ← **대조**. 되돌림 단계의 기여를 분리 (exp-053 재현)
#   각각 DL 블렌드 w {0.65, 0.70} + GBM 단독.
#
# 판정 Δ = 보드 챔피언(hpo×dC_w65). 주기준 CV 4-fold (홀드아웃·corr은 참고).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp100_twofactor.py
# ═══════════════════════════════════════════════════════
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, make_sample_weight, prepare_matrices
from exp035_raw_dl import FOLD_LIST, CACHE_DIR
from exp049_clean import KEY, PP_NEW
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp056_hpo import RESULT_JSON as E56_JSON
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp065_gbm3 import build_train_frame, slice_fold
from exp110_effweight import EFF_PARQUET
from exp111_effpred import fill_effhat

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EFF_FLOOR = 0.40


def add_frame():
    train_df, cols = build_train_frame()
    eff = pd.read_parquet(EFF_PARQUET)[["forecast_kst_dtm", "group", "eff"]] \
        .rename(columns={"group": "kpx_group"})
    train_df = train_df.merge(eff, on=["forecast_kst_dtm", "kpx_group"], how="left")
    # ws≥5 밖은 "측정 가능한 열화 없음" 규약 → 1.0
    train_df["eff"] = train_df["eff"].fillna(1.0)
    return train_df, cols


def train_potential_fold(tr_all, va_all, cols, seeds, jobs, e56):
    """타깃을 y' = min(y/max(eff,floor), cap) 로 바꾼 GBM. 가용성 가중 미적용."""
    import lightgbm as lgb
    import xgboost as xgbm
    rows, clipped, n = [], 0, 0
    for group in [1, 2, 3]:
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
        va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
        if len(tr_g) == 0 or len(va_g) == 0:
            continue
        X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
        y = tr_g["target"].to_numpy()
        e = np.clip(tr_g["eff"].to_numpy(), EFF_FLOOR, None)
        y_pot = np.minimum(y / e, cap)
        clipped += int((y / e > cap).sum()); n += len(y)
        w_tr = make_sample_weight(y, cap, low_weight=0.05)      # 가용성 가중 없음 (이중 차감 방지)
        pl = np.mean([lgb.LGBMRegressor(n_estimators=e56["lgb_n"], objective="mae",
                      random_state=s, n_jobs=jobs, verbose=-1, **e56["lgb"])
                      .fit(X_tr, y_pot, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        px = np.mean([xgbm.XGBRegressor(n_estimators=e56["xgb_n"], objective="reg:absoluteerror",
                      tree_method="hist", random_state=s, n_jobs=jobs, verbosity=0, **e56["xgb"])
                      .fit(X_tr, y_pot, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        rows.append(pd.DataFrame({"group": group,
                                  "forecast_kst_dtm": va_g["forecast_kst_dtm"].to_numpy(),
                                  "pot": (0.5 * pl + 0.5 * px) / cap,
                                  "ehat": va_g["eff_hat"].to_numpy()}))
    return pd.concat(rows, ignore_index=True), clipped, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-104 2-요인 분해 (잠재발전량 × 가동률) ###", flush=True)
    train_df, cols = add_frame()
    e56 = json.loads(E56_JSON.read_text())
    cache, champ_bias = load_cache_full()

    need = [c for c in ("cf_pot", "cf_ehat") if c not in
            merge_cached_cols(cache, CACHE_DIR / "exp100_gbm_pot.parquet", ["cf_pot", "cf_ehat"])]
    if need:
        out = {"cf_pot": np.full(len(cache), np.nan), "cf_ehat": np.full(len(cache), np.nan)}
        ppath = CACHE_DIR / "exp100_gbm_pot_partial.parquet"
        if ppath.exists():
            prev = pd.read_parquet(ppath)
            mm = cache[KEY].merge(prev, on=KEY, how="left")
            for c in out:
                out[c] = mm[c].to_numpy()
        tot_c = tot_n = 0
        for fname, fold in FOLD_LIST:
            sel = (cache["fold"] == fname).to_numpy()
            if not np.isnan(out["cf_pot"][sel]).any():
                continue
            tf = time.time()
            tr_all, va_all = slice_fold(train_df, fold)
            tr_all, va_all = fill_effhat(tr_all, va_all, cols, args.jobs)
            p, c, n = train_potential_fold(tr_all, va_all, cols, args.seeds, args.jobs, e56)
            tot_c += c; tot_n += n
            m = cache.loc[sel, ["group", "forecast_kst_dtm"]].merge(
                p, on=["group", "forecast_kst_dtm"], how="left")
            assert m["pot"].notna().all(), f"정렬 실패 {fname}"
            out["cf_pot"][sel] = m["pot"].to_numpy()
            out["cf_ehat"][sel] = m["ehat"].to_numpy()
            for c_ in out:
                cache[c_] = out[c_]
            cache[KEY + ["cf_pot", "cf_ehat"]].to_parquet(ppath, index=False)
            print(f"  [{fname}] {time.time()-tf:.0f}s | 잠재 평균 {np.nanmean(m['pot']):.3f} "
                  f"| êff 평균 {np.nanmean(m['ehat']):.3f}", flush=True)
        for c_ in out:
            cache[c_] = out[c_]
        save_cols(cache, CACHE_DIR / "exp100_gbm_pot.parquet", ["cf_pot", "cf_ehat"])
        print(f"  라벨 복원 시 cap 초과 clip: {tot_c}/{tot_n} ({tot_c/max(tot_n,1)*100:.1f}%)",
              flush=True)

    cache["cf_potxeff"] = cache["cf_pot"] * cache["cf_ehat"]
    mm = cache["target"].notna().to_numpy()
    for c in ("cf_pot", "cf_potxeff"):
        v, h = cache[c].to_numpy(), cache["cf_hpo"].to_numpy()
        print(f"  corr({c}, cf_hpo) = {np.corrcoef(v[mm], h[mm])[0,1]:.4f} (참고) | "
              f"평균 이동 {(v-h).mean()*100:+.3f}%cap", flush=True)

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언 재현: CV {ref['cv'][0]:.4f} | HOLD {ref['hold'][0]:.4f}  (기대 0.6556/0.6535)",
          flush=True)
    res = {"hpo_solo": ev("cf_hpo", None, 1.0, PP_NEW)}
    for tag, col in (("potxeff", "cf_potxeff"), ("[대조]pot", "cf_pot")):
        res[f"{tag}_solo"] = ev(col, None, 1.0, PP_NEW)
        for w in (0.65, 0.70):
            res[f"{tag}_w{int(w*100)}"] = ev(col, "dl_cln03", w, PP_NEW)
    print_table(res, ref)
    print(f"\nGBM 단독 ΔCV: potxeff {res['potxeff_solo']['cv'][0]-res['hpo_solo']['cv'][0]:+.4f} "
          f"| [대조]pot {res['[대조]pot_solo']['cv'][0]-res['hpo_solo']['cv'][0]:+.4f}")
    print("\n※ [대조]pot 은 되돌림 없이 잠재발전량만 쓴 것 = exp-053 구조 재현. "
          "potxeff와의 차이가 **곱해서 되돌리는 단계의 기여**다.", flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
