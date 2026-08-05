# ═══════════════════════════════════════════════════════
# AN-19: GBM vs BIV-DRN 정면 대결 — 비대칭 2개 제거
#
# AN-17/18 비교의 문제: GBM은 352피처·전체 행, DRN은 21피처·enfo 행.
# 이번: 동일 행(enfo 커버)·피처 교차 4셀 + 동일 지표
#   gbm352 / gbm21 : LGBM u,v 성분별 (그룹별 학습) — 점예보
#   drn21 / drn352 : BIV-DRN (그룹 임베딩 공동 학습) — 분포
# ES는 점예보도 퇴화분포로 채점 (proper — 산포 없음이 벌점됨).
#
# 실행(루트에서): C:\miniconda3\python.exe -X utf8 "open (1)/testing/an19_fair2.py"
# ═══════════════════════════════════════════════════════
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import pandas as pd

from an10_paper import crps_ens
from an12_emos import build_scada_vec, energy_score
from an18_bivpp2026 import build_features, fit_drn, nll_np
from exp035_raw_dl import FOLD_LIST
from exp056_hpo import RESULT_JSON as E56_JSON
from exp065_gbm3 import build_train_frame, slice_fold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

N_SAMP = 32
RNG = np.random.default_rng(0)


def main():
    t0 = time.time()
    print("### AN-19 GBM vs DRN 정면 대결 (동일 행 · 피처 교차) ###", flush=True)
    train_df, use_cols = build_train_frame()
    vec = build_scada_vec()
    for g in [1, 2, 3]:
        train_df = train_df.merge(
            vec[g].assign(kpx_group=g), on=["forecast_kst_dtm", "kpx_group"],
            how="left", suffixes=("", "_d"))
        for c in ["park_ws", "u_obs", "v_obs"]:
            if f"{c}_d" in train_df:
                train_df[c] = train_df[c].fillna(train_df.pop(f"{c}_d"))
    df, feats21 = build_features(train_df)
    feats352 = [c for c in use_cols if c in df.columns]
    print(f"프레임 {df.shape} | 21피처 {len(feats21)} · 352피처 {len(feats352)} "
          f"— {time.time()-t0:.0f}s", flush=True)
    e56 = json.loads(E56_JSON.read_text())
    import lightgbm as lgb
    folds = dict(FOLD_LIST)

    recs = []
    for fname in ["fold_2024Q1", "fold_2024Q2", "fold_2024Q3", "fold_2024Q4"]:
        tt = time.time()
        tr_all, va_all = slice_fold(df, folds[fname])
        tr = tr_all[tr_all["u_obs"].notna()].sort_values(
            ["forecast_kst_dtm", "kpx_group"]).reset_index(drop=True)
        va = va_all.sort_values(["forecast_kst_dtm", "kpx_group"]).reset_index(drop=True)
        vam = va["u_obs"].notna().to_numpy()
        y_vec = va[["u_obs", "v_obs"]].to_numpy()[vam]
        y_ws = va["park_ws"].to_numpy()[vam]
        gtr = tr["kpx_group"].to_numpy() - 1
        gva = va["kpx_group"].to_numpy() - 1
        Ytr = tr[["u_obs", "v_obs"]].to_numpy(dtype=np.float32)

        def ang(P):
            return np.abs((np.rad2deg(np.arctan2(P[:, 1], P[:, 0])
                           - np.arctan2(y_vec[:, 1], y_vec[:, 0])) + 180) % 360
                          - 180).mean()

        # ── GBM 점벡터 (그룹별, 피처 2종)
        for ftag, cols in (("gbm352", feats352), ("gbm21", feats21)):
            P = np.full((len(va), 2), np.nan)
            for g in [1, 2, 3]:
                tm, vm = gtr == g - 1, gva == g - 1
                for j, comp in enumerate(["u_obs", "v_obs"]):
                    mdl = lgb.LGBMRegressor(n_estimators=e56["lgb_n"],
                                            objective="mae", random_state=0,
                                            n_jobs=8, verbose=-1, **e56["lgb"])
                    mdl.fit(tr.loc[tm, cols], tr.loc[tm, comp])
                    P[vm, j] = mdl.predict(va.loc[vm, cols])
            Pm = P[vam]
            recs.append(dict(fold=fname, model=ftag,
                             ws_mae=np.abs(np.linalg.norm(Pm, 1) - y_ws).mean()
                             if False else
                             np.abs(np.linalg.norm(Pm, axis=1) - y_ws).mean(),
                             ws_crps=np.abs(np.linalg.norm(Pm, axis=1) - y_ws).mean(),
                             dir_mae=ang(Pm),
                             vec_rmse=np.sqrt(((Pm - y_vec) ** 2).sum(1).mean()),
                             es=np.linalg.norm(Pm - y_vec, axis=1).mean(),
                             logs=np.nan))

        # ── DRN (피처 2종)
        for ftag, cols in (("drn21", feats21), ("drn352", feats352)):
            mu_ = tr[cols].mean()
            sd_ = tr[cols].std().replace(0, 1)
            Xtr = np.nan_to_num(((tr[cols] - mu_) / sd_).to_numpy(dtype=np.float32))
            Xva = np.nan_to_num(((va[cols] - mu_) / sd_).to_numpy(dtype=np.float32))
            mu, sig, rho = fit_drn(Xtr, gtr, Ytr, Xva, gva)
            z1 = RNG.standard_normal((vam.sum(), N_SAMP))
            z2 = RNG.standard_normal((vam.sum(), N_SAMP))
            r_ = rho[vam][:, None]
            su = mu[vam][:, None, 0] + sig[vam][:, None, 0] * z1
            sv = mu[vam][:, None, 1] + sig[vam][:, None, 1] * (
                r_ * z1 + np.sqrt(1 - r_ ** 2) * z2)
            samp = np.stack([su, sv], -1)
            spd = np.linalg.norm(samp, axis=2)
            recs.append(dict(fold=fname, model=ftag,
                             ws_mae=np.abs(spd.mean(1) - y_ws).mean(),
                             ws_crps=crps_ens(spd, y_ws).mean(),
                             dir_mae=ang(mu[vam]),
                             vec_rmse=np.sqrt(((mu[vam] - y_vec) ** 2).sum(1).mean()),
                             es=energy_score(samp, y_vec).mean(),
                             logs=nll_np(y_vec, mu[vam], sig[vam], rho[vam]).mean()))
        print(f"  [{fname}] {time.time()-tt:.0f}s", flush=True)

    out = pd.DataFrame(recs)
    agg = out.groupby("model")[["ws_mae", "ws_crps", "dir_mae", "vec_rmse", "es",
                                "logs"]].mean().sort_values("es")
    print("\n=== 4폴드 평균 (동일 행 = enfo 커버 ∩ 나셀 존재) ===", flush=True)
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)
    print("\n주: 점예보(gbm*)의 es = 퇴화분포 ES = 평균 벡터오차 노름 · "
          "ws_crps = 점예보 CRPS = MAE. 분포(drn*)는 산포 덕에 같은 μ라도 ES↓ 가능",
          flush=True)
    out.to_csv(ROOT / "open (1)" / "testing" / "an19_results.csv", index=False,
               encoding="utf-8-sig")
    print(f"\n상세 → testing/an19_results.csv | 총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
