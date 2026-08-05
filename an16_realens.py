# ═══════════════════════════════════════════════════════
# AN-16: 진짜 앙상블로 EMOS 재실행 — ECMWF enfo 16멤버 (u10, v10)
#
# 사용자 지적: 논문(Schuhen 2012)의 입력은 UWME 8멤버 "진짜 앙상블"인데
#   AN-12는 5개 이종 결정론 모델로 대체했다. → exp-073 디코드 자산에
#   enfo 멤버 원본(10u/10v × 16멤버 × 그룹, 2023-01~)이 있어 원형 재현 가능.
#
# 구성: 멤버별 3h 스텝 → 시간별 선형보간(밸리드 KST = init 12z + step + 21h),
#       AN-12의 fit_emos(uni/biv/rot) 그대로, 실측 = 나셀 벡터.
# 대조: 원시 enfo 앙상블 · (참조 수치) AN-12/13 5모델판.
# 주의: enfo는 2023-01-18부터 → Q1 폴드 학습창 ~11.5개월.
#
# 실행(프로젝트 루트에서): C:\miniconda3\python.exe -X utf8 "open (1)/testing/an16_realens.py"
# ═══════════════════════════════════════════════════════
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from an10_paper import crps_ens
from an12_emos import build_scada_vec, fit_emos, sample_biv, energy_score
from exp035_raw_dl import CACHE_DIR, FOLD_LIST
from exp065_gbm3 import build_train_frame, slice_fold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ENFO_DIR = ROOT / "open (1)" / "external" / "enfo_decoded"
MEM_CACHE = CACHE_DIR / "an16_enfo_members.parquet"
K_MEM = 16


def build_member_frame():
    """enfo 디코드 CSV → 시간별 멤버 u,v (그룹×시각×멤버). 캐시."""
    if MEM_CACHE.exists():
        return pd.read_parquet(MEM_CACHE)
    files = sorted(ENFO_DIR.glob("*.csv"))
    print(f"  enfo CSV {len(files)}개 로드...", flush=True)
    raw = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    w = raw.pivot_table(index=["init_date", "step", "group", "member"],
                        columns="param", values="val").reset_index()
    w["valid_kst"] = pd.to_datetime(w["init_date"], format="%Y%m%d") \
        + pd.to_timedelta(w["step"] + 21, unit="h")
    rows = []
    for (init, g, m), sub in w.groupby(["init_date", "group", "member"]):
        sub = sub.sort_values("step")
        hours = pd.date_range(sub["valid_kst"].iloc[0], sub["valid_kst"].iloc[-1],
                              freq="h")
        sec = sub["valid_kst"].astype("int64").to_numpy()
        hsec = hours.astype("int64").to_numpy()
        rows.append(pd.DataFrame({
            "forecast_kst_dtm": hours, "group": g, "member": m,
            "u": np.interp(hsec, sec, sub["10u"].to_numpy()),
            "v": np.interp(hsec, sec, sub["10v"].to_numpy())}))
    mem = pd.concat(rows, ignore_index=True)
    mem = mem.drop_duplicates(["forecast_kst_dtm", "group", "member"], keep="last")
    mem.to_parquet(MEM_CACHE, index=False)
    print(f"  멤버 프레임 {len(mem)}행 캐시 → {MEM_CACHE.name}", flush=True)
    return mem


def main():
    t0 = time.time()
    print("### AN-16 진짜 앙상블(enfo 16멤버) EMOS ###", flush=True)
    mem = build_member_frame()
    print(f"멤버 프레임 {len(mem)}행 | 기간 {mem['forecast_kst_dtm'].min()} ~ "
          f"{mem['forecast_kst_dtm'].max()} — {time.time()-t0:.0f}s", flush=True)

    train_df, _ = build_train_frame()
    vec = build_scada_vec()
    for g in [1, 2, 3]:
        train_df = train_df.merge(
            vec[g].assign(kpx_group=g), on=["forecast_kst_dtm", "kpx_group"],
            how="left", suffixes=("", "_d"))
        for c in ["park_ws", "u_obs", "v_obs"]:
            if f"{c}_d" in train_df:
                train_df[c] = train_df[c].fillna(train_df.pop(f"{c}_d"))

    uw = mem.pivot_table(index=["forecast_kst_dtm", "group"], columns="member",
                         values="u")
    vw = mem.pivot_table(index=["forecast_kst_dtm", "group"], columns="member",
                         values="v")
    print(f"멤버 wide {uw.shape} | 결측 {uw.isna().mean().mean()*100:.2f}%", flush=True)

    folds = dict(FOLD_LIST)
    recs = []
    for fname in ["fold_2024Q1", "fold_2024Q2", "fold_2024Q3", "fold_2024Q4"]:
        for g in [1, 2, 3]:
            tt = time.time()
            tr_all, va_all = slice_fold(train_df, folds[fname])
            tr_g = tr_all.loc[tr_all["kpx_group"] == g].sort_values(
                "forecast_kst_dtm").reset_index(drop=True)
            va_g = va_all.loc[va_all["kpx_group"] == g].sort_values(
                "forecast_kst_dtm").reset_index(drop=True)

            def mem_mat(df_g):
                idx = pd.MultiIndex.from_arrays(
                    [df_g["forecast_kst_dtm"], np.full(len(df_g), g)])
                U = uw.reindex(idx).to_numpy()
                V = vw.reindex(idx).to_numpy()
                return np.stack([U, V], -1)                 # [N, 16, 2]

            M_tr, M_va = mem_mat(tr_g), mem_mat(va_g)
            has_tr = ~np.isnan(M_tr).any(axis=(1, 2))
            has_va = ~np.isnan(M_va).any(axis=(1, 2))
            trm = tr_g["u_obs"].notna().to_numpy() & has_tr
            vam = va_g["u_obs"].notna().to_numpy() & has_va
            if trm.sum() < 1000 or vam.sum() < 100:
                print(f"  [{fname} g{g}] 표본 부족 (tr {trm.sum()} va {vam.sum()}) "
                      f"— 건너뜀", flush=True)
                continue
            mk = lambda M: dict(mu=M.mean(1), s2=M.var(1),
                                theta=np.arctan2(M.mean(1)[:, 1], M.mean(1)[:, 0]))
            tr = {**mk(M_tr[trm]), "y": np.c_[tr_g.loc[trm, "u_obs"],
                                              tr_g.loc[trm, "v_obs"]]}
            va_full = mk(M_va)
            y_vec = np.c_[va_g["u_obs"], va_g["v_obs"]][vam]
            y_ws = va_g["park_ws"].to_numpy()[vam]

            def add(model, **kw):
                recs.append(dict(fold=fname, group=g, model=model,
                                 n=int(vam.sum()), **kw))

            ang = lambda p, o: np.abs((np.rad2deg(np.arctan2(p[:, 1], p[:, 0])
                                       - np.arctan2(o[:, 1], o[:, 0])) + 180) % 360 - 180)
            raw_mu = va_full["mu"][vam]
            add("raw_enfo", ws_mae=np.abs(
                    np.linalg.norm(M_va[vam], axis=2).mean(1) - y_ws).mean(),
                dir_mae=ang(raw_mu, y_vec).mean(),
                vec_rmse=np.sqrt(((raw_mu - y_vec) ** 2).sum(1).mean()),
                es=energy_score(M_va[vam], y_vec).mean(),
                ws_crps=crps_ens(np.linalg.norm(M_va[vam], axis=2), y_ws).mean(),
                spread_skill=float(np.corrcoef(
                    M_va[vam].std(1).mean(1),
                    np.linalg.norm(raw_mu - y_vec, axis=1))[0, 1]))

            for mode in ["uni", "biv", "rot"]:
                mu_va, sig_va, r_va = fit_emos(tr, va_full, mode)
                samp = sample_biv(mu_va[vam], sig_va[vam], r_va[vam])
                spd = np.linalg.norm(samp, axis=2)
                add(f"emos_{mode}",
                    ws_mae=np.abs(spd.mean(1) - y_ws).mean(),
                    dir_mae=ang(mu_va[vam], y_vec).mean(),
                    vec_rmse=np.sqrt(((mu_va[vam] - y_vec) ** 2).sum(1).mean()),
                    es=energy_score(samp, y_vec).mean(),
                    ws_crps=crps_ens(spd, y_ws).mean())
            print(f"  [{fname} g{g}] tr {trm.sum()} va {vam.sum()} "
                  f"— {time.time()-tt:.0f}s", flush=True)

    df = pd.DataFrame(recs)
    cols = ["ws_mae", "ws_crps", "dir_mae", "vec_rmse", "es"]
    agg = df.groupby("model")[cols].mean().sort_values("es")
    print("\n=== 4폴드·3그룹 평균 — AN-12/13(5모델판) 참조: raw 3.07/28.6°/ES3.70 · "
          "emos_rot 1.88/23.1°/2.08 · premix_hyb 1.50/21.4°/1.73 ===", flush=True)
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)
    ss = df[df["model"] == "raw_enfo"]["spread_skill"].mean()
    print(f"\nenfo 원시 spread-skill 상관: {ss:.3f} (교환가능 앙상블의 실질 정보)",
          flush=True)
    df.to_csv(ROOT / "open (1)" / "testing" / "an16_results.csv", index=False,
              encoding="utf-8-sig")
    print(f"\n상세 → testing/an16_results.csv | 총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
