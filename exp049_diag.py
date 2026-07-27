# ═══════════════════════════════════════════════════════
# exp-049 진단: SCADA 터빈 가용성 ↔ 학습 라벨·모델 오차 (읽기 전용, 모델 학습 0)
#
# 가설: 학습 라벨에는 터빈 일부가 꺼진(정비·고장) 시간이 섞여 있고, 모델은 그 시간의
#   낮은 라벨을 "그 날씨의 정상 발전량"으로 학습 → 전 터빈 가동 시간에 체계적 과소예측.
#   exp-030의 "CF 0.7~0.9 71% 과소예측 = FiCR 손실 33%"의 후보 원인.
#   검증 포인트(스모킹 건): signed error(pred−actual)가 가용률과 **음의 상관**이어야 함
#   (저가용 시간 → 실제가 낮아 과대예측 / 만가용 시간 → 과소예측).
#
# 터빈→그룹 (info.xlsx): 그룹1 = VESTAS 1~6 / 그룹2 = VESTAS 7~12 / 그룹3 = UNISON 1~5
#   그룹 내 터빈 용량 동일 → 가용률 = 작동 터빈 수 / 전체 수.
#
# 가용성 정의 (시간 h, 터빈 i): 10분 슬롯 6개 중 5개 이상에서 power ≤ 1kW(또는 NaN)
#   인데 바람은 있음(자기 ws ≥ 5m/s, ws NaN이면 그룹 중앙값 ws로 대체) → 비가용.
#   저풍속(ws<5) 무발전은 정상으로 취급. strict = NaN도 비가용 / lenient = 명시적 0만.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp049_diag.py
# ═══════════════════════════════════════════════════════
import sys

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, DATA_DIR
from exp035_raw_dl import load_gbm_cache
from exp039_dl_diverse import DL_PRED_CACHE as E39_CACHE
from exp040_gbm_spd import GBM_CACHE as EXT_GBM_CACHE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

KEY = ["fold", "group", "forecast_kst_dtm"]
GROUP_TURBINES = {
    1: [("vestas", i) for i in range(1, 7)],
    2: [("vestas", i) for i in range(7, 13)],
    3: [("unison", i) for i in range(1, 6)],
}


def load_scada():
    v = pd.read_csv(DATA_DIR / "train" / "scada_vestas_train.csv", encoding="utf-8-sig",
                    parse_dates=["kst_dtm"])
    u = pd.read_csv(DATA_DIR / "train" / "scada_unison_train.csv", encoding="utf-8-sig",
                    parse_dates=["kst_dtm"])
    return v.merge(u, on="kst_dtm", how="outer").sort_values("kst_dtm").reset_index(drop=True)


def availability(sc):
    """시간별 그룹 가용률 (strict/lenient) + 그룹 SCADA 평균 풍속."""
    sc = sc.copy()
    sc["hour"] = sc["kst_dtm"].dt.floor("h")
    out = {"hour": None}
    rows = []
    for g, turbines in GROUP_TURBINES.items():
        p_cols = [f"{mk}_wtg{i:02d}_power_kw10m" for mk, i in turbines]
        w_cols = [f"{mk}_wtg{i:02d}_ws" for mk, i in turbines]
        grp = sc[["hour"] + p_cols + w_cols]
        # 그룹 중앙값 ws (터빈 자기 ws 결측 대체용)
        med_ws = grp[w_cols].median(axis=1)
        agg = {"g_ws": grp.assign(m=med_ws).groupby("hour")["m"].mean()}
        for (mk, i), pc, wc in zip(turbines, p_cols, w_cols):
            ws_eff = grp[wc].fillna(med_ws)
            windy = ws_eff >= 5.0
            prod = grp[pc] > 1.0
            dead_strict = (~prod.fillna(False)) & windy          # NaN power도 비가용
            dead_lenient = (grp[pc] <= 1.0) & windy               # 명시적 0만
            slots = grp.groupby("hour")[pc].size()
            agg[f"ds_{mk}{i}"] = dead_strict.groupby(grp["hour"]).sum()
            agg[f"dl_{mk}{i}"] = dead_lenient.fillna(False).groupby(grp["hour"]).sum()
            agg[f"n_{mk}{i}"] = slots
        A = pd.DataFrame(agg)
        n_t = len(turbines)
        # 터빈별: 슬롯 6개 중 5개 이상 dead → 그 시간 비가용
        un_s = sum((A[f"ds_{mk}{i}"] >= np.maximum(A[f"n_{mk}{i}"] - 1, 5)).astype(int)
                   for mk, i in turbines)
        un_l = sum((A[f"dl_{mk}{i}"] >= np.maximum(A[f"n_{mk}{i}"] - 1, 5)).astype(int)
                   for mk, i in turbines)
        rows.append(pd.DataFrame({
            "forecast_kst_dtm": A.index, "group": g,
            "avail_strict": 1 - un_s / n_t, "avail_lenient": 1 - un_l / n_t,
            "scada_ws": A["g_ws"].to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True)


def main():
    print("### exp-049 진단: SCADA 가용성 ↔ 라벨·모델 오차 ###", flush=True)
    sc = load_scada()
    print(f"SCADA 결합: {sc.shape}, 기간 {sc['kst_dtm'].min()} ~ {sc['kst_dtm'].max()}")
    av = availability(sc)
    av["forecast_kst_dtm"] = pd.to_datetime(av["forecast_kst_dtm"])

    # ── 1. 가용률 분포 ──
    print("\n--- 1. 가용률 분포 (바람 있는 시간 기준 비가용 판정, strict) ---")
    av["year"] = av["forecast_kst_dtm"].dt.year
    for g in [1, 2, 3]:
        a = av[av["group"] == g]
        print(f"그룹{g}: 시간수 {len(a)} | avail 평균 {a['avail_strict'].mean():.3f} | "
              f"<1.0 비율 {(a['avail_strict'] < 0.999).mean()*100:.1f}% | "
              f"<0.9 {(a['avail_strict'] < 0.9).mean()*100:.1f}% | "
              f"<0.7 {(a['avail_strict'] < 0.7).mean()*100:.1f}% | "
              f"(lenient <0.9: {(a['avail_lenient'] < 0.9).mean()*100:.1f}%)")
    print("연도별 avail 평균 (strict):")
    print(av.pivot_table(index="year", columns="group", values="avail_strict", aggfunc="mean").round(3).to_string())

    # ── 2. 라벨 검증 (모델 무관): 같은 풍속에서 저가용 시간의 CF가 실제로 낮은가 ──
    labels = pd.read_csv(DATA_DIR / "train" / "train_labels.csv", encoding="utf-8-sig",
                         parse_dates=["kst_dtm"]).rename(columns={"kst_dtm": "forecast_kst_dtm"})
    lab = labels.melt(id_vars="forecast_kst_dtm", var_name="gcol", value_name="target")
    lab["group"] = lab["gcol"].str.extract(r"(\d)").astype(int)
    lab = lab.merge(av, on=["forecast_kst_dtm", "group"], how="inner")
    lab["cap"] = lab["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{g}"])
    lab["cf"] = lab["target"] / lab["cap"]
    sub = lab[(lab["scada_ws"] >= 6) & (lab["scada_ws"] < 14) & lab["cf"].notna()].copy()
    sub["ws_bin"] = pd.cut(sub["scada_ws"], [6, 8, 10, 12, 14])
    print("\n--- 2. 같은 SCADA 풍속에서 가용률별 평균 CF (라벨 직접 검증) ---")
    pt = sub.pivot_table(index="ws_bin",
                         columns=pd.cut(sub["avail_strict"], [-0.01, 0.7, 0.9, 0.999, 1.01],
                                        labels=["<0.7", "0.7-0.9", "0.9-1", "1.0"]),
                         values="cf", aggfunc=["mean", "size"], observed=True)
    print(pt.round(3).to_string())

    # ── 3. 모델 오차와의 관계 (현 챔피언 블렌드, CV 캐시) ──
    cache, _ = load_gbm_cache()
    cache = cache.rename(columns={"gbm_cf": "cf_g132"})
    ext = pd.read_parquet(EXT_GBM_CACHE)
    m = cache[KEY].merge(ext[KEY + ["cf_gspd"]], on=KEY, how="left")
    cache["cf_gspd"] = m["cf_gspd"].to_numpy()
    dl = pd.read_parquet(E39_CACHE)
    m = cache[KEY].merge(dl[KEY + ["dl_q60"]], on=KEY, how="left")
    cache["dl_q60"] = m["dl_q60"].to_numpy()
    cache["blend"] = 0.65 * cache["cf_gspd"] + 0.35 * cache["dl_q60"]
    cache["cap"] = cache["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"])
    cache["acf"] = cache["target"] / cache["cap"]
    cache["forecast_kst_dtm"] = pd.to_datetime(cache["forecast_kst_dtm"])
    cache = cache.merge(av[["forecast_kst_dtm", "group", "avail_strict", "avail_lenient"]],
                        on=["forecast_kst_dtm", "group"], how="left")
    ev = cache[(cache["target"] >= cache["cap"] * 0.10) & cache["avail_strict"].notna()].copy()
    ev["err"] = ev["blend"] - ev["acf"]          # signed, CF 단위 (+ = 과대예측)
    print(f"\n--- 3. 모델 오차 vs 가용률 (평가대상 {len(ev)}시간, blend=0.65·gspd+0.35·q60, 보정 전) ---")
    print(f"corr(signed_err, avail_strict) = {np.corrcoef(ev['err'], ev['avail_strict'])[0,1]:+.4f}  (가설: 음수)")
    bins = pd.cut(ev["avail_strict"], [-0.01, 0.7, 0.9, 0.999, 1.01], labels=["<0.7", "0.7-0.9", "0.9-1", "1.0"])
    print(ev.groupby(bins, observed=True).agg(n=("err", "size"), signed=("err", "mean"),
                                              mae=("err", lambda x: x.abs().mean())).round(4).to_string())

    pocket = ev[(ev["acf"] >= 0.7) & (ev["acf"] < 0.9)]
    print(f"\n--- 4. CF 0.7~0.9 과소예측 포켓의 가용률 ---")
    print(f"포켓 시간수 {len(pocket)} | 과소예측 비율 {(pocket['err'] < 0).mean()*100:.0f}% | "
          f"포켓 평균 avail {pocket['avail_strict'].mean():.3f} vs 전체 {ev['avail_strict'].mean():.3f}")
    big = ev[ev["err"].abs() > 0.15]
    print(f"큰 오차(|err|>15%cap) {len(big)}시간 평균 avail {big['avail_strict'].mean():.3f} | "
          f"과대예측 큰오차 {len(big[big['err']>0.15])}시간 avail {big[big['err']>0.15]['avail_strict'].mean():.3f}")

    # 저장 (후속 개입 실험용)
    out = DATA_DIR / "external" / "exp049_avail.parquet"
    av.drop(columns=["year"]).to_parquet(out, index=False)
    print(f"\n가용률 저장 → {out.name}")


if __name__ == "__main__":
    main()
