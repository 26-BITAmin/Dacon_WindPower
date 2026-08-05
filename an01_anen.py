# ═══════════════════════════════════════════════════════
# AN-01: Analog Ensemble (AnEn) 트랙 — NCAR RAL / Delle Monache et al. 2013 (MWR)
#
# 방법:
#   검증 시각 t의 예보 상태벡터(예측자 V개 × 시간창 ±1h)와 가장 유사한 과거 시각
#   t'들을 학습 기간에서 찾아, 그 시각들의 "실측 CF"를 앙상블 멤버로 삼는다.
#   유사도 (논문 식 그대로):
#     d(t,t') = Σ_i (w_i/σ_i) · sqrt( Σ_{j=-1..1} (F_{i,t+j} − A_{i,t'+j})² )
#   σ_i = 학습 기간 예보값 표준편차, w_i = 예측자 가중(기본 1, 풍향 sin/cos는 0.5씩).
#
# 논문 예측자(10m 풍속 예측): 10m 풍속·풍향, 2m 기온, 지표기압 → 우리 대응
#   P_PAPER = ldaps_ws_10m + ldaps_wd_(sin|cos)_10m + ldaps 2m t + ldaps sp
#   P_HUB   = 허브고도·다소스 확장 (gfs 100m ws/wd, ec_ws, jmsm/jgsm ws, shear, t, sp)
#
# 프로토콜: smoke = Q4 폴드 · 구성 여러 개 → 평가마스크 CF MAE·signed·corr(cf_hpo)
#           gate  = 4-fold 캐시 생성 → make_ev 사다리 (트랙 자체 판정 + 챔피언 대비 정보)
#
# 실행: C:\miniconda3\python.exe -X utf8 an01_anen.py --stage smoke
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp035_raw_dl import FOLD_LIST, CACHE_DIR
from exp049_clean import KEY, PP_NEW
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp065_gbm3 import build_train_frame, corr_on_eval, slice_fold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 예측자 세트: (컬럼, 가중치) — 풍향은 sin/cos 각 0.5로 "방향 1변수" 취급
P_PAPER = [("ldaps_ws_10m", 1.0), ("ldaps_wd_sin_10m", 0.5), ("ldaps_wd_cos_10m", 0.5),
           ("ldaps_heightAboveGround_2_t", 1.0), ("ldaps_surface_0_sp", 1.0)]
P_HUB = [("gfs_ws_100m", 1.0), ("gfs_wd_sin_100m", 0.5), ("gfs_wd_cos_100m", 0.5),
         ("ldaps_ws_10m", 1.0), ("ec_ws", 1.0), ("jmsm_ws", 1.0), ("jgsm_ws", 1.0),
         ("gfs_wind_shear_100_10", 0.5),
         ("ldaps_heightAboveGround_2_t", 0.5), ("ldaps_surface_0_sp", 0.5)]
P_WSONLY = [("gfs_ws_100m", 1.0), ("ldaps_ws_10m", 1.0), ("ec_ws", 1.0),
            ("jmsm_ws", 1.0), ("jgsm_ws", 1.0)]
PRED_SETS = {"paper": P_PAPER, "hub": P_HUB, "wsonly": P_WSONLY}

K_MAX = 300
TWIN = 1          # 시간창 반폭 (±1h)


def lagged_block(g_df, cols):
    """그룹 시계열(시간 정렬)에서 [N, V, 2*TWIN+1] 시간창 블록 생성. 결측 ffill/bfill."""
    arr = g_df[cols].ffill().bfill().to_numpy(dtype=np.float64)
    N, V = arr.shape
    out = np.empty((N, V, 2 * TWIN + 1), dtype=np.float64)
    for k, j in enumerate(range(-TWIN, TWIN + 1)):
        sh = np.roll(arr, -j, axis=0)
        if j < 0:
            sh[:(-j)] = arr[0]
        elif j > 0:
            sh[-j:] = arr[-1]
        out[:, :, k] = sh
    return out


def anen_dists(F_va, A_tr, sig, w, chunk=256):
    """Delle Monache 거리 행렬 [N_va, N_tr] (float32).
    F_va [Nv,V,L], A_tr [Nt,V,L], sig/w [V]"""
    Nv, V, L = F_va.shape
    Nt = A_tr.shape[0]
    D = np.empty((Nv, Nt), dtype=np.float32)
    coef = (w / np.maximum(sig, 1e-9)).astype(np.float64)
    for s in range(0, Nv, chunk):
        e = min(s + chunk, Nv)
        acc = np.zeros((e - s, Nt), dtype=np.float64)
        for i in range(V):
            sq = np.zeros((e - s, Nt), dtype=np.float64)
            for k in range(L):
                diff = F_va[s:e, i, k][:, None] - A_tr[None, :, i, k]
                sq += diff * diff
            acc += coef[i] * np.sqrt(sq)
        D[s:e] = acc.astype(np.float32)
    return D


def build_analogs(train_df, fold, group, pset, hour_win=None):
    """폴드×그룹: 상위 K_MAX 아날로그 (인덱스·거리·학습 실측 CF)."""
    cols = [c for c, _ in pset]
    w = np.array([wt for _, wt in pset])
    cap = CAPACITY_KWH[f"kpx_group_{group}"]
    tr_all, va_all = slice_fold(train_df, fold)
    tr_g = tr_all.loc[(tr_all["kpx_group"] == group)].sort_values(
        "forecast_kst_dtm").reset_index(drop=True)
    va_g = va_all.loc[va_all["kpx_group"] == group].sort_values(
        "forecast_kst_dtm").reset_index(drop=True)
    A = lagged_block(tr_g, cols)
    F = lagged_block(va_g, cols)
    ok = tr_g["target"].notna().to_numpy()          # 라벨 있는 행만 아날로그 후보
    sig = np.nanstd(tr_g[cols].to_numpy(dtype=np.float64), axis=0)
    D = anen_dists(F, A[ok], sig, w)
    y_cf = (tr_g.loc[ok, "target"].to_numpy() / cap)
    if hour_win is not None:                        # 시각대 제약 (논문의 search window)
        h_tr = tr_g.loc[ok, "forecast_kst_dtm"].dt.hour.to_numpy()
        h_va = va_g["forecast_kst_dtm"].dt.hour.to_numpy()
        dh = np.abs(h_va[:, None] - h_tr[None, :])
        dh = np.minimum(dh, 24 - dh)
        D = np.where(dh <= hour_win, D, np.inf)
    k = min(K_MAX, D.shape[1])
    idx = np.argpartition(D, k - 1, axis=1)[:, :k]
    rows = np.arange(D.shape[0])[:, None]
    dsub = D[rows, idx]
    order = np.argsort(dsub, axis=1)
    idx, dsub = idx[rows, order], dsub[rows, order]
    return dict(va_g=va_g, cap=cap, ana_cf=y_cf[idx], ana_d=dsub)


def predict_from_analogs(ana, K, kernel="eq", q=None):
    cf = ana["ana_cf"][:, :K]
    d = ana["ana_d"][:, :K]
    if q is not None:
        return np.quantile(cf, q, axis=1)
    if kernel == "inv":
        wk = 1.0 / (d + 1e-6)
        wk = np.where(np.isfinite(wk), wk, 0.0)
        return (cf * wk).sum(1) / np.maximum(wk.sum(1), 1e-12)
    return cf.mean(1)


def stage_smoke(train_df, jobs):
    fold = dict(FOLD_LIST)["fold_2024Q4"]
    cache, _ = load_cache_full()
    print("[smoke] Q4 폴드 — AnEn 구성 비교 (기준 = 챔피언 캐시 cf_hpo)", flush=True)

    # 기준 (cf_hpo, 3-seed 캐시)
    ref = {}
    for group in [1, 2, 3]:
        sub = cache[(cache["fold"] == fold["name"]) & (cache["group"] == group)]
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        m = sub["target"].notna() & (sub["target"] >= cap * 0.10)
        ref[group] = dict(sub=sub, cap=cap, m=m.to_numpy(),
                          y=(sub["target"] / cap).to_numpy())
    ref_mae = np.mean([np.abs(ref[g]["sub"]["cf_hpo"].to_numpy()[ref[g]["m"]]
                              - ref[g]["y"][ref[g]["m"]]).mean() for g in [1, 2, 3]])
    print(f"기준 cf_hpo MAE {ref_mae:.5f}", flush=True)

    ana_cache = {}
    for ptag in ["paper", "hub", "wsonly"]:
        for hw in [None, 3]:
            key = (ptag, hw)
            tt = time.time()
            ana_cache[key] = {g: build_analogs(train_df, fold, g, PRED_SETS[ptag], hw)
                              for g in [1, 2, 3]}
            print(f"  거리계산 {ptag}/hw={hw}: {time.time()-tt:.0f}s", flush=True)

    rows = []
    for (ptag, hw), anas in ana_cache.items():
        for K in [10, 20, 50, 100, 200]:
            for kern in ["eq", "inv"]:
                maes, sgns, cats, refcats = [], [], [], []
                for g in [1, 2, 3]:
                    ana = anas[g]
                    cf = predict_from_analogs(ana, K, kern)
                    va = ana["va_g"]
                    m = va["target"].notna().to_numpy() & \
                        (va["target"].to_numpy() >= ana["cap"] * 0.10)
                    y = va["target"].to_numpy() / ana["cap"]
                    j = va[["forecast_kst_dtm"]].merge(
                        ref[g]["sub"][["forecast_kst_dtm", "cf_hpo"]],
                        on="forecast_kst_dtm", how="left")
                    maes.append(np.abs(cf[m] - y[m]).mean())
                    sgns.append((cf[m] - y[m]).mean())
                    cats.append(cf[m])
                    refcats.append(j["cf_hpo"].to_numpy()[m])
                cat, rc = np.concatenate(cats), np.concatenate(refcats)
                okm = np.isfinite(cat) & np.isfinite(rc)
                rows.append(dict(pset=ptag, hw=hw, K=K, kern=kern,
                                 mae=np.mean(maes), signed=np.mean(sgns),
                                 corr=np.corrcoef(cat[okm], rc[okm])[0, 1]))
    df = pd.DataFrame(rows).sort_values("mae")
    df["d_mae"] = df["mae"] - ref_mae
    with pd.option_context("display.width", 150, "display.max_rows", 100):
        print(df.to_string(index=False, float_format=lambda x: f"{x:+.5f}"), flush=True)
    print("\ngo/no-go: corr < 0.99 (실다양성) ∧ MAE 파트너급(dl_cln03 수준 이내)", flush=True)


def stage_gate(train_df, config_list, seeds_unused=None):
    """선정 구성들을 4-fold 전체 캐시로 만들고 make_ev 사다리 평가."""
    cache, champ_bias = load_cache_full()
    for tag, ptag, hw, K, kern, q in config_list:
        col = f"cf_{tag}"
        cpath = CACHE_DIR / f"an01_{tag}.parquet"
        if col not in merge_cached_cols(cache, cpath, [col]):
            preds = []
            for fname, fold in FOLD_LIST:
                tf = time.time()
                for group in [1, 2, 3]:
                    ana = build_analogs(train_df, fold, group, PRED_SETS[ptag], hw)
                    cf = predict_from_analogs(ana, K, kern, q)
                    preds.append(pd.DataFrame({
                        "fold": fold["name"], "group": group,
                        "forecast_kst_dtm": ana["va_g"]["forecast_kst_dtm"].to_numpy(),
                        col: cf}))
                print(f"  [{tag}:{fname}] {time.time()-tf:.0f}s", flush=True)
            p = pd.concat(preds, ignore_index=True)
            m = cache[KEY].merge(p, on=KEY, how="left")
            assert m[col].notna().all(), f"정렬 실패: {col}"
            cache[col] = m[col].to_numpy()
            save_cols(cache, cpath, [col])
            print(f"  캐시 저장 → {cpath.name}", flush=True)

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} | HOLD {ref['hold'][0]:.4f}",
          flush=True)
    res = {"hpo_solo": ev("cf_hpo", None, 1.0, PP_NEW)}
    hpov = cache["cf_hpo"].to_numpy()
    for tag, *_ in config_list:
        col = f"cf_{tag}"
        for b in ("cf_hpo", "dl_cln03"):
            print(f"  corr({col}, {b}) = {corr_on_eval(cache, col, b):.4f}", flush=True)
        print(f"  평균 이동 ({col} − cf_hpo): "
              f"{(cache[col].to_numpy() - hpov).mean()*100:+.3f}%cap", flush=True)
        res[f"{tag}_solo"] = ev(col, None, 1.0, PP_NEW)
        res[f"{tag}_w65"] = ev(col, "dl_cln03", 0.65, PP_NEW)
        for m_, mt in ((0.9, "t90"), (5 / 6, "t83"), (0.75, "t75"), (2 / 3, "t67")):
            mix = f"mx_{tag}_{mt}"
            cache[mix] = m_ * hpov + (1 - m_) * cache[col].to_numpy()
            res[f"{tag}{mt}_w65"] = ev(mix, "dl_cln03", 0.65, PP_NEW)
    print_table(res, ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "gate"], default="smoke")
    ap.add_argument("--configs", default="")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    print("### AN-01 Analog Ensemble (AnEn) 트랙 ###", flush=True)
    train_df, _ = build_train_frame()
    print(f"입력 준비 완료 — {time.time()-t0:.0f}s", flush=True)

    if args.stage == "smoke":
        stage_smoke(train_df, args.jobs)
    else:
        # 형식: tag:pset:hw:K:kern[:q]  (hw는 none 또는 정수, q 생략시 평균)
        cfgs = []
        for c in args.configs.split(","):
            f = c.strip().split(":")
            hw = None if f[2] == "none" else int(f[2])
            q = float(f[5]) if len(f) > 5 else None
            cfgs.append((f[0], f[1], hw, int(f[3]), f[4], q))
        stage_gate(train_df, cfgs)

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
