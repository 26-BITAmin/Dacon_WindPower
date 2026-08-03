# ═══════════════════════════════════════════════════════
# exp-114 제출: 레벨 제거 DL 멤버 `dl_hpo_L0` — 변경점 1개 (DL 멤버만)
#
#   챔피언(v_a40s93p27_ens3_w65_g3s97, LB 0.64987) 대비 **변경점 단 1개**:
#     DL 멤버 dl_cln03(384×3·drop0.15·ep120·lr1e-3·wd1e-4)
#       → dl_hpo(640×4·drop0.349·ep80·lr4.3e-4·wd1e-6) + **그룹별 레벨 정렬(L0)**
#   GBM(ens3 20-seed)·pA·편향맵·PP_NEW·그룹 스케일(0.98/0.98/0.97)·**w 0.65 전부 동일 캐시**.
#   DL 입력(spd 348)·손실(q60)·가용성 가중(λ0.05)도 동일 — 바뀌는 건 아키텍처 하이퍼 5개뿐.
#   → 새로 들어가는 정보 0. 재학습은 DL 5-seed 하나뿐.
#
# 근거 (exp-100 → exp-101 → exp-102 → exp-103):
#   dl_hpo = exp-058 **무제약** Optuna 우승자. 당시 게이트는 통과(+0.0035/+0.0018)했으나
#   DL 단독 signed 편향이 +0.022 → **+0.062(3배)** 라 판별 규칙("예측 평균을 올리는 변경은
#   2025 역전")에 걸려 미제출됐다. L0는 **그 레벨만** 제거한다.
#     · 28개 DL 멤버 전부에 같은 처리 → 중앙값 +0.0002 (공짜 점심 아님), dl_hpo_L0만 +0.0025로 1위
#     · 대조 cf_bd50(exp-076 상향형 GBM)은 레벨 제거 시 −0.0015로 붕괴 = 그쪽은 레벨이 전부였음
#     · exp-103 선택폴드(Q4) 제외 재판정: **+0.0014** ← 정직한 기대치 (4폴드 +0.0025의 61%가 Q4)
#
# L0 정의 (라벨 미사용 → 2025에서도 동일 산출):
#     shift(g) = mean(dl_hpo[g]) − mean(dl_cln03[g]);   dl_hpo_L0 = dl_hpo − shift(g)
#   CV에서는 채점 폴드를 제외하고 추정(LOFO)했고, 2025에는 폴드가 없으므로 **테스트 예측
#   자체에서 추정**한다. 두 값을 나란히 출력해 크게 어긋나면 보이게 한다.
#
# ⚠ 알려진 위험 (제출 판단 시 볼 것)
#   - 같은 exp-058 탐색의 **다른** 가지(균형제약 512×5, 레벨 유지, w0.70)는 **LB −0.0029로 역전**됨.
#   - 전 변형에서 Q3 폴드가 음수 (w0.65가 가장 깊음), Δn −0.0006.
#   - 게이트는 표본이 하나(CV·홀드아웃 동일 26,352행) · LB 1σ ≈ 0.0006.
#   - w0.75가 프로파일은 더 깨끗하나(Δn −0.0001) **변경점 2개**가 되므로 여기서는 만들지 않는다.
#
#   ⚠ 파일만 만든다. 제출 슬롯 소비는 사용자 판단.
# 실행: C:\miniconda3\python.exe -X utf8 exp112_submit_hpoL0.py [--dl_seeds 5]
# ═══════════════════════════════════════════════════════
import argparse
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import (CAPACITY_KWH, TARGET_COLS, DATA_DIR,
                         apply_rolling_smoothing, apply_bias_correction_v2)
from exp012_feature_v11 import build_v11
from exp020_jma_features import build_jma_frame, JG_COLS, JM_COLS
from exp020_submit import add_jma
from exp033_dl import predict, GIDX
from exp035_grid_features import build_all as build_spatial_all
from exp035_raw_dl import CACHE_DIR
from exp039_dl_diverse import make_loss, train_one
from exp049_clean import merge_avail
from exp050_lambda_sweep import make_xy_lam

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ALPHA, CLIP_LO, SNAP_THR, SNAP_TO = 0.40, 0.16, 0.27, 0.93
SCALES = {1: 0.98, 2: 0.98, 3: 0.97}
W = 0.65                                                    # 챔피언 유지 (변경점 1개 원칙)
LAM = 0.05                                                  # exp-058 게이트와 동일 가중
E49_PRED = CACHE_DIR / "exp049_test_pred_cln03_g20_d5.parquet"   # pA·ws·dl_cf(챔피언 DL 5-seed)
E49_BIAS = CACHE_DIR / "exp049_test_bias_cln03_g20_d5.pkl"
ENS_TEST = CACHE_DIR / "exp073_test_gbm_ens3_g20.parquet"        # GBM ens3 20-seed
HPO_JSON = CACHE_DIR / "exp058_dlhpo_best.json"                  # 무제약 우승 cfg
HPO_TEST = CACHE_DIR / "exp112_test_dl_hpo_d5.parquet"
GATE_HPO = CACHE_DIR / "exp058_dl_hpo.parquet"                   # 게이트 OOF (참고용 shift)
GATE_CLN = CACHE_DIR / "exp049_dl_clean.parquet"
CHAMP_FILE = "v_a40s93p27_ens3_w65_g3s97_submit.csv"


def build(pred, dl_col, coarse_map, sample_submission, w=W):
    """챔피언 후처리 전체 (편향 → rolling → 스냅 → clip → 그룹 스케일)."""
    parts = []
    for group in [1, 2, 3]:
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        g = pred.loc[pred["kpx_group"] == group]
        pA = g["pA"].to_numpy()
        p = apply_bias_correction_v2(
            (w * g["gbm_cf_ens3"].to_numpy() + (1 - w) * g[dl_col].to_numpy()) * cap,
            g["ws"].to_numpy(), group, coarse_map)
        p = apply_rolling_smoothing(pd.Series(p), window=3, alpha=ALPHA).to_numpy().copy()
        p[pA >= SNAP_THR] = np.maximum(p[pA >= SNAP_THR], cap * SNAP_TO)
        p = np.clip(p, cap * CLIP_LO, cap)
        p = np.clip(p * SCALES[group], 0, cap)
        parts.append(pd.DataFrame({"forecast_id": g["forecast_id"].to_numpy(),
                                   "col": f"kpx_group_{group}", "pred": p}))
    sub = pd.concat(parts).pivot(index="forecast_id", columns="col", values="pred").reset_index()
    sub = sample_submission[["forecast_id", "forecast_kst_dtm"]].merge(sub, on="forecast_id", how="left")
    assert sub.shape[0] == sample_submission.shape[0], "행 수 불일치"
    assert sub[TARGET_COLS].isna().sum().sum() == 0, "NaN 존재"
    assert (sub["forecast_id"] == sample_submission["forecast_id"]).all(), "forecast_id 불일치"
    assert (sub["forecast_kst_dtm"] == sample_submission["forecast_kst_dtm"]).all(), "dtm 불일치"
    for col in TARGET_COLS:
        assert sub[col].max() <= CAPACITY_KWH[col] and sub[col].min() >= 0, f"{col} 범위 이탈"
    return sub


def gate_shift():
    """게이트(2024 OOF)에서의 그룹별 레벨 차이 — 2025 추정치와 대조용."""
    try:
        h = pd.read_parquet(GATE_HPO); c = pd.read_parquet(GATE_CLN)
        key = ["fold", "group", "forecast_kst_dtm"]
        m = h[key + ["dl_hpo"]].merge(c[key + ["dl_cln03"]], on=key, how="inner")
        return (m.assign(d=m["dl_hpo"] - m["dl_cln03"])
                  .groupby("group")["d"].mean().to_dict())
    except Exception as e:
        print(f"  (게이트 shift 대조 생략: {e})", flush=True)
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl_seeds", type=int, default=5)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    cfg = json.loads(HPO_JSON.read_text())
    print("### exp-114 제출: dl_hpo_L0 (변경점 1개 = DL 멤버) ###", flush=True)
    print(f"DL cfg (exp-058 무제약 우승): {cfg}", flush=True)

    # === 입력 (게이트와 동일: spd 348) ===
    train_df, test_df, col_sets, sample_submission = build_v11()
    jma = build_jma_frame()
    train_df = add_jma(train_df, jma); test_df = add_jma(test_df, jma)
    cols = col_sets["v10TPS"] + JG_COLS + JM_COLS
    _, spd, _, spd_cols = build_spatial_all()
    train_df = train_df.merge(spd, on="forecast_kst_dtm", how="left")
    test_df = test_df.merge(spd, on="forecast_kst_dtm", how="left")
    vcols = list(cols) + spd_cols
    train_df = merge_avail(train_df)
    print(f"DL 입력 {len(vcols)}피처 | 학습행 {int(train_df['target'].notna().sum()):,}", flush=True)

    # === dl_hpo 5-seed 전 학습창 학습 → 2025 예측 ===
    if HPO_TEST.exists() and not args.refresh:
        dlp = pd.read_parquet(HPO_TEST)
        print(f"DL test 캐시 로드: {HPO_TEST.name}", flush=True)
    else:
        import torch
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        sca = StandardScaler()
        Xtr, gtr, ytr, wtr = make_xy_lam(train_df, vcols, imp, sca, LAM, fit=True)
        te = test_df.sort_values(["kpx_group", "forecast_kst_dtm"]).copy()
        Xte = sca.transform(imp.transform(te[vcols].to_numpy(dtype=np.float32))).astype(np.float32)
        gte = te["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
        loss_fn = make_loss("q60")
        preds = []
        for s in range(args.dl_seeds):
            m = train_one(Xtr, gtr, ytr, wtr, cfg, loss_fn, s)
            preds.append(predict(m, Xte, gte))
            del m
            torch.cuda.empty_cache()
            print(f"  DL seed {s} 완료 ({time.time()-t0:.0f}s)", flush=True)
        dlp = pd.DataFrame({"forecast_id": te["forecast_id"].to_numpy(),
                            "kpx_group": te["kpx_group"].to_numpy(),
                            "dl_cf_hpo": np.mean(preds, axis=0)})
        dlp.to_parquet(HPO_TEST, index=False)
        print(f"DL test 캐시 저장 → {HPO_TEST.name}", flush=True)

    # === 무변경 성분 ===
    pred = (pd.read_parquet(E49_PRED)
            .merge(pd.read_parquet(ENS_TEST), on=["forecast_id", "kpx_group"], how="left")
            .merge(dlp, on=["forecast_id", "kpx_group"], how="left"))
    assert pred["gbm_cf_ens3"].notna().all() and pred["dl_cf"].notna().all() \
        and pred["dl_cf_hpo"].notna().all(), "성분 병합 실패"
    with open(E49_BIAS, "rb") as f:
        coarse_map = pickle.load(f)
    champ = pd.read_csv(DATA_DIR / CHAMP_FILE, encoding="utf-8-sig")

    # === 무결성: 챔피언 DL로 재조합하면 챔피언 파일과 완전 일치해야 한다 ===
    chk = build(pred, "dl_cf", coarse_map, sample_submission)
    dmax = np.abs(chk[TARGET_COLS].to_numpy() - champ[TARGET_COLS].to_numpy()).max()
    ok = dmax < 1e-6
    print(f"\n[무결성] 챔피언 DL 재조합 vs 챔피언 파일 최대 차이 {dmax:.6f} kWh "
          f"({'일치' if ok else '★불일치 — 중단'})", flush=True)
    if not ok:
        sys.exit(1)

    # === L0: 그룹별 레벨 정렬 (라벨 미사용) ===
    gs = gate_shift()
    print(f"\n[레벨] dl_hpo − dl_cln03 그룹 평균", flush=True)
    print(f"  {'그룹':<6}{'2025 test':>14}{'2024 게이트':>16}", flush=True)
    pred["dl_cf_hpoL0"] = pred["dl_cf_hpo"]
    for g in [1, 2, 3]:
        m = pred["kpx_group"] == g
        sh = (pred.loc[m, "dl_cf_hpo"] - pred.loc[m, "dl_cf"]).mean()
        pred.loc[m, "dl_cf_hpoL0"] = pred.loc[m, "dl_cf_hpo"] - sh
        print(f"  g{g:<5}{sh*100:>+13.3f}%cap{gs.get(g, float('nan'))*100:>+15.3f}%cap", flush=True)
    for c in ("dl_cf_hpo", "dl_cf_hpoL0"):
        r = np.corrcoef(pred[c], pred["dl_cf"])[0, 1]
        print(f"  corr({c}, 챔피언 DL) = {r:.4f}", flush=True)

    # === 제출 파일 ===
    sub = build(pred, "dl_cf_hpoL0", coarse_map, sample_submission)
    out = "v_a40s93p27_ens3_hpoL0_w65_g3s97_submit.csv"
    sub[["forecast_id", "forecast_kst_dtm"] + TARGET_COLS].to_csv(
        DATA_DIR / out, index=False, encoding="utf-8-sig")
    arr = sub[TARGET_COLS].to_numpy() - champ[TARGET_COLS].to_numpy()
    print(f"\n[제출] {out}", flush=True)
    print(f"  챔피언 대비 평균 이동 {arr.mean()/21600*100:+.3f}%cap "
          f"| 평균 |차이| {np.abs(arr).mean()/21600*100:.3f}%cap", flush=True)
    for i, col in enumerate(TARGET_COLS):
        cap = CAPACITY_KWH[col]
        d = (sub[col].to_numpy() - champ[col].to_numpy())
        print(f"  {col}: 이동 {d.mean()/cap*100:+.3f}%cap | 스냅(≥0.93cap) "
              f"{int((sub[col] >= cap*0.93).sum())} (챔피언 {int((champ[col] >= cap*0.93).sum())})",
              flush=True)
    print("\n※ 판별 규칙 점검 — 평균 이동이 0 근처여야 한다(L0의 목적). "
          "양수로 크면 exp-055/058/076과 같은 역전 프로파일.", flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
