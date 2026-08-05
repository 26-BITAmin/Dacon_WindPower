# ═══════════════════════════════════════════════════════
# AN-05: AnEn 예측자 가중 최적화 — 문헌의 최대 레버
#
# AN-03/04: 어느 자리에 섞어도 음수 (최선 −0.0002), Q4만 +/Q1·Q3 − 패턴.
# 남은 개선축 = 유사도 자체. 변수별 거리 D_i를 사전계산해 두면
# 임의 가중 w에 대해 D = Σ w_i·D_i 가 행렬 가중합만으로 나옴 → 조합 108개 탐색.
#
# 과적합 방지: 가중은 Q1~Q3 평균 MAE로 고르고 Q4는 손대지 않는다
#   (exp-060~063 교훈: Q4 단독 최적화는 타 폴드에서 붕괴).
# 탐색공간 (그룹별): ws 4소스 고정 1.0, gfs100 ∈{1,2}, 풍향 ∈{0,.5,1},
#   시어 ∈{0,.5}, 기온 ∈{0,.5,1}, 기압 ∈{0,.5,1}
# 비용 절감: 후보·검증행 1/2 서브샘플 (가중 "순위" 비교에는 충분)
#
# 실행: C:\miniconda3\python.exe -X utf8 an05_wopt.py
# ═══════════════════════════════════════════════════════
import itertools
import sys
import time

import numpy as np
import pandas as pd

from an01_anen import lagged_block
from exp005_qopt import CAPACITY_KWH
from exp035_raw_dl import FOLD_LIST
from exp065_gbm3 import build_train_frame, slice_fold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# (컬럼, 탐색레벨 튜플) — 레벨 1개면 고정
SPACE = [("gfs_ws_100m", (1.0, 2.0)),
         ("ldaps_ws_10m", (1.0,)), ("ec_ws", (1.0,)),
         ("jmsm_ws", (1.0,)), ("jgsm_ws", (1.0,)),
         ("wd_pair", (0.0, 0.5, 1.0)),          # gfs 100m sin+cos, 각 w/2
         ("gfs_wind_shear_100_10", (0.0, 0.5)),
         ("ldaps_heightAboveGround_2_t", (0.0, 0.5, 1.0)),
         ("ldaps_surface_0_sp", (0.0, 0.5, 1.0))]
K_EVAL = 100
HW = 3


def pervar_dists(train_df, fold, group, sub_tr=2, sub_va=2):
    """변수별 거리 블록 [V, Nva', Ntr'] (필터·서브샘플 적용, float32)."""
    cols = []
    for c, _ in SPACE:
        cols += ["gfs_wd_sin_100m", "gfs_wd_cos_100m"] if c == "wd_pair" else [c]
    tr_all, va_all = slice_fold(train_df, fold)
    tr_g = tr_all.loc[tr_all["kpx_group"] == group].sort_values(
        "forecast_kst_dtm").reset_index(drop=True)
    va_g = va_all.loc[va_all["kpx_group"] == group].sort_values(
        "forecast_kst_dtm").reset_index(drop=True)
    cap = CAPACITY_KWH[f"kpx_group_{group}"]
    A = lagged_block(tr_g, cols)
    F = lagged_block(va_g, cols)
    av = np.nan_to_num(tr_g["avail_strict"].to_numpy(dtype=float), nan=1.0)
    ok = tr_g["target"].notna().to_numpy() & (av >= 0.999)
    tr_idx = np.flatnonzero(ok)[::sub_tr]
    va_idx = np.arange(len(va_g))[::sub_va]
    va_lab = va_g["target"].notna().to_numpy()[va_idx]
    va_idx = va_idx[va_lab]
    sig = np.nanstd(tr_g[cols].to_numpy(dtype=np.float64), axis=0)
    h_tr = tr_g["forecast_kst_dtm"].dt.hour.to_numpy()[tr_idx]
    h_va = va_g["forecast_kst_dtm"].dt.hour.to_numpy()[va_idx]
    dh = np.abs(h_va[:, None] - h_tr[None, :])
    hour_bad = np.minimum(dh, 24 - dh) > HW

    blocks, ci = [], 0
    for c, _ in SPACE:
        ncomp = 2 if c == "wd_pair" else 1
        acc = np.zeros((len(va_idx), len(tr_idx)), dtype=np.float64)
        for k in range(ncomp):
            j = ci + k
            sq = np.zeros_like(acc)
            for L in range(A.shape[2]):
                diff = F[va_idx, j, L][:, None] - A[tr_idx, j, L][None, :]
                sq += diff * diff
            acc += (0.5 if ncomp == 2 else 1.0) / max(sig[j], 1e-9) * np.sqrt(sq)
        blocks.append(acc.astype(np.float32))
        ci += ncomp
    y_va = va_g["target"].to_numpy()[va_idx] / cap
    y_tr = tr_g["target"].to_numpy()[tr_idx] / cap
    m_ev = y_va >= 0.10
    return np.stack(blocks), y_tr, y_va, m_ev, hour_bad


def eval_weights(stack, y_tr, y_va, m_ev, hour_bad, w):
    D = np.tensordot(np.asarray(w, dtype=np.float32), stack, axes=1)
    D[hour_bad] = np.inf
    k = min(K_EVAL, D.shape[1])
    idx = np.argpartition(D, k - 1, axis=1)[:, :k]
    cf = y_tr[idx].mean(1)
    return np.abs(cf[m_ev] - y_va[m_ev]).mean()


def main():
    t0 = time.time()
    print("### AN-05 AnEn 예측자 가중 최적화 (Q1~Q3 탐색 / Q4 검증) ###", flush=True)
    train_df, _ = build_train_frame()
    print(f"입력 준비 완료 — {time.time()-t0:.0f}s", flush=True)

    folds = dict(FOLD_LIST)
    combos = list(itertools.product(*[lv for _, lv in SPACE]))
    print(f"가중 조합 {len(combos)}개", flush=True)
    base_w = tuple(lv[0] if len(lv) == 1 else (1.0 if c != "gfs_wind_shear_100_10"
                   else 0.5) for c, lv in SPACE)  # AN-02 hub 기본형 근사

    best = {}
    for g in [1, 2, 3]:
        mae_sum = np.zeros(len(combos))
        for fn in ["fold_2024Q1", "fold_2024Q2", "fold_2024Q3"]:
            tt = time.time()
            stack, y_tr, y_va, m_ev, hbad = pervar_dists(train_df, folds[fn], g)
            maes = np.array([eval_weights(stack, y_tr, y_va, m_ev, hbad, w)
                             for w in combos])
            mae_sum += maes
            del stack
            print(f"  그룹{g} {fn}: 최선 {maes.min():.5f} / 기본형 "
                  f"{maes[combos.index(base_w)]:.5f} — {time.time()-tt:.0f}s", flush=True)
        rank = np.argsort(mae_sum)
        best[g] = combos[rank[0]]
        print(f"[그룹{g}] 최적 가중:", flush=True)
        for (c, _), w in zip(SPACE, best[g]):
            print(f"    {c:<32}{w:.1f}", flush=True)
        print(f"    Q1~3 평균 MAE {mae_sum[rank[0]]/3:.5f} vs 기본형 "
              f"{mae_sum[combos.index(base_w)]/3:.5f} "
              f"(Δ {(mae_sum[rank[0]]-mae_sum[combos.index(base_w)])/3:+.5f})", flush=True)
        top5 = [(combos[i], mae_sum[i] / 3) for i in rank[:5]]
        for w, m in top5:
            print(f"    상위: {tuple(round(x,1) for x in w)} → {m:.5f}", flush=True)

    # Q4 검증 (탐색에 미사용)
    print("\n[Q4 검증] (전해상도 아님 — 동일 서브샘플 조건 비교)", flush=True)
    for g in [1, 2, 3]:
        stack, y_tr, y_va, m_ev, hbad = pervar_dists(train_df, folds["fold_2024Q4"], g)
        m_b = eval_weights(stack, y_tr, y_va, m_ev, hbad, base_w)
        m_o = eval_weights(stack, y_tr, y_va, m_ev, hbad, best[g])
        del stack
        print(f"  그룹{g}: 기본형 {m_b:.5f} → 최적 {m_o:.5f} (Δ {m_o-m_b:+.5f})", flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
