# ═══════════════════════════════════════════════════════
# exp-114 점검: L0의 "평균 이동 0"이 후처리를 통과해도 유지되는가
#
# 발견: exp-101/102가 보고한 "블렌드 평균이동 +0.000%cap"은 **후처리 이전** 원시 블렌드에서
#   잰 값이다(`exp101_levelctrl.py:127` — PP 적용 전 `w*gbm + (1-w)*dl`의 평균 차이).
#   그런데 exp-114 제출 파일(후처리·스케일 완료)은 **+0.131%cap 상향**이 남는다.
#   판별 규칙("예측 평균을 올리는 변경은 2025 역전")은 **제출값 기준**이므로 이쪽이 맞는 잣대다.
#
# 원인 후보: PP_NEW의 두 단계가 **비감소 연산**이라 상측 꼬리 변화를 한 방향으로만 반영한다.
#     스냅  p[pA≥0.27] = max(p, 0.93cap)     ← 아래로는 못 내림
#     clip  p = clip(p, 0.16cap, cap)        ← 아래로는 못 내림
#   dl_hpo는 drop 0.349 · q60이라 상측 꼬리가 넓다. 평균을 맞춰도 **모양**은 다르다.
#
# 재는 것:
#   A. 2025 test — PP 단계별 누적 평균 이동 (원시 → 편향 → rolling → 스냅 → clip → 스케일)
#   B. 2024 게이트(CV 4폴드) — 같은 분해. 게이트 점수는 어차피 PP 이후로 채점되므로
#      ΔCV +0.0025/+0.0014에는 이 효과가 **이미 반영돼 있다**. 크기만 확인한다.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp112b_shiftcheck.py
# ═══════════════════════════════════════════════════════
import pickle
import sys

import numpy as np
import pandas as pd

from exp005_qopt import (CAPACITY_KWH, apply_rolling_smoothing, apply_bias_correction_v2)
from exp035_raw_dl import CACHE_DIR, HOLD_FOLD
from exp049_clean import PP_NEW
from exp050_lambda_sweep import merge_cached_cols
from exp060_pergroup_hpo import load_cache_full, build_slices
from exp102_delevel import delevel_lofo
from exp114_submit_ens3L0 import (ALPHA, CLIP_LO, SNAP_THR, SNAP_TO, SCALES, W,
                                 E49_PRED, E49_BIAS, ENS_TEST, HPO_TEST)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STAGES = ["원시 블렌드", "+편향보정", "+rolling", "+스냅", "+clip", "+그룹스케일"]


def pp_stages(cf, cap, ws, pA, group, bias_map, scale=True):
    """PP_NEW를 단계별로 끊어 각 단계 직후 값을 돌려준다 (kWh)."""
    out = []
    p = cf * cap
    out.append(p.copy())
    p = apply_bias_correction_v2(p, ws, group, bias_map);              out.append(p.copy())
    p = apply_rolling_smoothing(pd.Series(p), 3, ALPHA).to_numpy().copy(); out.append(p.copy())
    p = p.copy(); p[pA >= SNAP_THR] = np.maximum(p[pA >= SNAP_THR], cap * SNAP_TO); out.append(p.copy())
    p = np.clip(p, cap * CLIP_LO, cap);                                out.append(p.copy())
    p = np.clip(p * (SCALES[group] if scale else 1.0), 0, cap);        out.append(p.copy())
    return out


def report(title, rows):
    """rows: list of (group, cap, base_stages, cand_stages)"""
    print(f"\n═══ {title} ═══", flush=True)
    print(f"  {'단계':<14}" + "".join(f"{f'g{g}':>11}" for g, *_ in rows[:3]) + f"{'전체':>11}", flush=True)
    for i, name in enumerate(STAGES):
        cells, num, den = [], 0.0, 0.0
        for g, cap, b, c in rows:
            d = (c[i] - b[i]).mean() / cap * 100
            cells.append(f"{d:>+10.3f}%")
            num += (c[i] - b[i]).sum() / cap * 100 * len(b[i]) / len(b[i])
            den += 1
        tot = np.mean([(c[i] - b[i]).mean() / cap * 100 for g, cap, b, c in rows])
        print(f"  {name:<14}" + "".join(cells) + f"{tot:>+10.3f}%", flush=True)


def main():
    print("### exp-114: L0의 평균 정렬이 후처리를 통과하는가 ###", flush=True)

    # ── A. 2025 test ────────────────────────────────────
    pred = (pd.read_parquet(E49_PRED)
            .merge(pd.read_parquet(ENS_TEST), on=["forecast_id", "kpx_group"], how="left")
            .merge(pd.read_parquet(HPO_TEST), on=["forecast_id", "kpx_group"], how="left"))
    with open(E49_BIAS, "rb") as f:
        bias = pickle.load(f)
    rows = []
    for g in [1, 2, 3]:
        s = pred[pred["kpx_group"] == g]
        cap = CAPACITY_KWH[f"kpx_group_{g}"]
        sh = (s["dl_cf_hpo"] - s["dl_cf"]).mean()
        base_dl, cand_dl = s["dl_cf"].to_numpy(), s["dl_cf_hpo"].to_numpy() - sh
        gb = s["gbm_cf_ens3"].to_numpy()
        args = (cap, s["ws"].to_numpy(), s["pA"].to_numpy(), g, bias)
        rows.append((g, cap,
                     pp_stages(W * gb + (1 - W) * base_dl, *args),
                     pp_stages(W * gb + (1 - W) * cand_dl, *args)))
    report("A. 2025 test — 단계별 누적 평균 이동 (챔피언 대비)", rows)
    for g, cap, b, c in rows:
        n_b = int((b[-1] >= cap * 0.93).sum()); n_c = int((c[-1] >= cap * 0.93).sum())
        hi_b = int((b[2] >= cap * SNAP_TO).sum()); hi_c = int((c[2] >= cap * SNAP_TO).sum())
        print(f"    g{g}: 스냅 전 이미 ≥0.93cap {hi_b} → {hi_c} | 최종 ≥0.93cap {n_b} → {n_c}",
              flush=True)

    # ── B. 2024 게이트 (CV 4폴드) ───────────────────────
    cache, champ_bias = load_cache_full()
    assert merge_cached_cols(cache, CACHE_DIR / "exp058_dl_hpo.parquet", ["dl_hpo"]), "게이트 캐시 필요"
    delevel_lofo(cache, "dl_hpo", "dl_cln03", "dl_hpo_L0")
    acc = {i: [] for i in range(len(STAGES))}
    for (fn, g), sub in build_slices(cache):
        if fn == HOLD_FOLD["name"]:
            continue
        cap = CAPACITY_KWH[f"kpx_group_{g}"]
        gb = cache["cf_hpo"].to_numpy()[sub.index]
        args = (cap, sub["ws"].to_numpy(), sub["pA"].to_numpy(), g, champ_bias[fn])
        b = pp_stages(W * gb + (1 - W) * cache["dl_cln03"].to_numpy()[sub.index], *args, scale=False)
        c = pp_stages(W * gb + (1 - W) * cache["dl_hpo_L0"].to_numpy()[sub.index], *args, scale=False)
        for i in range(len(STAGES)):
            acc[i].append((c[i] - b[i]).mean() / cap * 100)
    print(f"\n═══ B. 2024 게이트(CV 4폴드) — 단계별 누적 평균 이동 ═══", flush=True)
    print("  (그룹 스케일은 게이트 미적용)", flush=True)
    for i, name in enumerate(STAGES[:-1]):
        print(f"  {name:<14}{np.mean(acc[i]):>+10.3f}%", flush=True)
    print("\n※ 게이트 ΔCV는 **PP 이후 예측으로 채점**되므로 이 상향분은 이미 점수에 반영돼 있다.", flush=True)
    print("  문제는 점수가 아니라 **판별 규칙의 잣대**(제출값 평균 이동)를 pre-PP로 읽었다는 것.", flush=True)


if __name__ == "__main__":
    main()
