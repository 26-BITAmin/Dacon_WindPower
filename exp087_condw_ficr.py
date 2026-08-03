# ═══════════════════════════════════════════════════════
# exp-087: 진단 — "FiCR 구간에서만 DL 비중 상향" (조건부 w, 캐시 전용, 재학습 0)
#
# 발상 (사용자): FiCR은 actual 가중이라 사실상 **고발전 시간대 지표**이고 1-NMAE는 전 시간
#   동등 가중이다. 두 지표가 서로 다른 시간대를 보므로, 고발전 구간에서만 DL(=FiCR 담당)
#   비중을 올리면 FiCR을 사고 NMAE를 덜 내줄 수 있다.
#
# 선행 결과 (exp-046, 2024 게이트에서 기각):
#   - **실측** CF 조건부 최적 w: CF 0.7~0.9 → 0.30 / CF 0.9+ → 0.00 (= 이 발상과 같은 방향)
#   - 그러나 **예측** CF 조건으로 바꾸면 희석돼 소멸, LOFO −0.0010/−0.0006
#   - 관측 가능 레짐(pA·풍속)에서는 오히려 w*≈1.0 (DL 빼기) — 제안과 반대
#
# 재검 근거: 위 판정은 전부 2024 게이트. exp-083 제출에서 **게이트가 DL의 FiCR 기여를
#   ~4배 과소평가**함이 LB로 측정됨(w 0.65→0.80: 게이트 Δf −0.0007 vs LB −0.00289).
#   저울이 의심스러우므로 **그 저울이 무엇을 말하는지 먼저 정확히 본다.**
#
# 이 스크립트는 **탐색이 아니라 진단**이다 (exp-046의 실패 모드 = CV 격자 선택 과적합).
#   목적: ① 조건 영역이 FiCR 질량의 몇 %를 쥐는가 ② 그 영역에서 w를 내리면
#   Δ1-NMAE와 ΔFiCR이 각각 어느 방향·크기로 움직이는가 (부호와 기울기만 본다)
#
# 실행: C:\miniconda3\python.exe -X utf8 exp087_condw_ficr.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np

from exp005_qopt import CAPACITY_KWH
from exp049_clean import PP_NEW
from exp060_pergroup_hpo import load_cache_full, make_ev

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

W_LO = 0.65                                   # 조건 밖 = 챔피언 유지
W_HI = [0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.20, 0.00]   # 조건 안 (0.65 = 챔피언 재현)


def main():
    t0 = time.time()
    print("### exp-087 진단: FiCR 구간 조건부 DL 비중 ###", flush=True)
    cache, champ_bias = load_cache_full()
    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"챔피언 재현: CV {ref['cv'][0]:.4f} (N {ref['cv'][1]:.4f} F {ref['cv'][2]:.4f}) | "
          f"HOLD {ref['hold'][0]:.4f}  (기대 0.6556/0.6535)", flush=True)

    g = cache["group"].to_numpy()
    caps = np.array([CAPACITY_KWH[f"kpx_group_{int(x)}"] for x in g])
    y = cache["target"].to_numpy()
    cf_act = y / caps
    blend65 = 0.65 * cache["cf_hpo"].to_numpy() + 0.35 * cache["dl_cln03"].to_numpy()
    pA = cache["pA"].to_numpy()
    ws = cache["ws"].to_numpy()

    # ── 조건 후보 (전부 **관측 가능**한 것만 — 실측 CF는 오라클 참고용) ──
    CONDS = {
        "pA>=0.27 (스냅권)": pA >= 0.27,
        "pA>=0.15":          pA >= 0.15,
        "pred_CF>=0.70":     blend65 >= 0.70,
        "pred_CF>=0.55":     blend65 >= 0.55,
        "ws>=11m/s":         ws >= 11.0,
        "[오라클] act_CF>=0.70": cf_act >= 0.70,
    }

    # ── ① 각 조건이 FiCR 질량의 몇 %를 쥐는가 ──
    ok = ~np.isnan(y) & (y >= caps * 0.10)          # 평가 대상 시간
    mass_all = np.nansum(cf_act[ok])
    print(f"\n--- ① 조건 영역의 레버리지 (평가시간 n={ok.sum():,}) ---", flush=True)
    print(f"{'조건':<24}{'시간 비율':>10}{'FiCR 질량 비율':>16}", flush=True)
    for name, c in CONDS.items():
        m = ok & c
        print(f"{name:<24}{m.sum()/ok.sum()*100:>9.1f}%"
              f"{np.nansum(cf_act[m])/mass_all*100:>15.1f}%", flush=True)

    # ── ② 조건 안에서만 w를 바꿨을 때 두 지표가 각각 어디로 가는가 ──
    print(f"\n--- ② 조건부 w 사다리 (조건 밖 w={W_LO} 고정, Δ = 챔피언 대비) ---", flush=True)
    for name, c in CONDS.items():
        if name.startswith("[오라클]"):
            continue
        print(f"\n  [{name}]", flush=True)
        print(f"    {'w_hi':>6}{'ΔCV':>10}{'Δhold':>10}{'Δ1-NMAE':>11}{'ΔFiCR':>10}"
              f"{'평균이동':>11}", flush=True)
        for wh in W_HI:
            wrow = np.where(c, wh, W_LO)
            cache["cw"] = wrow * cache["cf_hpo"] + (1 - wrow) * cache["dl_cln03"]
            r = ev("cw", None, 1.0, PP_NEW)
            shift = (cache["cw"].to_numpy() - blend65).mean() * 100
            mark = "  ← 챔피언" if wh == 0.65 else ""
            print(f"    {wh:>6.2f}{r['cv'][0]-ref['cv'][0]:>+10.4f}"
                  f"{r['hold'][0]-ref['hold'][0]:>+10.4f}"
                  f"{r['cv'][1]-ref['cv'][1]:>+11.4f}{r['cv'][2]-ref['cv'][2]:>+10.4f}"
                  f"{shift:>+10.3f}%{mark}", flush=True)

    # ── ③ 참고: 전역 w 사다리 (조건 없음) — 오늘 LB 프로브의 게이트측 대응 ──
    print(f"\n--- ③ 참고: 전역 w 사다리 (조건 없음) ---", flush=True)
    print(f"    {'w':>6}{'ΔCV':>10}{'Δhold':>10}{'Δ1-NMAE':>11}{'ΔFiCR':>10}{'평균이동':>11}",
          flush=True)
    for w in [0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]:
        r = ev("cf_hpo", "dl_cln03", w, PP_NEW)
        blend = w * cache["cf_hpo"].to_numpy() + (1 - w) * cache["dl_cln03"].to_numpy()
        print(f"    {w:>6.2f}{r['cv'][0]-ref['cv'][0]:>+10.4f}"
              f"{r['hold'][0]-ref['hold'][0]:>+10.4f}"
              f"{r['cv'][1]-ref['cv'][1]:>+11.4f}{r['cv'][2]-ref['cv'][2]:>+10.4f}"
              f"{(blend-blend65).mean()*100:>+10.3f}%"
              f"{'  ← 챔피언' if w == 0.65 else ''}", flush=True)

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
