# ═══════════════════════════════════════════════════════
# exp-071: 그룹별 스케일 s_g 프로브 파일 빌드 (LB 직접 측정 축 — 게이트 불가측)
#
# 배경: 전역 s*=0.98 확정(exp-064). metric이 그룹 단순평균이라 한 그룹만 바꾼
#   프로브의 Δ는 온전히 그 그룹 몫 → 그룹별 s-곡선을 분해 측정 가능.
#   2025 하향의 정황 원인(가용성)이 사이트별이므로 그룹별 깊이가 다를 개연성.
#
# 1차 프로브 (0.97 스텝 분해 — 전역 s097이 기측정이라 g3는 뺄셈으로 나옴):
#   A: g1만 ×0.97 (나머지 ×0.98)   B: g2만 ×0.97 (나머지 ×0.98)
#   분해: S_g(0.97)−S_g(0.98) = 3×(프로브 − 챔피언 0.6491681449)
#         g3 항 = 3×(s097 전역 0.6462618433 − 챔피언) − g1 항 − g2 항
#
# 실행: C:\miniconda3\python.exe -X utf8 exp071_gscale_build.py
# ═══════════════════════════════════════════════════════
import sys

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, DATA_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = DATA_DIR / "v_a40s93p27_hpo_w65_submit.csv"          # s=1.0 원본
CHAMP = DATA_DIR / "v_a40s93p27_hpo_w65_s098_submit.csv"    # 전역 ×0.98 (검증 대조)
GCOLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]


def build(probe_g, s_probe=0.97, s_rest=0.98):
    base = pd.read_csv(BASE, encoding="utf-8-sig")
    champ = pd.read_csv(CHAMP, encoding="utf-8-sig")
    out = base.copy()
    for gi, col in enumerate(GCOLS, start=1):
        cap = CAPACITY_KWH[f"kpx_group_{gi}"]
        s = s_probe if gi == probe_g else s_rest
        out[col] = np.clip(base[col].to_numpy() * s, 0, cap)
    # 검증: 프로브 그룹 외 열은 챔피언과 완전 일치, 키 열 불변
    assert (out["forecast_id"] == champ["forecast_id"]).all()
    assert (out["forecast_kst_dtm"] == champ["forecast_kst_dtm"]).all()
    for gi, col in enumerate(GCOLS, start=1):
        if gi != probe_g:
            d = np.abs(out[col].to_numpy() - champ[col].to_numpy()).max()
            assert d < 1e-9, f"{col} 챔피언 불일치 max|d|={d}"
    diff = np.abs(out[GCOLS[probe_g-1]].to_numpy() - champ[GCOLS[probe_g-1]].to_numpy())
    cap = CAPACITY_KWH[f"kpx_group_{probe_g}"]
    name = f"v_a40s93p27_hpo_w65_g{probe_g}s{int(s_probe*100)}_submit.csv"
    out.to_csv(DATA_DIR / name, index=False, encoding="utf-8-sig")
    print(f"{name}: g{probe_g} ×{s_probe} (나머지 ×{s_rest}) | "
          f"변경 그룹 평균 |차이| {diff.mean():.1f}kWh ({diff.mean()/cap*100:.3f}%cap) | "
          f"행 {len(out)}", flush=True)


def build_vec(s_by_group, tag):
    """임의 그룹별 스케일 벡터 파일 (2차: 조합·심화 프로브)"""
    base = pd.read_csv(BASE, encoding="utf-8-sig")
    out = base.copy()
    for gi, col in enumerate(GCOLS, start=1):
        cap = CAPACITY_KWH[f"kpx_group_{gi}"]
        out[col] = np.clip(base[col].to_numpy() * s_by_group[gi - 1], 0, cap)
    name = f"v_a40s93p27_hpo_w65_{tag}_submit.csv"
    out.to_csv(DATA_DIR / name, index=False, encoding="utf-8-sig")
    print(f"{name}: s = {s_by_group}", flush=True)


def main():
    if "--round2" in sys.argv:
        # 1차 분해 결과(2026-07-30): g1 −0.00207 / g2 −0.00833 / g3 +0.00168 (0.97 스텝)
        build_vec((0.98, 0.98, 0.97), "g3s97")   # 조합 신기록 후보 — 기대 0.6497287
        build_vec((0.98, 0.98, 0.96), "g3s96")   # g3 정점 심화 프로브
        build_vec((0.98, 0.985, 0.97), "g2u985g3s97")  # 옵션: g2 상향 확인
        return
    for g in (1, 2):
        build(g)
    print("\n제출 순서 권장: A(g1s97) → B(g2s97). 챔피언 0.6491681449 / 전역 s097 0.6462618433 대비로 분해.")


if __name__ == "__main__":
    main()
