# ═══════════════════════════════════════════════════════
# exp-102: **레벨 제거 DL 멤버** — exp-058 무제약 우승자의 정보 성분만 회수
#
# 배경 (exp-100 → exp-101):
#   exp-100(전 멤버 쌍 1,704셀)에서 ΔCV 상위가 전부 `dl_hpo` 쌍이었고,
#   exp-101이 그 이득을 레벨/정보로 분해했다 —
#     · 순수 오프셋 통제(정보 0, 동일 이동 +0.778%cap): ΔCV +0.0011
#     · dl_hpo 원본:                                    ΔCV +0.0031
#     · dl_hpo 그룹별 레벨 제거(이동 0):                ΔCV +0.0024  ← 살아남음
#     · 대조 cf_bd50(exp-076 상향형 GBM) 레벨 제거:     +(-0.0002) → -0.0015  ← 무너짐
#   즉 exp-076 부스터는 레벨이 전부였지만 dl_hpo는 **레벨을 빼도 이득이 남는다**.
#
#   dl_hpo = exp-058 무제약 Optuna 우승자(640×4·drop0.349·ep80·lr4.3e-4).
#   당시 **미제출 기각 사유가 정확히 그 레벨**이었다(signed +0.022 → +0.062, 3배).
#   사용자 판별 규칙 "예측 평균을 올리는 변경은 2025에서 역전"에 걸린 케이스이므로,
#   레벨을 제거하면 규칙에 걸리지 않는 성분만 남는지 확인한다.
#
# exp-101의 약점 3개를 여기서 막는다:
#   (A) 이동량을 **채점 대상 폴드의 예측으로 추정**했음 → 폴드 제외(LOFO) 추정으로 교체.
#   (B) "레벨 제거가 아무 멤버에게나 통하는 공짜 점심"일 수 있음 → 전 DL 멤버 통제.
#   (C) 폴드 일관·프로파일(Δn·Δf) 미확인 → w 사다리 + 폴드별 표.
#
# 배포 가능성: 이동량 추정에 **라벨을 쓰지 않는다**(멤버 예측 평균끼리의 차이).
#   2025 테스트에서도 동일하게 산출 가능.
#
# 재학습 0 · 캐시 전용.
# 실행: C:\miniconda3\python.exe -X utf8 exp102_delevel.py
# ═══════════════════════════════════════════════════════
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from exp035_raw_dl import CACHE_DIR
from exp049_clean import PP_NEW
from exp050_lambda_sweep import merge_cached_cols
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SKIP_PAT = re.compile(r"(_partial|test_|^kma_|^tmp_)")
WS = (0.55, 0.60, 0.65, 0.70, 0.75)
REF_DL = "dl_cln03"


def attach_all_dl(cache):
    out = []
    for path in sorted(Path(CACHE_DIR).glob("*.parquet")):
        if SKIP_PAT.search(path.name):
            continue
        try:
            names = pq.ParquetFile(path).schema.names
        except Exception:
            continue
        want = [c for c in names if c.startswith("dl_") and c not in cache.columns]
        if want:
            out += merge_cached_cols(cache, path, want)
    return out


def delevel_lofo(cache, src, ref, out):
    """(fold, group)별 이동량을 **그 폴드를 제외한** 나머지 폴드에서 추정해 제거.
    라벨 미사용 — 두 멤버의 예측 평균 차이만 사용."""
    mm = cache["target"].notna().to_numpy()
    fold = cache["fold"].to_numpy()
    grp = cache["group"].to_numpy()
    d = (cache[src].to_numpy() - cache[ref].to_numpy())
    vals = cache[src].to_numpy().copy()
    info = {}
    for f in np.unique(fold):
        for g in np.unique(grp):
            sel = (fold == f) & (grp == g)
            other = mm & (fold != f) & (grp == g)          # ← 채점 폴드 제외
            if other.sum() == 0:
                continue
            shift = d[other].mean()
            vals[sel] -= shift
            info[(f, g)] = shift
    cache[out] = vals
    return info


def main():
    t0 = time.time()
    print("### exp-102: 레벨 제거 DL 멤버 (LOFO 추정) ###", flush=True)
    cache, champ_bias = load_cache_full()
    added = attach_all_dl(cache)
    dls = [c for c in cache.columns if c.startswith("dl_")]
    print(f"DL 멤버 {len(dls)}개 로드", flush=True)

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", REF_DL, 0.65, PP_NEW)
    ch_cv, ch_hd = ref["cv"][0], ref["hold"][0]
    ch_n, ch_f = ref["cv"][1], ref["cv"][2]
    cv_folds = list(ref["folds"].keys())
    print(f"챔피언 재현: CV {ch_cv:.4f} | HOLD {ch_hd:.4f}  (기대 0.6556/0.6535)", flush=True)

    # ── (B) 전 DL 멤버 레벨 제거 통제 (w0.65·w0.70) ──────
    print("\n[B] 전 DL 멤버 × 레벨 제거 통제 — '레벨 제거가 공짜 점심인가'", flush=True)
    print(f"{'DL 멤버':<13}{'이동%cap':>9}{'원본ΔCV':>10}{'LVL0ΔCV':>10}{'차이':>9}"
          f"{'원본Δhd':>10}{'LVL0Δhd':>10}", flush=True)
    mm = cache["target"].notna()
    base_mean = cache.loc[mm, REF_DL].mean()
    gen = []
    for c in sorted(dls):
        lv = f"{c}_L0"
        delevel_lofo(cache, c, REF_DL, lv)
        r0, r1 = ev("cf_hpo", c, 0.65, PP_NEW), ev("cf_hpo", lv, 0.65, PP_NEW)
        shift = (cache.loc[mm, c].mean() - base_mean) * 100
        gen.append((c, shift, r0["cv"][0] - ch_cv, r1["cv"][0] - ch_cv,
                    r0["hold"][0] - ch_hd, r1["hold"][0] - ch_hd))
    for c, s, a, b, ha, hb in sorted(gen, key=lambda r: -r[3]):
        print(f"{c:<13}{s:>+9.2f}{a:>+10.4f}{b:>+10.4f}{b-a:>+9.4f}{ha:>+10.4f}{hb:>+10.4f}",
              flush=True)

    # ── (A)(C) dl_hpo_L0 w 사다리 + 혼합 ────────────────
    print("\n[A/C] dl_hpo 레벨 제거 — w 사다리 · 챔피언 DL 혼합", flush=True)
    res = {}
    for w in WS:
        res[f"hpo_w{int(w*100)}"] = ev("cf_hpo", "dl_hpo", w, PP_NEW)
        res[f"hpoL0_w{int(w*100)}"] = ev("cf_hpo", "dl_hpo_L0", w, PP_NEW)
    for frac in (0.25, 0.50, 0.75):
        col = f"dl_mixL0_{int(frac*100)}"
        cache[col] = (frac * cache["dl_hpo_L0"].to_numpy()
                      + (1 - frac) * cache[REF_DL].to_numpy())
        for w in (0.60, 0.65, 0.70, 0.75):
            res[f"mix{int(frac*100)}_w{int(w*100)}"] = ev("cf_hpo", col, w, PP_NEW)
    print_table(res, ref)

    # ── 안전 프로파일 판정 ───────────────────────────────
    print("\n[판정] 안전 프로파일 = ΔCV>+0.001 · Δhold>0 · Δn≥−0.0002 · Δf>0", flush=True)
    safe = []
    for k, v in res.items():
        dcv, dhd = v["cv"][0] - ch_cv, v["hold"][0] - ch_hd
        dn_, df_ = v["cv"][1] - ch_n, v["cv"][2] - ch_f
        if dcv > 0.001 and dhd > 0 and dn_ >= -0.0002 and df_ > 0:
            safe.append((k, dcv, dhd, dn_, df_, v))
    if not safe:
        print("  해당 셀 없음 — Δn 조건 완화(≥−0.0006) 재판정:", flush=True)
        for k, v in res.items():
            dcv, dhd = v["cv"][0] - ch_cv, v["hold"][0] - ch_hd
            dn_, df_ = v["cv"][1] - ch_n, v["cv"][2] - ch_f
            if dcv > 0.001 and dhd > 0 and dn_ >= -0.0006 and df_ > 0:
                safe.append((k, dcv, dhd, dn_, df_, v))
    for k, dcv, dhd, dn_, df_, v in sorted(safe, key=lambda r: -r[1]):
        fd = " ".join(f"{v['folds'][fn]-ref['folds'][fn]:+.4f}" for fn in cv_folds)
        npos = sum(v["folds"][fn] - ref["folds"][fn] > 0 for fn in cv_folds)
        b = 0.65 * cache.loc[mm, "cf_hpo"]  # 표기용
        print(f"  {k:<16} ΔCV {dcv:+.4f} / Δhd {dhd:+.4f} | Δn {dn_:+.4f} Δf {df_:+.4f}"
              f" | fold {fd} ({npos}/4 양수)", flush=True)

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
