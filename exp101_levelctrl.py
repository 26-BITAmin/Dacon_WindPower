# ═══════════════════════════════════════════════════════
# exp-101: exp-100 상위 셀의 **레벨 통제** — 정보인가 상향 이동인가
#
# exp-100(전 멤버 쌍 교차 1,704셀)에서 ΔCV 상위가 전부 `dl_hpo`(또는 상향형 GBM
# `cf_bd50/bd100`)와의 쌍이었고, 프로파일이 **Δn(1-NMAE) 음수 · Δf(FiCR) 큰 양수**로
# 균일했다. 실측: dl_hpo 평균 CF는 챔피언 DL 대비 **+2.22%cap**로 28개 DL 멤버 중 최대.
#
# 이 프로젝트에는 "예측 평균을 올리는 변경은 게이트를 통과해도 2025에서 역전된다"는
# 판별 규칙이 있다(exp-055/058/068/076). dl_hpo는 그 규칙으로 exp-058에서
# **미제출 기각**된 바로 그 무제약 Optuna 우승자다.
#
# 그래서 귀속을 가른다:
#   A. 순수 레벨 통제 — 챔피언 DL에 **상수 오프셋만** 더한 가짜 멤버 사다리.
#      dl_hpo의 +0.02223을 그대로 재현하면 ΔCV가 얼마나 나오는가?
#   B. 레벨 제거 — dl_hpo에서 **그룹별 평균 이동만 빼서** 챔피언 DL 평균에 맞춘 뒤 재채점.
#      ΔCV가 살아남으면 정보, 무너지면 레벨.
#   C. 같은 처리를 상향형 GBM(cf_bd50)에도 적용 (GBM 쪽 상향 통제).
#
# 재학습 0 · 캐시 전용.
# 실행: C:\miniconda3\python.exe -X utf8 exp101_levelctrl.py
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
NEED = ["dl_hpo", "dl_hpob", "cf_bd50", "cf_bd100", "cf_gfens", "cf_gf3",
        "cf_ens3", "cf_pcaA16", "cf_xgb", "cf_rc4"]
WS = (0.55, 0.60, 0.65, 0.70, 0.75)


def attach(cache):
    for path in sorted(Path(CACHE_DIR).glob("*.parquet")):
        if SKIP_PAT.search(path.name):
            continue
        try:
            names = pq.ParquetFile(path).schema.names
        except Exception:
            continue
        want = [c for c in names if c in NEED and c not in cache.columns]
        if want:
            merge_cached_cols(cache, path, want)


def group_recenter(cache, src, ref, out):
    """src 를 그룹별 평균이 ref 와 같아지도록 상수 이동 (평가행 기준, 타깃 미사용)"""
    mm = cache["target"].notna()
    vals = cache[src].to_numpy().copy()
    shifts = {}
    for g, sub in cache.loc[mm].groupby("group"):
        d = sub[src].mean() - sub[ref].mean()
        shifts[g] = d
        vals[(cache["group"] == g).to_numpy()] -= d
    cache[out] = vals
    return shifts


def main():
    t0 = time.time()
    print("### exp-101: 레벨 통제 — 정보인가 상향 이동인가 ###", flush=True)
    cache, champ_bias = load_cache_full()
    attach(cache)
    missing = [c for c in NEED if c not in cache.columns]
    if missing:
        print(f"  [경고] 캐시 누락: {missing}", flush=True)

    mm = cache["target"].notna()
    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    ch_cv, ch_hd = ref["cv"][0], ref["hold"][0]
    print(f"\n챔피언 재현: CV {ch_cv:.4f} | HOLD {ch_hd:.4f}  (기대 0.6556/0.6535)", flush=True)

    d_hpo = cache.loc[mm, "dl_hpo"].mean() - cache.loc[mm, "dl_cln03"].mean()
    print(f"dl_hpo 평균 이동 = {d_hpo:+.5f} CF ({d_hpo*100:+.2f}%cap)", flush=True)

    res = {}

    # ── A. 순수 오프셋 사다리 (가짜 멤버) ────────────────
    print("\n[A] 순수 레벨 통제 — dl_cln03 + δ (정보 0, 이동만)", flush=True)
    offs = [0.005, 0.010, 0.015, float(d_hpo), 0.030]
    for off in offs:
        col = f"dl_off{int(round(off*1000)):03d}"
        cache[col] = cache["dl_cln03"].to_numpy() + off
        for w in WS:
            res[f"off{int(round(off*1000)):03d}_w{int(w*100)}"] = ev("cf_hpo", col, w, PP_NEW)

    # ── B. dl_hpo 레벨 제거 ──────────────────────────────
    sh = group_recenter(cache, "dl_hpo", "dl_cln03", "dl_hpo_lvl0")
    print(f"\n[B] dl_hpo 그룹별 레벨 제거: " +
          " ".join(f"g{int(g)} {v:+.4f}" for g, v in sh.items()), flush=True)
    for w in WS:
        res[f"hpo_w{int(w*100)}"] = ev("cf_hpo", "dl_hpo", w, PP_NEW)
        res[f"hpoLVL0_w{int(w*100)}"] = ev("cf_hpo", "dl_hpo_lvl0", w, PP_NEW)
    # 최강 GBM 파트너(gfens)와도 동일 대조
    if "cf_gfens" in cache.columns:
        for w in (0.60, 0.65, 0.70):
            res[f"gfensXhpo_w{int(w*100)}"] = ev("cf_gfens", "dl_hpo", w, PP_NEW)
            res[f"gfensXhpoLVL0_w{int(w*100)}"] = ev("cf_gfens", "dl_hpo_lvl0", w, PP_NEW)

    # ── C. GBM 상향형(cf_bd50) 동일 처리 ─────────────────
    if "cf_bd50" in cache.columns:
        shg = group_recenter(cache, "cf_bd50", "cf_hpo", "cf_bd50_lvl0")
        print(f"[C] cf_bd50 그룹별 레벨 제거: " +
              " ".join(f"g{int(g)} {v:+.4f}" for g, v in shg.items()), flush=True)
        for w in (0.65, 0.70, 0.75):
            res[f"bd50Xcln_w{int(w*100)}"] = ev("cf_bd50", "dl_cln03", w, PP_NEW)
            res[f"bd50LVL0Xcln_w{int(w*100)}"] = ev("cf_bd50_lvl0", "dl_cln03", w, PP_NEW)

    print_table(res, ref)

    # ── 요약: 이동량 대비 ΔCV ────────────────────────────
    print("\n[요약] 블렌드 평균 이동 대비 ΔCV (w0.65 기준)", flush=True)
    print(f"{'셀':<20}{'블렌드 평균이동(%cap)':>22}{'ΔCV':>10}{'Δn':>9}{'Δf':>9}", flush=True)
    base_blend = 0.65 * cache.loc[mm, "cf_hpo"] + 0.35 * cache.loc[mm, "dl_cln03"]
    def report(name, gcol, dcol, w=0.65):
        if name not in res:
            return
        b = w * cache.loc[mm, gcol] + (1 - w) * cache.loc[mm, dcol]
        r = res[name]
        print(f"{name:<20}{(b.mean()-base_blend.mean())*100:>+22.3f}"
              f"{r['cv'][0]-ch_cv:>+10.4f}{r['cv'][1]-ref['cv'][1]:>+9.4f}"
              f"{r['cv'][2]-ref['cv'][2]:>+9.4f}", flush=True)
    report(f"off{int(round(d_hpo*1000)):03d}_w65", "cf_hpo", f"dl_off{int(round(d_hpo*1000)):03d}")
    report("hpo_w65", "cf_hpo", "dl_hpo")
    report("hpoLVL0_w65", "cf_hpo", "dl_hpo_lvl0")
    if "cf_bd50" in cache.columns:
        report("bd50Xcln_w65", "cf_bd50", "dl_cln03")
        report("bd50LVL0Xcln_w65", "cf_bd50_lvl0", "dl_cln03")

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
