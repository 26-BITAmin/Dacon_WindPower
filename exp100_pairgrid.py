# ═══════════════════════════════════════════════════════
# exp-100: 캐시 전 멤버 **완전 쌍 교차** (GBM × DL × w, 재학습 0)
#
# 근거: 이 프로젝트 역대 최대 상승 exp-042(+0.0052)는 **새 학습이 아니라 교차**였다.
#   exp-040(GBM 격자편차: NMAE만 개선, 블렌드 총점 ±0)과
#   exp-039(DL q60: FiCR만 개선, 게이트 미달) — 둘 다 **상대를 챔피언으로 고정**한
#   단독 판정에서는 기각/무개선이었는데, 서로 교차하니 두 지표가 동시에 유지됐다.
#
#   그 이후 exp-045~097의 모든 멤버 실험은 다시 **상대 멤버를 챔피언 고정**한 채 판정했다.
#   → 캐시에 남은 GBM 멤버 × DL 멤버 조합 대부분이 **한 번도 채점된 적 없음**.
#   exp-098이 ASOS 4×4만 덮었으므로(0셀 통과), 여기서 **전 멤버**로 확장한다.
#
# 방법: 재학습 0. external/*.parquet 의 cf_*/dl_* 중 전 폴드 완전한 컬럼만 사용.
#   1단계 — 전 쌍 × w0.65
#   2단계 — 1단계 상위/양수 쌍만 w 사다리 {0.55,0.60,0.70,0.75}
#
# 다중비교 주의: 셀이 수백 개이므로 "ΔCV 최고"만으로 채택하지 않는다.
#   채택 후보 = ΔCV>+0.001 **및** 4폴드 전부 양수 **및** Δhold>0.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp100_pairgrid.py
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

WS_LADDER = (0.55, 0.60, 0.70, 0.75)
W_BASE = 0.65
# 파일명에 이게 들어가면 제외 (부분 캐시 / 테스트셋 예측 / 원자료)
SKIP_PAT = re.compile(r"(_partial|test_|^kma_|^tmp_)")


def discover(cache):
    """external/*.parquet 을 훑어 cf_*/dl_* 완전 컬럼을 cache 에 붙인다."""
    gbms, dls, dupes = [], [], []
    for path in sorted(Path(CACHE_DIR).glob("*.parquet")):
        if SKIP_PAT.search(path.name):
            continue
        try:
            names = pq.ParquetFile(path).schema.names
        except Exception:
            continue
        wanted = [c for c in names if c.startswith(("cf_", "dl_")) and c not in cache.columns]
        already = [c for c in names if c.startswith(("cf_", "dl_")) and c in cache.columns]
        dupes += [(path.name, c) for c in already]
        if not wanted:
            continue
        for c in merge_cached_cols(cache, path, wanted):
            (gbms if c.startswith("cf_") else dls).append(c)
    return gbms, dls, dupes


def main():
    t0 = time.time()
    print("### exp-100: 전 멤버 쌍 교차 (재학습 0) ###", flush=True)
    cache, champ_bias = load_cache_full()          # cf_cln03/dl_cln03/cf_ec/cf_hpo 포함
    base_g = [c for c in cache.columns if c.startswith("cf_")]
    base_d = [c for c in cache.columns if c.startswith("dl_")]
    add_g, add_d, dupes = discover(cache)
    gbms = base_g + add_g
    dls = base_d + add_d
    print(f"\nGBM 멤버 {len(gbms)}개: {gbms}", flush=True)
    print(f"DL  멤버 {len(dls)}개: {dls}", flush=True)
    if dupes:
        print(f"(중복 컬럼 무시: {len(dupes)}건 — 먼저 붙인 파일 우선)", flush=True)

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", W_BASE, PP_NEW)
    ch_cv, ch_hd = ref["cv"][0], ref["hold"][0]
    print(f"\n챔피언 재현: CV {ch_cv:.4f} (1-NMAE {ref['cv'][1]:.4f} FiCR {ref['cv'][2]:.4f})"
          f" | HOLD {ch_hd:.4f}  (기대 0.6556/0.6535)", flush=True)

    mm = cache["target"].notna()

    # ── 1단계: 전 쌍 × w0.65 ─────────────────────────────
    n_cells = len(gbms) * len(dls)
    print(f"\n[1단계] 전 쌍 {len(gbms)}×{len(dls)} = {n_cells}셀 (w={W_BASE})", flush=True)
    stage1, done = {}, 0
    for g in gbms:
        for d in dls:
            stage1[(g, d)] = ev(g, d, W_BASE, PP_NEW)
            done += 1
            if done % 100 == 0:
                print(f"  ...{done}/{n_cells} ({(time.time()-t0)/60:.1f}분)", flush=True)

    ranked = sorted(stage1.items(), key=lambda kv: -kv[1]["cv"][0])
    print(f"\n--- 1단계 상위 20 (w0.65, ΔCV 순) ---", flush=True)
    print(f"{'GBM':<12}{'DL':<12}{'CV':>8}{'ΔCV':>9}{'HOLD':>8}{'Δhd':>9}{'corr':>8}", flush=True)
    for (g, d), r in ranked[:20]:
        c = np.corrcoef(cache.loc[mm, g], cache.loc[mm, d])[0, 1]
        print(f"{g:<12}{d:<12}{r['cv'][0]:>8.4f}{r['cv'][0]-ch_cv:>+9.4f}"
              f"{r['hold'][0]:>8.4f}{r['hold'][0]-ch_hd:>+9.4f}{c:>8.4f}", flush=True)

    # ── 2단계: 유망 쌍만 w 사다리 ────────────────────────
    promising = [k for k, r in ranked[:30]]
    promising += [k for k, r in stage1.items()
                  if r["cv"][0] > ch_cv and r["hold"][0] > ch_hd and k not in promising]
    print(f"\n[2단계] 유망 쌍 {len(promising)}개 × w {WS_LADDER} "
          f"= {len(promising)*len(WS_LADDER)}셀", flush=True)

    res = {f"{g[3:]}x{d[3:]}_w{int(W_BASE*100)}": stage1[(g, d)] for g, d in promising}
    for g, d in promising:
        for w in WS_LADDER:
            res[f"{g[3:]}x{d[3:]}_w{int(w*100)}"] = ev(g, d, w, PP_NEW)

    top = dict(sorted(res.items(),
                      key=lambda kv: -min(kv[1]["cv"][0] - ch_cv,
                                          kv[1]["hold"][0] - ch_hd))[:22])
    print_table(top, ref)

    # ── 채택 후보 판정 (다중비교 방어) ───────────────────
    cv_folds = list(ref["folds"].keys())
    def fold_all_pos(r):
        return all(r["folds"][fn] - ref["folds"][fn] > 0 for fn in cv_folds)

    all_cells = dict(res)
    all_cells.update({f"{g[3:]}x{d[3:]}_w65": r for (g, d), r in stage1.items()})
    strict = [(k, v) for k, v in all_cells.items()
              if v["cv"][0] - ch_cv > 0.001 and v["hold"][0] - ch_hd > 0 and fold_all_pos(v)]
    magonly = [(k, v) for k, v in all_cells.items() if v["cv"][0] - ch_cv > 0.001]

    print(f"\n[채택 판정] 전 {len(all_cells)}셀", flush=True)
    print(f"  ΔCV>+0.001                     : {len(magonly)}셀", flush=True)
    print(f"  + Δhold>0 + 4폴드 전부 양수     : {len(strict)}셀  ← 채택 후보", flush=True)
    for k, v in sorted(strict, key=lambda kv: -kv[1]["cv"][0])[:10]:
        fd = " ".join(f"{v['folds'][fn]-ref['folds'][fn]:+.4f}" for fn in cv_folds)
        print(f"    {k:<26} ΔCV {v['cv'][0]-ch_cv:+.4f} / Δhold {v['hold'][0]-ch_hd:+.4f}"
              f" | Δn {v['cv'][1]-ref['cv'][1]:+.4f} Δf {v['cv'][2]-ref['cv'][2]:+.4f} | {fd}", flush=True)
    if not strict and magonly:
        print("  (매그니튜드만 통과한 셀 — 폴드/홀드 조건 미충족)", flush=True)
        for k, v in sorted(magonly, key=lambda kv: -kv[1]["cv"][0])[:10]:
            fd = " ".join(f"{v['folds'][fn]-ref['folds'][fn]:+.4f}" for fn in cv_folds)
            print(f"    {k:<26} ΔCV {v['cv'][0]-ch_cv:+.4f} / Δhold {v['hold'][0]-ch_hd:+.4f} | {fd}", flush=True)

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
