# ═══════════════════════════════════════════════════════
# exp-103: exp-102 후보의 **탐색 폴드(Q4) 제외** 재판정 + 분해
#
# 결정적 우려: `dl_hpo`는 exp-058에서 **목적함수 = Q4 블렌드 Score**로 25트라이얼
#   Optuna 선택된 멤버다. 즉 CV 4폴드 중 Q4는 **선택에 쓰인 폴드**이므로
#   그 폴드의 이득은 선택 편향을 포함한다. exp-060/062가 정확히 이걸로 기각됐다
#   ("Q4만 +0.005~7, 타 폴드 −0.004~10").
#   exp-102 결과에서도 Q4가 최대 폴드였다(hpoL0_w75: Q1 +0.0027 Q2 +0.0012
#   Q3 −0.0009 **Q4 +0.0048**).
#
# 그래서 **Q4를 뺀 pooled 점수**로 다시 잰다 (폴드 델타 평균이 아니라 진짜 pooled).
#   Q1+Q2+Q3 pooled Δ가 게이트(+0.001)를 유지하면 선택 편향만으로는 설명 안 됨.
#   무너지면 exp-060/062와 같은 처분.
#
# 추가: 그룹별 분해 · 블렌드 평균 이동 확인 · 레벨 제거 추정치의 견고성.
#
# 재학습 0 · 캐시 전용.
# 실행: C:\miniconda3\python.exe -X utf8 exp103_q4free.py
# ═══════════════════════════════════════════════════════
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from exp005_qopt import CAPACITY_KWH
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import CACHE_DIR, FOLD_LIST, score_from_tidy
from exp039_dl_diverse import build_slices
from exp049_clean import PP_NEW, postprocess_flex
from exp050_lambda_sweep import merge_cached_cols
from exp060_pergroup_hpo import load_cache_full

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SKIP_PAT = re.compile(r"(_partial|test_|^kma_|^tmp_)")
HOLD = HOLD_FOLD["name"]
CV_FOLDS = [fn for fn, _ in FOLD_LIST if fn != HOLD]
SEARCH_FOLD = "fold_2024Q4"          # exp-058 Optuna 목적함수 폴드
REF_DL = "dl_cln03"


def attach(cache, prefix="dl_"):
    for path in sorted(Path(CACHE_DIR).glob("*.parquet")):
        if SKIP_PAT.search(path.name):
            continue
        try:
            names = pq.ParquetFile(path).schema.names
        except Exception:
            continue
        want = [c for c in names if c.startswith(prefix) and c not in cache.columns]
        if want:
            merge_cached_cols(cache, path, want)


def delevel_lofo(cache, src, ref, out):
    mm = cache["target"].notna().to_numpy()
    fold, grp = cache["fold"].to_numpy(), cache["group"].to_numpy()
    d = cache[src].to_numpy() - cache[ref].to_numpy()
    vals = cache[src].to_numpy().copy()
    for f in np.unique(fold):
        for g in np.unique(grp):
            other = mm & (fold != f) & (grp == g)
            if other.sum():
                vals[(fold == f) & (grp == g)] -= d[other].mean()
    cache[out] = vals


def make_tidy(cache, champ_bias):
    slices = build_slices(cache)

    def tidy_of(gcol, dcol, w, pp=PP_NEW):
        blend = w * cache[gcol].to_numpy() + (1 - w) * cache[dcol].to_numpy()
        parts = []
        for (fname, group), sub in slices:
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            pred = postprocess_flex(blend[sub.index], cap, sub["ws"].to_numpy(),
                                    sub["pA"].to_numpy(), group, champ_bias[fname], **pp)
            parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
        return pd.concat(parts, ignore_index=True)
    return tidy_of


def main():
    t0 = time.time()
    print("### exp-103: Q4(탐색 폴드) 제외 재판정 ###", flush=True)
    cache, champ_bias = load_cache_full()
    attach(cache)
    delevel_lofo(cache, "dl_hpo", REF_DL, "dl_hpo_L0")
    delevel_lofo(cache, "dl_hpob", REF_DL, "dl_hpob_L0")
    for frac in (0.75,):
        cache[f"dl_mixL0_{int(frac*100)}"] = (frac * cache["dl_hpo_L0"].to_numpy()
                                              + (1 - frac) * cache[REF_DL].to_numpy())
    tidy_of = make_tidy(cache, champ_bias)

    CELLS = [
        ("champ",      "cf_hpo", REF_DL,          0.65),
        ("hpoL0_w65",  "cf_hpo", "dl_hpo_L0",     0.65),
        ("hpoL0_w70",  "cf_hpo", "dl_hpo_L0",     0.70),
        ("hpoL0_w75",  "cf_hpo", "dl_hpo_L0",     0.75),
        ("hpo_w75",    "cf_hpo", "dl_hpo",        0.75),
        ("mix75_w75",  "cf_hpo", "dl_mixL0_75",   0.75),
        ("hpobL0_w70", "cf_hpo", "dl_hpob_L0",    0.70),
    ]
    tidies = {n: tidy_of(g, d, w) for n, g, d, w in CELLS}
    base = tidies["champ"]

    def sc(t, mask):
        return score_from_tidy(t[mask(t)], "pred")

    cv_all = lambda t: t["fold"] != HOLD
    cv_noq4 = lambda t: (t["fold"] != HOLD) & (t["fold"] != SEARCH_FOLD)
    only_q4 = lambda t: t["fold"] == SEARCH_FOLD
    hold_m = lambda t: t["fold"] == HOLD

    print(f"\n{'셀':<12}{'CV(4폴드)':>11}{'Δ':>9}"
          f"{'CV−Q4(3폴드)':>14}{'Δ':>9}{'Q4만':>9}{'Δ':>9}{'HOLD':>9}{'Δ':>9}", flush=True)
    b = {k: sc(base, m)[0] for k, m in
         dict(a=cv_all, n=cv_noq4, q=only_q4, h=hold_m).items()}
    rows = {}
    for n, *_ in CELLS:
        t = tidies[n]
        v = {k: sc(t, m)[0] for k, m in
             dict(a=cv_all, n=cv_noq4, q=only_q4, h=hold_m).items()}
        rows[n] = v
        print(f"{n:<12}{v['a']:>11.4f}{v['a']-b['a']:>+9.4f}"
              f"{v['n']:>14.4f}{v['n']-b['n']:>+9.4f}"
              f"{v['q']:>9.4f}{v['q']-b['q']:>+9.4f}"
              f"{v['h']:>9.4f}{v['h']-b['h']:>+9.4f}", flush=True)

    print(f"\n[해석 기준] Q4 제외 pooled Δ가 +0.001 이상이면 선택 편향만으로 설명 불가", flush=True)

    # ── 폴드별 분해 (Score는 전 그룹 pooled로만 정의되므로 그룹 분해는 하지 않음) ──
    print(f"\n[폴드별 Δ — 어느 분기가 이득을 만드는가]", flush=True)
    print(f"{'셀':<12}" + "".join(f"{fn.replace('fold_',''):>10}" for fn in CV_FOLDS)
          + "  탐색폴드(Q4) 비중", flush=True)
    for n, *_ in CELLS:
        if n == "champ":
            continue
        d = []
        for fn in CV_FOLDS:
            mf = lambda t, fn=fn: t["fold"] == fn
            d.append(sc(tidies[n], mf)[0] - sc(base, mf)[0])
        share = d[-1] / sum(x for x in d) if sum(d) else float("nan")
        print(f"{n:<12}" + "".join(f"{x:>+10.4f}" for x in d) + f"{share:>18.0%}", flush=True)

    # ── 1-NMAE / FiCR 프로파일 (Q4 제외) ────────────────
    print(f"\n[프로파일 — Q4 제외 3폴드 pooled]", flush=True)
    bn, bf = sc(base, cv_noq4)[1], sc(base, cv_noq4)[2]
    print(f"{'셀':<12}{'1-NMAE':>10}{'Δn':>9}{'FiCR':>9}{'Δf':>9}{'평균이동%cap':>13}", flush=True)
    mm = cache["target"].notna()
    bm = (0.65 * cache.loc[mm, "cf_hpo"] + 0.35 * cache.loc[mm, REF_DL]).mean()
    for n, g, d, w in CELLS:
        s = sc(tidies[n], cv_noq4)
        shift = ((w * cache.loc[mm, g] + (1 - w) * cache.loc[mm, d]).mean() - bm) * 100
        print(f"{n:<12}{s[1]:>10.4f}{s[1]-bn:>+9.4f}{s[2]:>9.4f}{s[2]-bf:>+9.4f}"
              f"{shift:>+13.3f}", flush=True)

    # ── 레벨 제거 추정기 견고성 ──────────────────────────
    # 결론이 "레벨을 어떻게 뺐는가"에 의존하면 안 된다. 3가지 추정기 비교.
    print(f"\n[추정기 견고성] 레벨 제거 방식별 (w0.65 = 변경점 1개 = DL 멤버 교체만)", flush=True)
    mm = cache["target"].notna().to_numpy()
    fold, grp = cache["fold"].to_numpy(), cache["group"].to_numpy()
    d = cache["dl_hpo"].to_numpy() - cache[REF_DL].to_numpy()

    v = cache["dl_hpo"].to_numpy() - d[mm].mean()            # 전역 단일 상수
    cache["dl_L0_glob"] = v
    v = cache["dl_hpo"].to_numpy().copy()                    # 그룹별 (폴드 무관)
    for g in np.unique(grp):
        v[grp == g] -= d[mm & (grp == g)].mean()
    cache["dl_L0_grp"] = v

    for nm, col in [("LOFO(그룹×폴드)", "dl_hpo_L0"),
                    ("전역 단일 상수", "dl_L0_glob"),
                    ("그룹별(폴드 무관)", "dl_L0_grp")]:
        t = tidy_of("cf_hpo", col, 0.65)
        a, n_, h = sc(t, cv_all), sc(t, cv_noq4), sc(t, hold_m)
        shift = ((0.65 * cache.loc[mm, "cf_hpo"] + 0.35 * cache.loc[mm, col]).mean() - bm) * 100
        print(f"  {nm:<18} ΔCV(4폴드) {a[0]-b['a']:+.4f} | ΔCV(−Q4) {n_[0]-b['n']:+.4f}"
              f" | Δhold {h[0]-b['h']:+.4f} | Δn {n_[1]-bn:+.4f} | 이동 {shift:+.3f}%cap", flush=True)

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
