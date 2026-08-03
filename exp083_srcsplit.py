# ═══════════════════════════════════════════════════════
# exp-083: 예보 소스 분할 GBM 멤버 — "정보 비대칭"으로 탈상관을 강제
#
# 배경 (기전 논거):
#   - exp-035(+0.0051 LB, 세션 최대)의 이득 기전은 **오차 탈상관**이었고, exp-040은 같은
#     정보를 GBM에도 주자 corr 0.818→0.840으로 오르며 블렌드 총점이 ±0이 됨을 실증했다.
#     → 이 프로젝트에서 실제로 값을 낸 유일한 구조는 **멤버 간 정보 비대칭**.
#   - 반대로 "같은 정보 위 다른 함수족"은 전부 수렴: HGB corr 0.9985(exp-065),
#     파라미터 변형 0.999(exp-060), pcinv 타깃 교체 0.9974(PC-01), TabPFN 0.9839(exp-069).
#     corr 축에서 유일하게 **미탐색으로 남은 레버 = 정보를 줄이는 것**.
#   - 기상 다중모델 앙상블의 표준 논거: 서로 다른 NWP의 오차는 부분적으로 독립.
#     현재 GBM 1개가 352피처(LDAPS·GFS·JMA·EC)를 전부 보므로 그 오차는 가장 신뢰하는
#     소스의 오차에 지배된다. 소스를 뺀 멤버는 **다른 NWP의 오차 구조**를 갖는다.
#
# 반대 증거 (사전 기록):
#   - P2 "단독 정확도 ≠ 파트너 가치"가 5회 실증 — 소스 제한 멤버는 단독으로는 반드시 열세.
#   - exp-069는 corr 0.9839(실제 다양성)에도 홀드아웃 단조 악화로 기각.
#   판정은 게이트에 맡긴다. 사전 기준: corr(cf_var, cf_hpo) < 0.99 면 다양성 실재.
#
# 변형 2개 (GBM만 — DL·후처리·분류기·블렌드 w 전부 챔피언 고정):
#   noL : LDAPS 계열 전부 제거 (GFS+JMA+EC+달력) — 주력 소스를 뺀 멤버
#   noG : GFS 계열 전부 제거 (LDAPS+JMA+EC+달력)
#   ※ 두 소스를 섞어 만든 파생(phy_*·불일치도 등)은 **양쪽 변형에서 모두 제거**해야
#     "소스 제한"이 성립하므로 mixed 블록으로 분류해 뺀다.
#
# 게이트: 챔피언 GBM 하이퍼(e56)·cln03 가중·3-seed·4-fold+홀드아웃,
#   혼합 사다리 cf_hpo⊕cf_var {5/6, 3/4, 2/3} × dl_cln03 w{0.65, 0.70} + 단독.
#   판정 Δ = 보드 챔피언(hpo×dC_w65, 재현 기대 CV 0.6556 / HOLD 0.6535).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp083_srcsplit.py
# ═══════════════════════════════════════════════════════
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from exp035_raw_dl import FOLD_LIST, CACHE_DIR
from exp049_clean import KEY, PP_NEW
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp056_hpo import RESULT_JSON as E56_JSON, train_gate_fold
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp065_gbm3 import build_train_frame

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VARIANTS = ["noL", "noG"]


# 이름에 소스가 안 드러나는 파생의 실제 입력 (exp005_qopt·exp012_feature_v11 정의 확인분)
MIXED_NAMES = {"ws_disagreement",        # |ldaps_ws_10m − gfs_ws_10m|
               "wd_agreement",           # ldaps 풍향 · gfs 풍향 내적
               "disagreement_x_season"}  # ws_disagreement × 계절
L_NAMES = {"ws_x_season"}                # ldaps_ws_10m × 계절
SHARED_NAMES = {"high_variance_season"}  # 달력만


def classify(col):
    """피처를 소스 계열로 분류. mixed = 두 소스를 함께 쓴 파생 → 양 변형에서 제외."""
    c = col.lower()
    if c in MIXED_NAMES:
        return "mixed"
    if c in L_NAMES:
        return "L"
    if c in SHARED_NAMES:
        return "S"
    is_l = ("ldaps" in c) or c.startswith("sdl_") or c.startswith("spa_l") or c.startswith("tmp_l")
    is_g = ("gfs" in c) or c.startswith("sdg_") or c.startswith("spa_g") or c.startswith("tmp_g")
    if is_l and is_g:
        return "mixed"
    if is_l:
        return "L"
    if is_g:
        return "G"
    if c.startswith("tmp_day"):        # ldaps_ws_10m 일 단위 집계 (exp012 line 84)
        return "L"
    if ("jgsm" in c) or ("jmsm" in c):
        return "J"
    if c.startswith("ec_") or c.startswith("ens_"):
        return "E"
    # phy_* 는 ldaps·gfs 원변수를 함께 씀 (exp012 line 50~64) → mixed
    if c.startswith("phy_"):
        return "mixed"
    return "S"          # shared: 달력 파생 등 (달력 자체는 prepare_matrices가 별도 주입)


def partition(cols):
    fam = {}
    for c in cols:
        fam.setdefault(classify(c), []).append(c)
    return fam


def variant_cols(fam, tag):
    """소스 제한 컬럼 목록. mixed는 양쪽 모두 제외(제한의 무결성)."""
    keep = ["S", "J", "E"]
    keep += ["G"] if tag == "noL" else ["L"]
    out = []
    for k in keep:
        out += fam.get(k, [])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="noL,noG")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    tags = [t.strip() for t in args.variants.split(",") if t.strip()]
    for t in tags:
        assert t in VARIANTS, f"미정의 변형: {t}"

    t0 = time.time()
    print("### exp-083 예보 소스 분할 GBM 멤버 ###", flush=True)
    train_df, use_cols = build_train_frame()
    fam = partition(use_cols)
    print(f"입력 준비 완료 ({len(use_cols)}피처) — {time.time()-t0:.0f}s", flush=True)
    print("  소스 분해: " + " / ".join(
        f"{k}={len(v)}" for k, v in sorted(fam.items())), flush=True)
    for k in sorted(fam):
        print(f"    [{k}] 예: {fam[k][:4]}", flush=True)
    e56 = json.loads(E56_JSON.read_text())
    cache, champ_bias = load_cache_full()

    for tag in tags:
        col = f"cf_{tag}"
        cpath = CACHE_DIR / f"exp083_gbm_{tag}.parquet"
        ppath = CACHE_DIR / f"exp083_gbm_{tag}_partial.parquet"
        if col in merge_cached_cols(cache, cpath, [col]):
            continue
        cols = variant_cols(fam, tag)
        print(f"\n[{tag}] {len(cols)}피처 (제거 {len(use_cols)-len(cols)}개)", flush=True)
        out = np.full(len(cache), np.nan)
        if ppath.exists():                      # 폴드 단위 재개 (exp-068/PC-05 선례)
            prev = pd.read_parquet(ppath)
            out = cache[KEY].merge(prev[KEY + [col]], on=KEY, how="left")[col].to_numpy()
            done = [fn for fn, _ in FOLD_LIST
                    if not np.isnan(out[(cache["fold"] == fn).to_numpy()]).any()]
            print(f"  partial 재개: 완료 폴드 {done}", flush=True)
        for fname, fold in FOLD_LIST:
            sel = (cache["fold"] == fname).to_numpy()
            if not np.isnan(out[sel]).any():
                continue
            tf = time.time()
            p = train_gate_fold(train_df, cols, fold, args.seeds, args.jobs,
                                e56["lgb"], e56["lgb_n"], e56["xgb"], e56["xgb_n"])
            m = cache.loc[sel, KEY].merge(p.rename(columns={"cf_hpo": col}), on=KEY, how="left")
            assert m[col].notna().all(), f"정렬 실패 {fname}"
            out[sel] = m[col].to_numpy()
            cache[col] = out
            cache[KEY + [col]].to_parquet(ppath, index=False)
            print(f"  [{tag}:{fname}] {time.time()-tf:.0f}s (partial 저장)", flush=True)
        cache[col] = out
        save_cols(cache, cpath, [col])
        print(f"  캐시 저장 → {cpath.name}", flush=True)

    # ── 진단: 다양성이 실재하는가 ──
    mm = cache["target"].notna().to_numpy()
    print("", flush=True)
    for tag in tags:
        col = f"cf_{tag}"
        v, h, d = (cache[col].to_numpy(), cache["cf_hpo"].to_numpy(),
                   cache["dl_cln03"].to_numpy())
        print(f"  corr({col}, cf_hpo) = {np.corrcoef(v[mm], h[mm])[0,1]:.4f} | "
              f"corr({col}, dl_cln03) = {np.corrcoef(v[mm], d[mm])[0,1]:.4f} | "
              f"평균 이동 {(v - h).mean()*100:+.3f}%cap", flush=True)

    # ── 게이트 ──
    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} (1-NMAE {ref['cv'][1]:.4f} "
          f"FiCR {ref['cv'][2]:.4f}) | HOLD {ref['hold'][0]:.4f}  (기대 0.6556/0.6535)", flush=True)

    res = {"hpo_solo": ev("cf_hpo", None, 1.0, PP_NEW)}
    for tag in tags:
        col = f"cf_{tag}"
        res[f"{tag}_solo"] = ev(col, None, 1.0, PP_NEW)
        for m, mt in ((5 / 6, "t83"), (0.75, "t75"), (2 / 3, "t67")):
            mixcol = f"mix_{tag}_{mt}"
            cache[mixcol] = m * cache["cf_hpo"] + (1 - m) * cache[col]
            for w in (0.65, 0.70):
                res[f"{tag}{mt}_w{int(w*100)}"] = ev(mixcol, "dl_cln03", w, PP_NEW)
    print_table(res, ref)
    for tag in tags:
        d_cv = res[f"{tag}_solo"]["cv"][0] - res["hpo_solo"]["cv"][0]
        d_hd = res[f"{tag}_solo"]["hold"][0] - res["hpo_solo"]["hold"][0]
        print(f"GBM 단독 델타 ({tag}−hpo): ΔCV {d_cv:+.4f} / Δhold {d_hd:+.4f}")
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
