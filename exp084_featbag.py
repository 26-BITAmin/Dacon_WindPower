# ═══════════════════════════════════════════════════════
# exp-084: exp-083 noL의 **통제** — 소스 정렬이 원인인가, 그냥 피처 부분집합이면 되는가
#
# exp-083/b/c 결과: noL(LDAPS 계열 197개 제거, 155피처)을 챔피언 GBM과 42:58로 섞고
#   w0.80으로 DL과 블렌드하면 ΔCV +0.0021 / Δhold +0.0013 (양 매그니튜드 통과·
#   Δn +0.0006·Δf +0.0037 동반·LOFO +0.0009·평균 이동 +0.041%cap 중립).
#
# 그러나 "예보 소스를 뺐다"는 해석은 아직 통제되지 않았다. 대안 가설:
#   피처 수를 줄인 **아무 부분집합**이라도 오차가 탈상관돼 같은 이득이 난다
#   (= 피처 부분공간 배깅. LightGBM feature_fraction과 달리 멤버 단위로 고정).
#
# 통제 설계: **같은 크기(155)의 무작위 부분집합** 2개를 동일 프로토콜로 학습.
#   - rndA만으로 noL과 같은 이득 → 원인은 소스가 아니라 부분공간 다양성
#     (그렇다면 다음 라운드는 무작위 부분집합 K개 배깅으로 확장하는 쪽이 옳다)
#   - rndA가 뚜렷이 열등 → 소스 정렬이 실제 원인 (noL 축을 그대로 심화)
#   - rndA ⊕ rndB 평균은 "부분집합 여러 개"의 가산성도 같이 측정
#
# 판정 Δ = 보드 챔피언(hpo×dC_w65). 나머지 전부 챔피언 고정.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp084_featbag.py
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
from exp083_srcsplit import partition, variant_cols

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEEDS = {"rndA": 0, "rndB": 1}          # 무작위 부분집합의 재현 시드
SHARES = [(2 / 3, "t67"), (0.58, "t58"), (0.50, "t50")]
WS = [0.75, 0.80, 0.85]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    print("### exp-084 무작위 부분집합 통제 (noL의 원인 분리) ###", flush=True)
    train_df, use_cols = build_train_frame()
    fam = partition(use_cols)
    n_keep = len(variant_cols(fam, "noL"))
    print(f"입력 준비 완료 ({len(use_cols)}피처) — noL과 동일 크기 {n_keep}로 통제",
          flush=True)
    e56 = json.loads(E56_JSON.read_text())
    cache, champ_bias = load_cache_full()
    assert "cf_noL" in merge_cached_cols(cache, CACHE_DIR / "exp083_gbm_noL.parquet",
                                         ["cf_noL"]), "exp-083 noL 캐시 없음"

    for tag, sd in SEEDS.items():
        col = f"cf_{tag}"
        cpath = CACHE_DIR / f"exp084_gbm_{tag}.parquet"
        ppath = CACHE_DIR / f"exp084_gbm_{tag}_partial.parquet"
        if col in merge_cached_cols(cache, cpath, [col]):
            continue
        cols = list(np.random.RandomState(sd).choice(use_cols, n_keep, replace=False))
        fams = pd.Series([k for c in cols for k, v in fam.items() if c in v]).value_counts()
        print(f"\n[{tag}] {len(cols)}피처 무작위 (시드 {sd}) — 계열 구성 {dict(fams)}",
              flush=True)
        out = np.full(len(cache), np.nan)
        if ppath.exists():
            prev = pd.read_parquet(ppath)
            out = cache[KEY].merge(prev[KEY + [col]], on=KEY, how="left")[col].to_numpy()
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
            print(f"  [{tag}:{fname}] {time.time()-tf:.0f}s", flush=True)
        cache[col] = out
        save_cols(cache, cpath, [col])

    cache["cf_rndAB"] = 0.5 * cache["cf_rndA"] + 0.5 * cache["cf_rndB"]
    cache["cf_noLrnd"] = 0.5 * cache["cf_noL"] + 0.5 * cache["cf_rndA"]

    mm = cache["target"].notna().to_numpy()
    print("", flush=True)
    for c in ("cf_noL", "cf_rndA", "cf_rndB", "cf_rndAB", "cf_noLrnd"):
        v, h = cache[c].to_numpy(), cache["cf_hpo"].to_numpy()
        print(f"  corr({c:<10}, cf_hpo) = {np.corrcoef(v[mm], h[mm])[0,1]:.4f} | "
              f"평균 이동 {(v-h).mean()*100:+.3f}%cap", flush=True)
    print(f"  corr(cf_rndA, cf_rndB) = "
          f"{np.corrcoef(cache['cf_rndA'].to_numpy()[mm], cache['cf_rndB'].to_numpy()[mm])[0,1]:.4f}",
          flush=True)

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언 재현: CV {ref['cv'][0]:.4f} | HOLD {ref['hold'][0]:.4f}  (기대 0.6556/0.6535)",
          flush=True)

    res = {}
    for base in ("noL", "rndA", "rndAB", "noLrnd"):
        for m, mt in SHARES:
            mixcol = f"mx_{base}_{mt}"
            cache[mixcol] = m * cache["cf_hpo"] + (1 - m) * cache[f"cf_{base}"]
            for w in WS:
                res[f"{base}{mt}_w{int(w*100)}"] = ev(mixcol, "dl_cln03", w, PP_NEW)
    print_table(res, ref)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
