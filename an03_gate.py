# ═══════════════════════════════════════════════════════
# AN-03: AnEn 4-fold 게이트 — AN-02 상위 구성의 정식 평가
#
# 구성 (전부 hub 예측자·hw=3·filter 가용성 정제):
#   an_m100 : K=100 평균   (AN-02 스모크 MAE 최선 0.12571)
#   an_m200 : K=200 평균   (K 안정성 대조)
#   an_q60  : K=500 q60    (상향 프로파일, signed +0.011)
#
# 평가: make_ev 사다리 — 단독 / ×dl_cln03 w65 / 챔피언(cf_hpo)과 cf-공간 혼합
# 판정: 트랙 자체 기준(AN-01 대비), 챔피언 대비는 정보로만.
#
# 실행: C:\miniconda3\python.exe -X utf8 an03_gate.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from an01_anen import PRED_SETS
from an02_clean import build_analogs2, variants_from, predict
from exp035_raw_dl import FOLD_LIST, CACHE_DIR
from exp049_clean import KEY, PP_NEW
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp065_gbm3 import build_train_frame, corr_on_eval

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CONFIGS = [("an_m100", "filter", 100, "mean"),
           ("an_m200", "filter", 200, "mean"),
           ("an_q60", "filter", 500, "q60")]


def main():
    t0 = time.time()
    print("### AN-03 AnEn 4-fold 게이트 ###", flush=True)
    cache, champ_bias = load_cache_full()
    cols_needed = [f"cf_{t}" for t, *_ in CONFIGS]
    cpath = CACHE_DIR / "an03_gate.parquet"
    have = merge_cached_cols(cache, cpath, cols_needed)
    if not all(c in have for c in cols_needed):
        train_df, _ = build_train_frame()
        print(f"입력 준비 완료 — {time.time()-t0:.0f}s", flush=True)
        preds = {c: [] for c in cols_needed}
        for fname, fold in FOLD_LIST:
            tf = time.time()
            for group in [1, 2, 3]:
                ana = build_analogs2(train_df, fold, group, PRED_SETS["hub"], 3)
                tabs = variants_from(ana)
                base = dict(fold=fold["name"], group=group)
                tm = ana["va_g"]["forecast_kst_dtm"].to_numpy()
                for tag, avm, K, pm in CONFIGS:
                    cf = predict(tabs[avm], K, pm)
                    preds[f"cf_{tag}"].append(pd.DataFrame(
                        {**base, "forecast_kst_dtm": tm, f"cf_{tag}": cf}))
            print(f"  [{fname}] {time.time()-tf:.0f}s", flush=True)
        for c in cols_needed:
            p = pd.concat(preds[c], ignore_index=True)
            m = cache[KEY].merge(p, on=KEY, how="left")
            assert m[c].notna().all(), f"정렬 실패: {c}"
            cache[c] = m[c].to_numpy()
        save_cols(cache, cpath, cols_needed)
        print(f"  캐시 저장 → {cpath.name}", flush=True)

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} (1-NMAE {ref['cv'][1]:.4f} "
          f"FiCR {ref['cv'][2]:.4f}) | HOLD {ref['hold'][0]:.4f}", flush=True)
    res = {}
    hpov = cache["cf_hpo"].to_numpy()
    for tag, *_ in CONFIGS:
        col = f"cf_{tag}"
        for b in ("cf_hpo", "dl_cln03"):
            print(f"  corr({col}, {b}) = {corr_on_eval(cache, col, b):.4f}", flush=True)
        print(f"  평균 이동 ({col} − cf_hpo): "
              f"{(cache[col].to_numpy() - hpov).mean()*100:+.3f}%cap", flush=True)
        res[f"{tag}_solo"] = ev(col, None, 1.0, PP_NEW)
        res[f"{tag}_w65"] = ev(col, "dl_cln03", 0.65, PP_NEW)
        for m_, mt in ((0.9, "t90"), (5 / 6, "t83"), (0.75, "t75"), (2 / 3, "t67")):
            mix = f"mx_{tag}_{mt}"
            cache[mix] = m_ * hpov + (1 - m_) * cache[col].to_numpy()
            res[f"{tag}{mt}_w65"] = ev(mix, "dl_cln03", 0.65, PP_NEW)
    print_table(res, ref)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
