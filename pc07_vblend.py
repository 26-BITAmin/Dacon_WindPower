# ═══════════════════════════════════════════════════════
# PC-07 (R7): v-공간 블렌딩 — 블렌드를 CF 공간이 아니라 풍속 공간에서
#
# 근거: 트랙 파이프라인의 블렌드는 지금까지 전부 CF 공간 산술평균이었다
#   (v1 = 0.65·PC(v̂) + 0.35·dl_cln03). 파워커브는 어깨에서 볼록·정격 근처에서
#   오목이므로 Jensen에 의해 두 공간의 블렌드는 비동치 —
#   v-공간 블렌드 PC(w·v̂ + (1−w)·PC⁻¹(dl))는 특히 급경사 구간(5~9m/s,
#   손실의 57%)에서 다른 값을 낸다. 트랙 7부품 중 블렌드 공간은 미시도 축.
#
# 판정 (트랙 규칙): Δ = v1 (pcinv×dC_w65 CF공간, CV 0.6530/HOLD 0.6512).
#   [통제] 같은 v-공간 블렌드를 챔피언 쌍(cf_hpo×dl_cln03)에도 적용 —
#   이득이 트랙 고유(타깃·커브)의 것인지 일반 후처리 트릭인지 분리 (PC-04 규칙).
#
# 재학습 0 — 커브는 SCADA에서 fold-safe로 재구축(결정론), 멤버는 전부 캐시.
# 실행: C:\miniconda3\python.exe -X utf8 pc07_vblend.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np

from exp015_scada_pc import build_scada_hourly
from exp035_raw_dl import CACHE_DIR, FOLD_LIST
from exp049_clean import PP_NEW
from exp050_lambda_sweep import merge_cached_cols
from exp060_pergroup_hpo import load_cache_full, make_ev
from exp065_gbm3 import build_train_frame, slice_fold
from pc01_pcinv import build_curve, make_transform

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WS = (0.55, 0.60, 0.65, 0.70, 0.75)
CTRL_WS = (0.65,)


def main():
    t0 = time.time()
    print("### PC-07: v-공간 블렌딩 (재학습 0, 커브 재구축) ###", flush=True)
    cache, champ_bias = load_cache_full()
    assert merge_cached_cols(cache, CACHE_DIR / "exp082_gbm_pcinv.parquet", ["cf_pcinv"])
    merge_cached_cols(cache, CACHE_DIR / "pc05_dlvc.parquet", ["dl_dlvc"])

    train_df, _ = build_train_frame()
    scada_h = build_scada_hourly()
    print(f"준비 완료 ({time.time()-t0:.0f}s)", flush=True)

    # ── fold×그룹 커브 재구축(v1과 동일 규약: q=None bin 평균) → v-공간 블렌드 열 생성 ──
    new_cols = {}
    def ensure(col):
        if col not in new_cols:
            new_cols[col] = np.full(len(cache), np.nan)

    fold_by_name = dict(FOLD_LIST)
    for fname, fold in FOLD_LIST:
        tr_all, _ = slice_fold(train_df, fold)
        for group in (1, 2, 3):
            tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()]
            curve = build_curve(scada_h[group], tr_g, fold["train_end"], None)
            if curve is None:
                continue
            fwd, inv, _, _ = make_transform(curve)
            sel = ((cache["fold"] == fname) & (cache["group"] == group)).to_numpy()
            v_pc = inv(cache.loc[sel, "cf_pcinv"].to_numpy())
            v_dc = inv(cache.loc[sel, "dl_cln03"].to_numpy())
            v_hp = inv(cache.loc[sel, "cf_hpo"].to_numpy())
            v_vc = inv(cache.loc[sel, "dl_dlvc"].to_numpy()) if "dl_dlvc" in cache.columns else None
            for w in WS:
                col = f"vb_dC_w{int(w*100)}"
                ensure(col); new_cols[col][sel] = fwd(w * v_pc + (1 - w) * v_dc)
                if v_vc is not None:
                    col = f"vb_dlvc_w{int(w*100)}"
                    ensure(col); new_cols[col][sel] = fwd(w * v_pc + (1 - w) * v_vc)
            for w in CTRL_WS:
                col = f"CTRLvb_hpo_w{int(w*100)}"
                ensure(col); new_cols[col][sel] = fwd(w * v_hp + (1 - w) * v_dc)
        print(f"  [{fname}] 커브·변환 완료 ({time.time()-t0:.0f}s)", flush=True)

    for col, vals in new_cols.items():
        assert not np.isnan(vals).any(), f"{col} 미채움 행 존재"
        cache[col] = vals

    # ── 채점 ──────────────────────────────────────────────
    ev = make_ev(cache, champ_bias)
    v1 = ev("cf_pcinv", "dl_cln03", 0.65, PP_NEW)
    champ = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    v1_cv, v1_hd = v1["cv"][0], v1["hold"][0]
    cv_folds = list(v1["folds"].keys())
    print(f"\n트랙 v1 재현: CV {v1_cv:.4f} | HOLD {v1_hd:.4f}  (기대 0.6530/0.6512)", flush=True)
    print(f"챔피언 재현:  CV {champ['cv'][0]:.4f} | HOLD {champ['hold'][0]:.4f}", flush=True)

    res = {}
    for col in sorted(new_cols):
        res[col] = ev(col, None, 1.0, PP_NEW)
    # CF-공간 대응 쌍 (참고)
    for w in WS:
        res[f"cf_dC_w{int(w*100)}"] = ev("cf_pcinv", "dl_cln03", w, PP_NEW)

    print(f"\n{'셀':<18}{'CV':>8}{'Δv1':>9}{'HOLD':>8}{'Δv1':>9}{'Δn':>9}{'Δf':>9}  fold(Δv1)", flush=True)
    for name, r in sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - v1_cv,
                                                           kv[1]["hold"][0] - v1_hd)):
        fd = " ".join(f"{r['folds'][fn]-v1['folds'][fn]:+.4f}" for fn in cv_folds)
        print(f"{name:<18}{r['cv'][0]:>8.4f}{r['cv'][0]-v1_cv:>+9.4f}"
              f"{r['hold'][0]:>8.4f}{r['hold'][0]-v1_hd:>+9.4f}"
              f"{r['cv'][1]-v1['cv'][1]:>+9.4f}{r['cv'][2]-v1['cv'][2]:>+9.4f}  {fd}", flush=True)

    # 통제 행 해석 (챔피언 대비)
    for w in CTRL_WS:
        c = res[f"CTRLvb_hpo_w{int(w*100)}"]
        print(f"\n[통제] 챔피언 쌍 v-공간 w{w:.2f}: 챔피언(CF공간) 대비 "
              f"ΔCV {c['cv'][0]-champ['cv'][0]:+.4f} / Δhold {c['hold'][0]-champ['hold'][0]:+.4f}", flush=True)

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
