# ═══════════════════════════════════════════════════════
# exp-042: exp-040 base × exp-039 DL 교차 — 상보 프로파일 결합 (재학습 0)
#
# 배경: exp-040·exp-039가 각각 **점수의 반대쪽 절반**에서 이득을 내고 반대쪽에서 잃어
#   총점이 상쇄됐다 (현 챔피언 g132+DLspd w0.70: 1-NMAE 0.8824 / FiCR 0.4072 기준):
#     exp-040 gspd_w70·cB : 1-NMAE 0.8837 (+0.0013) / FiCR 0.4058 (−0.0014)   ← NMAE 쪽
#     exp-039 aw_w60      : 1-NMAE 0.8818 (−0.0006) / FiCR 0.4117 (+0.0045)   ← FiCR 쪽
#   두 축이 독립이면 교차(gspd base × aw/q60 DL)가 양쪽을 동시에 가질 수 있다.
#   두 구성요소 전부 parquet 캐시에 있으므로 **재학습 없이** 블렌드·후처리·채점만 하면 된다.
#
# 주의(다중비교): 이 스크립트는 수십 칸을 한 번에 채점한다. 칸이 많으면 노이즈로 양 게이트를
#   통과하는 셀이 우연히 나올 수 있으므로, 판정은 (a) 양 게이트 +0.001 초과,
#   (b) fold 부호 일관성, (c) 두 지표 동반 상승(전이 성공군 프로파일) 3중 조건으로 한다.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp042_cross.py
# ═══════════════════════════════════════════════════════
import argparse
import pickle
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import load_gbm_cache, postprocess, score_from_tidy, FOLD_LIST
from exp039_dl_diverse import DL_PRED_CACHE, build_slices
from exp040_gbm_spd import GBM_CACHE, BIAS_CACHE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

KEY = ["fold", "group", "forecast_kst_dtm"]
GBMS = ["g132", "gspd"]                                  # gspdG는 exp-040에서 홀드아웃 음수 → 제외
DLS = ["hub", "aw", "q60", "aw+q60", "hub+aw+q60"]
WS = [0.75, 0.70, 0.65, 0.60]
REF = ("g132", "hub", 0.70)                              # 현 LB 챔피언 구성


def load_all():
    champ, champ_bias = load_gbm_cache()
    cache = champ.rename(columns={"gbm_cf": "cf_g132"})

    ext = pd.read_parquet(GBM_CACHE)
    with open(BIAS_CACHE, "rb") as f:
        ext_bias = pickle.load(f)
    for c in [c for c in ext.columns if c.startswith("cf_")]:
        m = cache[KEY].merge(ext[KEY + [c]], on=KEY, how="left")
        assert m[c].notna().all(), f"{c} 정렬 실패"
        cache[c] = m[c].to_numpy()

    dl = pd.read_parquet(DL_PRED_CACHE)
    base_dls = [c[3:] for c in dl.columns if c.startswith("dl_")]
    for c in [f"dl_{v}" for v in base_dls]:
        m = cache[KEY].merge(dl[KEY + [c]], on=KEY, how="left")
        assert m[c].notna().all(), f"{c} 정렬 실패"
        cache[c] = m[c].to_numpy()
    # 혼합 DL
    for name in DLS:
        if "+" in name:
            cache[f"dl_{name}"] = np.mean([cache[f"dl_{v}"].to_numpy() for v in name.split("+")], axis=0)
    return cache, champ_bias, ext_bias, base_dls


def run(args):
    cache, champ_bias, ext_bias, base_dls = load_all()
    print(f"캐시 결합: {cache.shape} | GBM {[c[3:] for c in cache.columns if c.startswith('cf_')]} | "
          f"DL {base_dls}", flush=True)
    fold_names = [fn for fn, _ in FOLD_LIST]
    hold = HOLD_FOLD["name"]
    cv_folds = [fn for fn in fold_names if fn != hold]
    slices = build_slices(cache)

    def score(gbm, dl, w):
        parts = []
        for (fname, group), sub in slices:
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            cf = w * sub[f"cf_{gbm}"].to_numpy() + (1 - w) * sub[f"dl_{dl}"].to_numpy()
            # 편향맵은 exp-040에서 챔피언 coarse 맵(cB)이 전 칸 우세 → 챔피언 맵 고정
            pred = postprocess(cf, cap, sub["ws"].to_numpy(), sub["pA"].to_numpy(), group, champ_bias[fname])
            parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
        tidy = pd.concat(parts, ignore_index=True)
        return dict(cv=score_from_tidy(tidy[tidy["fold"] != hold], "pred"),
                    hold=score_from_tidy(tidy[tidy["fold"] == hold], "pred"),
                    folds={fn: score_from_tidy(tidy[tidy["fold"] == fn], "pred")[0] for fn in cv_folds})

    res, t0 = {}, time.time()
    for g in GBMS:
        for d in DLS:
            for w in WS:
                res[(g, d, w)] = score(g, d, w)
    print(f"{len(res)}칸 채점 {time.time()-t0:.0f}s", flush=True)

    r0 = res[REF]
    b_cv, b_hd = r0["cv"][0], r0["hold"][0]
    n0, f0 = r0["cv"][1], r0["cv"][2]
    print("\n" + "=" * 104)
    print(f"기준 챔피언 {REF}: CV {b_cv:.4f} (1-NMAE {n0:.4f} FiCR {f0:.4f}) | HOLD {b_hd:.4f}"
          f"   ※ exp-035 로그 0.6448/0.6434와 일치해야 재현 OK")
    print("=" * 104)
    print(f"{'base':<6}{'DL':<14}{'w':>5}{'CV':>8}{'ΔCV':>9}{'HOLD':>8}{'Δhold':>9}"
          f"{'1-NMAE':>9}{'Δnmae':>9}{'FiCR':>8}{'Δficr':>9}  판정")

    rows = []
    for (g, d, w), r in sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - b_cv, kv[1]["hold"][0] - b_hd)):
        dcv, dhd = r["cv"][0] - b_cv, r["hold"][0] - b_hd
        dn, df = r["cv"][1] - n0, r["cv"][2] - f0
        fold_d = [r["folds"][fn] - r0["folds"][fn] for fn in cv_folds]
        # 3중 조건: 양 게이트 +0.001 초과 / fold 부호 일관(음수 fold 없음) / 두 지표 동반 상승
        mag = dcv > 0.001 and dhd > 0.001
        cons = all(x > -0.0002 for x in fold_d)
        both = dn > 0 and df > 0
        verdict = ("★3중충족" if (mag and cons and both) else
                   ("통과(매그니튜드)" if mag else ("양게이트" if (dcv > 0 and dhd > 0) else "")))
        print(f"{g:<6}{d:<14}{w:>5.2f}{r['cv'][0]:>8.4f}{dcv:>+9.4f}{r['hold'][0]:>8.4f}{dhd:>+9.4f}"
              f"{r['cv'][1]:>9.4f}{dn:>+9.4f}{r['cv'][2]:>8.4f}{df:>+9.4f}  {verdict}")
        rows.append(dict(gbm=g, dl=d, w=w, dcv=dcv, dhold=dhd, dnmae=dn, dficr=df,
                         mag=mag, cons=cons, both=both, folds=fold_d))

    # 상위 후보 fold 분해
    top = sorted(rows, key=lambda r: -min(r["dcv"], r["dhold"]))[:8]
    print(f"\n--- 상위 8칸 fold별 ΔCV (챔피언 대비) ---")
    print(f"{'base·DL·w':<26}" + "".join(f"{fn.replace('fold_',''):>10}" for fn in cv_folds)
          + "   fold일관  두지표")
    for r in top:
        print(f"{r['gbm']+'·'+r['dl']+'·'+format(r['w'],'.2f'):<26}"
              + "".join(f"{x:>+10.4f}" for x in r["folds"])
              + f"{'  O' if r['cons'] else '  X':>9}{'      O' if r['both'] else '      X'}")

    # 오차 상관 (교차가 탈상관을 어떻게 바꾸는지)
    caps = cache["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    acf = cache["target"].to_numpy() / caps
    ev = cache["target"].to_numpy() >= caps * 0.10
    print("\n--- GBM×DL 오차 상관 (평가대상 시간, 보정 전) ---")
    M = pd.DataFrame({g: {d: round(float(np.corrcoef(cache[f"cf_{g}"].to_numpy()[ev] - acf[ev],
                                                     cache[f"dl_{d}"].to_numpy()[ev] - acf[ev])[0, 1]), 4)
                          for d in DLS} for g in GBMS})
    print(M.to_string())
    return res


def main():
    global GBMS, DLS, WS
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbms", type=str, default=",".join(GBMS))
    ap.add_argument("--dls", type=str, default=",".join(DLS))
    ap.add_argument("--ws", type=str, default=",".join(str(w) for w in WS))
    args = ap.parse_args()
    GBMS = [v for v in args.gbms.split(",") if v]
    DLS = [v for v in args.dls.split(",") if v]
    WS = [float(w) for w in args.ws.split(",") if w]
    t0 = time.time()
    print("### exp-042: exp-040 base × exp-039 DL 교차 (재학습 없음) ###", flush=True)
    run(args)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
