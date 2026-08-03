# ═══════════════════════════════════════════════════════
# exp-116: GBM 3멤버 **심플렉스 전탐색** — 교체(swap)를 포함한 전 배합 (재학습 0)
#
# exp-115의 설계 결함. 거기서 돌린 사다리는 전부 `t·cf_hpo + (1−t)·cf_cat` 형태라
#   **LGB와 XGB를 항상 같은 비율로 묶은 채 둘을 똑같이 희석**하기만 했다:
#     mix83 = LGB 41.5 / XGB 41.5 / Cat 17 ...  mix50 = LGB 25 / XGB 25 / Cat 50
#   즉 **"XGB를 빼고 CatBoost를 넣는다"** 같은 교체 셀은 한 칸도 측정되지 않았다.
#   (사용자 지적: "기존 두 종류에서 switch 같은 걸 안 해보고 단독 catboost랑 그냥 비교한 건가")
#
# 교체가 의미 있을 수 있는 이유: exp-065 실측 **corr(cf_lgb, cf_xgb) = 0.9978** — 현직 두 멤버는
#   거의 같은 예측이다. 반면 **corr(cf_cat, cf_hpo) = 0.9877**로 CatBoost가 더 멀다.
#   그렇다면 LGB의 파트너로 XGB보다 CatBoost가 나을 수 있다 — 이건 희석 사다리로는 못 재는 방향이다.
#
# 세 멤버 예측이 전부 캐시에 있으므로 **재학습 0**:
#   cf_lgb·cf_xgb (`exp065_gbm_members.parquet`) · cf_cat (`exp115_gbm_cat.parquet`)
#
# 격자: (a,b,c) = (LGB, XGB, Cat), a+b+c=1, 1/12 간격 전탐색 = 91셀 × DL w{0.65, 0.70}.
#   명명 셀 — LX(0.5/0.5/0 = 현직 cf_hpo) · **LC(0.5/0/0.5 = XGB 교체)** ·
#            **XC(0/0.5/0.5 = LGB 교체)** · L · X · C(단독).
#
# 다중비교 방어: 91셀에서 최댓값을 고르는 것이므로 **양수 셀이 나오면 LOFO 필수**
#   (exp-088에서 54칸 최댓값 +0.0008이 LOFO −0.0012로 뒤집힌 전례).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp116_gbm_simplex.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp035_raw_dl import CACHE_DIR, HOLD_FOLD
from exp049_clean import PP_NEW
from exp050_lambda_sweep import merge_cached_cols
from exp060_pergroup_hpo import load_cache_full, make_ev

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STEP = 12                                    # 1/12 간격 → 91셀
WS = (0.65, 0.70)
NAMED = {(6, 6, 0): "LX(현직)", (6, 0, 6): "LC(XGB교체)", (0, 6, 6): "XC(LGB교체)",
         (12, 0, 0): "L(단독)", (0, 12, 0): "X(단독)", (0, 0, 12): "C(단독)",
         (4, 4, 4): "LXC(균등)"}
TMP = "cf_tmp"


def main():
    t0 = time.time()
    print("### exp-116: GBM 3멤버 심플렉스 전탐색 (교체 포함, 재학습 0) ###", flush=True)
    cache, champ_bias = load_cache_full()
    assert merge_cached_cols(cache, CACHE_DIR / "exp065_gbm_members.parquet", ["cf_lgb", "cf_xgb"])
    assert merge_cached_cols(cache, CACHE_DIR / "exp115_gbm_cat.parquet", ["cf_cat"])

    # ── 무결성: LGB·XGB 50:50이 현직 cf_hpo와 일치하는가 ──
    recon = 0.5 * cache["cf_lgb"].to_numpy() + 0.5 * cache["cf_xgb"].to_numpy()
    dmax = float(np.nanmax(np.abs(recon - cache["cf_hpo"].to_numpy())))
    # 캐시가 float32라 왕복 오차 ~1e-8이 정상. 1e-9은 너무 빡빡한 문턱이었음.
    print(f"[무결성] LGB:XGB 50:50 재조합 vs cf_hpo 최대차 {dmax:.2e} "
          f"({'일치(float32 반올림 수준)' if dmax < 1e-6 else '★불일치'})", flush=True)

    # ── 멤버 프로파일 (평가 마스크에서 — exp-115의 교훈) ──
    lab = cache["target"].notna().to_numpy()
    cap = cache["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    acf = cache["target"].to_numpy() / cap
    evm = lab & (acf >= 0.10)
    print(f"\n[멤버 프로파일 — **평가행**(실측 ≥10%cap, {int(evm.sum()):,}행) 기준]", flush=True)
    print(f"  {'멤버':<10}{'평가행 MAE':>12}{'평가행 signed':>15}{'전체행 MAE':>12}", flush=True)
    for c in ("cf_lgb", "cf_xgb", "cf_hpo", "cf_cat"):
        e = cache[c].to_numpy() - acf
        print(f"  {c:<10}{np.abs(e[evm]).mean():>12.5f}{e[evm].mean()*100:>+14.3f}%"
              f"{np.abs(e[lab]).mean():>12.5f}", flush=True)
    print(f"\n[상관 (평가행)]", flush=True)
    for a, b in (("cf_lgb", "cf_xgb"), ("cf_lgb", "cf_cat"), ("cf_xgb", "cf_cat"),
                 ("cf_hpo", "cf_cat")):
        print(f"  corr({a}, {b}) = {np.corrcoef(cache.loc[evm, a], cache.loc[evm, b])[0,1]:.4f}",
              flush=True)

    # ── 심플렉스 전탐색 ──
    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언 재현: CV {ref['cv'][0]:.4f} | HOLD {ref['hold'][0]:.4f} (기대 0.6556/0.6535)",
          flush=True)
    L, X, C = (cache[c].to_numpy() for c in ("cf_lgb", "cf_xgb", "cf_cat"))
    rows = []
    for i in range(STEP + 1):
        for j in range(STEP + 1 - i):
            k = STEP - i - j
            cache[TMP] = (i * L + j * X + k * C) / STEP
            for w in WS:
                r = ev(TMP, "dl_cln03", w, PP_NEW)
                rows.append(dict(i=i, j=j, k=k, w=w,
                                 cv=r["cv"][0], hold=r["hold"][0],
                                 n=r["cv"][1], f=r["cv"][2],
                                 folds=[r["folds"][fn] for fn in sorted(r["folds"])]))
    df = pd.DataFrame(rows)
    df["dcv"] = df["cv"] - ref["cv"][0]
    df["dhd"] = df["hold"] - ref["hold"][0]
    df["dn"] = df["n"] - ref["cv"][1]
    df["df_"] = df["f"] - ref["cv"][2]
    df["name"] = [NAMED.get((r.i, r.j, r.k), f"{r.i}/{r.j}/{r.k}") for r in df.itertuples()]
    print(f"\n총 {len(df)}셀 채점 ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n═══ 명명 셀 (LGB/XGB/Cat 비율) ═══", flush=True)
    print(f"  {'셀':<14}{'w':>6}{'CV':>9}{'ΔCV':>9}{'Δhold':>9}{'Δn':>9}{'Δf':>9}", flush=True)
    for key, nm in NAMED.items():
        for w in WS:
            s = df[(df.i == key[0]) & (df.j == key[1]) & (df.k == key[2]) & (df.w == w)]
            if len(s):
                r = s.iloc[0]
                print(f"  {nm:<14}{w:>6.2f}{r.cv:>9.4f}{r.dcv:>+9.4f}{r.dhd:>+9.4f}"
                      f"{r.dn:>+9.4f}{r.df_:>+9.4f}", flush=True)

    print(f"\n═══ 전 91셀 상위 12 (ΔCV) ═══", flush=True)
    print(f"  {'LGB/XGB/Cat':<14}{'w':>6}{'ΔCV':>9}{'Δhold':>9}{'Δn':>9}{'Δf':>9}  폴드", flush=True)
    for r in df.sort_values("dcv", ascending=False).head(12).itertuples():
        fl = " ".join(f"{x - y:+.4f}" for x, y in zip(r.folds, sorted(ref["folds"].values(),
                      key=lambda _: 0) if False else [ref["folds"][fn] for fn in sorted(ref["folds"])]))
        print(f"  {r.i}/{r.j}/{r.k:<10}{r.w:>6.2f}{r.dcv:>+9.4f}{r.dhd:>+9.4f}"
              f"{r.dn:>+9.4f}{r.df_:>+9.4f}  {fl}", flush=True)

    best = df.sort_values("dcv", ascending=False).iloc[0]
    print(f"\n═══ 판정 ═══", flush=True)
    print(f"  91셀 최댓값 = {best.i}/{best.j}/{best.k} @ w{best.w:.2f}: "
          f"ΔCV {best.dcv:+.4f} / Δhold {best.dhd:+.4f}", flush=True)
    if best.dcv <= 0:
        print("  → **전 셀 음수. 교체 축도 닫힘** — 어떤 배합도 현직 LGB:XGB 50:50을 못 넘음.",
              flush=True)
    else:
        print("  → 양수 셀 존재. **91셀 최댓값이므로 LOFO 필수** (exp-088 전례: 54칸 최댓값 "
              "+0.0008 → LOFO −0.0012).", flush=True)
        folds = sorted(ref["folds"])
        tot = 0.0
        for fi, fn in enumerate(folds):
            oth = df.assign(o=[np.mean([f for x, f in enumerate(r.folds) if x != fi])
                               for r in df.itertuples()])
            pick = oth.sort_values("o", ascending=False).iloc[0]
            tot += pick.folds[fi] - ref["folds"][fn]
            print(f"    [{fn}] 타폴드 선택 {pick.i}/{pick.j}/{pick.k}@w{pick.w:.2f} "
                  f"→ 해당폴드 Δ {pick.folds[fi]-ref['folds'][fn]:+.4f}", flush=True)
        print(f"  **LOFO 평균 Δ = {tot/len(folds):+.4f}**", flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
