# ═══════════════════════════════════════════════════════
# exp-107: `eff` 가중을 **DL 쪽에** — 가용성 축의 마지막 미측정 칸
#
# 배경. `eff`(유효출력률) 활용 경로를 지금까지 5개 측정했고 전부 음수다:
#   exp-110 GBM 가중 −0.0036 / exp-111B GBM 피처 −0.0001 / exp-111C 후처리 LOFO −0.0087
#   / 2요인 구조분해 −0.0179 / 풍향섹터 후처리 LOFO −0.0106.
# **그런데 다섯 경로 전부 GBM 또는 후처리다. DL은 한 번도 eff를 받은 적이 없다.**
#
# 왜 DL이 다른가:
#   - `exp049_clean.py:121` — DL은 지금 `avail_factor(avail_strict, "cln03")`으로 가중된다.
#     그 지표는 EDA 12절에서 **g3 오라벨**로 확정됐다("완전가동" g3 시간의 66%가 실제 eff<0.8).
#   - 역할 분담이 다르다: GBM=1-NMAE 엔진 / **DL=FiCR 엔진**(q60 상측 익스펙타일).
#     점수는 FiCR에서 갈리고, 최대 손실 칸(exp-108: 0.117)이 g3다.
#   → **FiCR 담당 멤버가 FiCR 손실 최대 그룹에서 틀린 라벨로 가중되고 있다.**
#
# exp-110의 설계 결함을 여기서 고친다. 그때는 세 가지가 한꺼번에 바뀌었다 —
#   ① 지표 교체 ② 공격성 급증(g3 만가중 86.8%→52.5%) ③ 구조 변경(cln03 중간 티어가 비어버림).
#   그래서 "라벨이 나빠서 진 것"인지 "너무 세게 깎아서 진 것"인지 귀속이 불가능했다.
#
# 변형 2종:
#   effq — **그룹별 분위수 정합**. 각 그룹에서 cln03이 avail_strict에 부여하는 감량 *비율*을
#          그대로 eff 순위에 이식한다. 감량 대상 **행 수는 동일**, 라벨만 교체.
#          → ①만 격리. 이게 주 판정 대상.
#   effc — 임계값 없는 **연속** 가중 w = clip((eff−0.4)/0.6, 0.05, 1). 눈대중 컷 제거.
#          → 임계값 설계 자체가 문제였을 가능성에 대한 대조.
#
# 가드 (exp-110이 걸린 함정 재발 방지):
#   - **평균 이동(%cap)** — 상향은 판별 규칙상 게이트 통과와 무관하게 LB 역전군.
#   - Δn/Δf 분리 — FiCR만 오르고 NMAE가 죽는 형태를 즉시 식별.
#   - **L0 변형** — exp-102의 LOFO 레벨 제거를 붙여, 이득이 정보인지 레벨인지 분리.
#
# 판정 주기준 = CV 4-fold Δ (사용자 지시 2026-08-02: 홀드아웃·corr은 참고).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp107_dl_eff.py [--seeds 3]
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp035_raw_dl import build_inputs, FOLD_LIST, CACHE_DIR, DL_CFG
from exp049_clean import PP_NEW, merge_avail, avail_factor
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp110_effweight import EFF_PARQUET
from exp102_delevel import delevel_lofo

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EFF_CACHE = CACHE_DIR / "exp107_dl_eff.parquet"
PARTIAL = CACHE_DIR / "exp107_dl_eff_partial.parquet"
WS = (0.60, 0.65, 0.70)
SCHEMES = ("effq", "effc")


# ── 가중 팩터 ──────────────────────────────────────────
def qmatch_factor(eff, avail, groups, verbose=False):
    """그룹별로 cln03의 **감량 비율**을 eff 순위에 이식 (공격성 고정, 라벨만 교체).

    cln03: avail<0.9 → ×0.05,  0.9≤avail<0.999 → ×0.3,  else ×1.0
    여기 : eff 하위 p1 → ×0.05, 그다음 p2 → ×0.3, 나머지 ×1.0  (p1·p2는 그룹별 cln03 실측 비율)
    """
    a = np.nan_to_num(np.asarray(avail, dtype=float), nan=1.0)
    e = np.nan_to_num(np.asarray(eff, dtype=float), nan=1.0)   # eff 결측 = 열화 미측정 → 만가중 쪽
    out = np.ones(len(e))
    for g in np.unique(groups):
        m = groups == g
        p1 = float((a[m] < 0.9).mean())
        p2 = float((a[m] < 0.999).mean()) - p1
        r = pd.Series(e[m]).rank(pct=True, method="first").to_numpy()
        out[m] = np.where(r <= p1, 0.05, np.where(r <= p1 + p2, 0.3, 1.0))
        if verbose:
            print(f"    g{int(g)}: cln03 비율 ×0.05 {p1*100:5.1f}% · ×0.3 {p2*100:5.1f}% "
                  f"→ eff 컷 {np.quantile(e[m], p1):.3f} / {np.quantile(e[m], p1+p2):.3f}", flush=True)
    return out


def cont_factor(eff):
    """임계값 없는 연속 가중. eff 0.4 이하 → 0.05, 1.0 → 1.0 선형."""
    e = np.nan_to_num(np.asarray(eff, dtype=float), nan=1.0)
    return np.clip((e - 0.40) / 0.60, 0.05, 1.0)


def weight_factor(sub, scheme, verbose=False):
    grp = sub["kpx_group"].to_numpy()
    if scheme == "cln03":
        return avail_factor(sub["avail_strict"].to_numpy(), "cln03")
    if scheme == "effq":
        return qmatch_factor(sub["eff"].to_numpy(), sub["avail_strict"].to_numpy(), grp, verbose)
    if scheme == "effc":
        return cont_factor(sub["eff"].to_numpy())
    raise ValueError(scheme)


# ── DL 학습 (exp049 dl_fold_predict_clean과 동일, 가중만 교체) ──
def make_xy_eff(df, cols, imputer, scaler, scheme, fit=False, verbose=False):
    from exp033_dl import GIDX
    sub = df.loc[df["target"].notna()]
    caps = sub["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    Xraw = sub[cols].to_numpy(dtype=np.float32)
    Xs = (scaler.fit_transform(imputer.fit_transform(Xraw)) if fit
          else scaler.transform(imputer.transform(Xraw)))
    g = sub["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    cf = (sub["target"].to_numpy() / caps).astype(np.float32)
    w = np.where(cf < 0.10, 0.05, 1.0) * weight_factor(sub, scheme, verbose)
    return Xs.astype(np.float32), g, cf, w.astype(np.float32)


def dl_fold_predict_eff(tr_df, va_df, cols, seeds, scheme, verbose=False):
    import torch
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from exp033_dl import predict, GIDX
    from exp039_dl_diverse import make_loss, train_one
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sca = StandardScaler()
    Xtr, gtr, ytr, wtr = make_xy_eff(tr_df, cols, imp, sca, scheme, fit=True, verbose=verbose)
    Xva = sca.transform(imp.transform(va_df[cols].to_numpy(dtype=np.float32))).astype(np.float32)
    gva = va_df["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    loss_fn = make_loss("q60")
    preds = []
    for s in range(seeds):
        m = train_one(Xtr, gtr, ytr, wtr, DL_CFG, loss_fn, s)
        preds.append(predict(m, Xva, gva))
        del m
        torch.cuda.empty_cache()
    return np.mean(preds, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-107: eff 가중을 DL 쪽에 (가용성 축 마지막 미측정 칸) ###", flush=True)

    train_df, _, colsets = build_inputs(["spd"])
    vcols = colsets["spd"]
    train_df = merge_avail(train_df)
    eff = (pd.read_parquet(EFF_PARQUET)[["forecast_kst_dtm", "group", "eff"]]
           .rename(columns={"group": "kpx_group"}))
    n0 = len(train_df)
    train_df = train_df.merge(eff, on=["forecast_kst_dtm", "kpx_group"], how="left")
    assert len(train_df) == n0, "eff 병합에서 행 증식"
    lab = train_df["target"].notna()
    print(f"입력 {len(vcols)}피처 | 학습행 {int(lab.sum()):,} | eff 결측 "
          f"{train_df.loc[lab, 'eff'].isna().mean()*100:.1f}% (ws<5 등 → 만가중 처리)", flush=True)

    print("\n[가중 설계 — 공격성 정합 확인]", flush=True)
    sub = train_df.loc[lab]
    for sch in ("cln03",) + SCHEMES:
        f = weight_factor(sub, sch, verbose=(sch == "effq"))
        line = " ".join(f"g{g}={f[sub['kpx_group'].to_numpy()==g].mean():.3f}" for g in (1, 2, 3))
        print(f"  {sch:<6} 평균 팩터 {f.mean():.3f} | {line} | 만가중(=1.0) 비율 "
              f"{(f >= 0.999).mean()*100:.1f}%", flush=True)

    cache, champ_bias = load_cache_full()
    cols_new = [f"dl_{s}" for s in SCHEMES]
    have = merge_cached_cols(cache, EFF_CACHE, cols_new)
    if PARTIAL.exists():
        have += [c for c in merge_cached_cols(cache, PARTIAL, cols_new) if c not in have]

    for sch in SCHEMES:
        col = f"dl_{sch}"
        if col in have and cache[col].notna().all():
            print(f"\n[{col}] 캐시 재사용", flush=True)
            continue
        out = cache[col].to_numpy().copy() if col in cache.columns else np.full(len(cache), np.nan)
        for fname, fold in FOLD_LIST:
            sel = (cache["fold"] == fname).to_numpy()
            if not np.isnan(out[sel]).any():
                continue
            tf = time.time()
            tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
            keys = cache.loc[sel, ["group", "forecast_kst_dtm"]]
            va = keys.merge(train_df, left_on=["group", "forecast_kst_dtm"],
                            right_on=["kpx_group", "forecast_kst_dtm"], how="left")
            assert len(va) == len(keys), "정렬 실패"
            out[sel] = dl_fold_predict_eff(tr_all, va, vcols, args.seeds, sch)
            cache[col] = out
            save_cols(cache, PARTIAL, [c for c in cols_new if c in cache.columns])
            print(f"  [{col}:{fname}] {time.time()-tf:.0f}s", flush=True)
        cache[col] = out
        if col not in have:
            have.append(col)
    save_cols(cache, EFF_CACHE, cols_new)

    # ── 레벨 제거 변형 (이득이 정보인가 상향 이동인가) ──
    for sch in SCHEMES:
        delevel_lofo(cache, f"dl_{sch}", "dl_cln03", f"dl_{sch}_L0")

    mm = cache["target"].notna().to_numpy()
    print("\n[멤버 프로파일 — 상향 이동은 판별 규칙상 LB 역전군]", flush=True)
    print(f"  {'멤버':<14}{'corr(cf_hpo)':>14}{'평균이동 vs cln03':>20}", flush=True)
    for c in ["dl_cln03"] + [f"dl_{s}" for s in SCHEMES] + [f"dl_{s}_L0" for s in SCHEMES]:
        r = np.corrcoef(cache.loc[mm, c], cache.loc[mm, "cf_hpo"])[0, 1]
        sh = (cache[c].to_numpy() - cache["dl_cln03"].to_numpy())[mm].mean() * 100
        print(f"  {c:<14}{r:>14.4f}{sh:>+19.3f}%cap", flush=True)

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언 재현: CV {ref['cv'][0]:.4f} | HOLD {ref['hold'][0]:.4f} (기대 0.6556/0.6535)",
          flush=True)
    res = {}
    for sch in SCHEMES:
        for tag in (f"dl_{sch}", f"dl_{sch}_L0"):
            res[f"{tag}_solo"] = ev(tag, None, 1.0, PP_NEW)
            for w in WS:
                res[f"{tag}_w{int(w*100)}"] = ev("cf_hpo", tag, w, PP_NEW)
    res["dl_cln03_solo"] = ev("dl_cln03", None, 1.0, PP_NEW)
    print_table(res, ref)

    solo0 = res["dl_cln03_solo"]["cv"][0]
    print("\n[DL 단독 ΔCV — 챔피언 DL 대비] (멤버 자체 품질)", flush=True)
    for sch in SCHEMES:
        print(f"  dl_{sch}: {res[f'dl_{sch}_solo']['cv'][0]-solo0:+.4f}"
              f"  | L0 {res[f'dl_{sch}_L0_solo']['cv'][0]-solo0:+.4f}", flush=True)
    print("\n※ effq는 cln03과 **감량 행 수가 같다** — 차이는 오직 '어느 행을 깎았는가'(라벨 품질).", flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
