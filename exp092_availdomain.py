# ═══════════════════════════════════════════════════════
# exp-092: 정지 예측 — 도메인 피처 vs 전체 361 (exp-091 후속)
#
# 문제 제기 (사용자): "361개 다 쓰는 게 맞나? 터빈 작동에 영향 주는 요인을 먼저 생각해야."
#   타당하다. exp-091의 타깃은 **희소 사건**(양성 ~10%)이라 신호대잡음이 본 모델보다
#   훨씬 나쁘다. 게다가 352 중 **216(격자 편차)은 세 그룹이 동일 값**인 공간 풍속 구조라
#   "터빈이 서 있는가"와 물리적 연결이 약하고, exp-091 상위에 뜬 `day`(월중 며칠)는
#   노이즈 적합 정황이다.
#
# 도메인 정리 — 풍력터빈이 서는 이유:
#   ① 고풍속 컷아웃(>25m/s)  → 허브풍속·돌풍·일최대          [예보 가능]
#   ② 착빙(T≈−8~+2℃ + 고습)  → 기온·이슬점·습도·강설          [예보 가능]
#   ③ 뇌우/낙뢰              → 강수율·운량                    [부분]
#   ④ 계통 출력제한          → 시간·요일·주말 (수요 패턴)     [부분]
#   ⑤ 계획 정비              → 월·요일 (일정성)               [부분]
#   ⑥ 무작위 고장            → 없음                            [불가]
#
# 변형:
#   all   : exp-091과 동일한 361 (동일 런 대조)
#   dom   : 위 ①~⑤에 해당하는 원변수만 (+달력 +그룹)
#   domx  : dom + 명시적 물리 파생 (이슬점 강하·착빙지수·일최대 풍속/돌풍)
#           — 희소 사건이라 트리가 상호작용을 스스로 만들기 어려울 수 있어 직접 제공
#
# 판정: 평가 시간대(실측 ≥10%cap) AUC. exp-091 결과 = 0.6605(361, 그룹 포함).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp092_availdomain.py
# ═══════════════════════════════════════════════════════
import re
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, prepare_matrices
from exp035_raw_dl import FOLD_LIST, HOLD_FOLD
from exp065_gbm3 import build_train_frame

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

THR = 0.9
CV_FOLDS = [(fn, f) for fn, f in FOLD_LIST if fn != HOLD_FOLD["name"]]

# 도메인 블록 — 원인별로 필요한 원변수 (격자 편차 sdL_/sdG_ 는 전부 제외)
DOMAIN = {
    "①컷아웃/돌풍": r"ws_10m$|ws_100m$|_80_[uv]$|100_100[uv]$|gust|50MU|50MV|XBLWS|YBLWS|wind_shear",
    "②착빙":       r"_2_t$|_2_dpt$|_2_r$|_2_q$|_2_2t$|_2_2d$|_2_2r$|_2_2sh$|850_t$",
    "②강설/강수":   r"ncpcp|lsprate|lssrate|snol|SNOM|prate|_tp$",
    "③운량":       r"lcc|mcc|hcc|tcc|VLCDC",
    "보조(경계층·지형)": r"blh$|VRATE$|surface_0_h$|surface_0_sp$|prmsl$",
}


def pick_domain(cols):
    out, seen = {}, set()
    for name, pat in DOMAIN.items():
        hit = [c for c in cols
               if re.search(pat, c) and not c.startswith(("sdL_", "sdG_")) and c not in seen]
        seen.update(hit)
        out[name] = hit
    return out


def add_derived(df):
    """물리 파생 — ②착빙과 ①컷아웃의 상호작용을 명시적으로 제공."""
    def to_c(x):
        x = df[x].to_numpy(dtype=float)
        return x - 273.15 if np.nanmedian(x) > 200 else x
    t_l, td_l = to_c("ldaps_heightAboveGround_2_t"), to_c("ldaps_heightAboveGround_2_dpt")
    t_g, td_g = to_c("gfs_heightAboveGround_2_2t"), to_c("gfs_heightAboveGround_2_2d")
    df["d_tdd_l"] = t_l - td_l                       # 이슬점 강하 (0에 가까울수록 포화)
    df["d_tdd_g"] = t_g - td_g
    # 착빙지수: 기온 −3℃ 부근 × 포화도. 둘 다 충족할 때만 큼
    df["d_icing"] = np.exp(-((t_l + 3.0) / 5.0) ** 2) * np.exp(-(np.clip(t_l - td_l, 0, None) / 2.0))
    df["d_subzero"] = (t_l < 0).astype(float)
    # 일 최대 풍속·돌풍 (컷아웃은 순간이 아니라 그날의 사건) — 예보 블록 내부로 제한(리크 없음)
    blk = (df["forecast_kst_dtm"] - pd.Timedelta(hours=1)).dt.date
    g = df.groupby([df["kpx_group"], blk], sort=False)
    df["d_wsmax_day"] = g["gfs_ws_100m"].transform("max")
    df["d_gustmax_day"] = g["gfs_surface_0_gust"].transform("max")
    return df, ["d_tdd_l", "d_tdd_g", "d_icing", "d_subzero", "d_wsmax_day", "d_gustmax_day"]


def run_variant(df, cols, tag):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, r2_score
    rows, imps = [], []
    for fname, fold in CV_FOLDS:
        tr = df.loc[df["forecast_kst_dtm"] <= fold["train_end"]]
        va = df.loc[(df["forecast_kst_dtm"] >= fold["valid_start"]) &
                    (df["forecast_kst_dtm"] <= fold["valid_end"])]
        X_tr, X_va = prepare_matrices(tr, va, cols)
        y_tr = (tr["avail_strict"] < THR).astype(int).to_numpy()
        y_va = (va["avail_strict"] < THR).astype(int).to_numpy()
        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                 min_child_samples=40, subsample=0.8, colsample_bytree=0.6,
                                 random_state=0, n_jobs=8, verbose=-1).fit(X_tr, y_tr)
        reg = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                min_child_samples=40, subsample=0.8, colsample_bytree=0.6,
                                random_state=0, n_jobs=8, verbose=-1).fit(
            X_tr, tr["avail_strict"].to_numpy())
        p, pr = clf.predict_proba(X_va)[:, 1], reg.predict(X_va)
        cap = va["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
        ev = (va["target"].to_numpy() >= cap * 0.10) & va["target"].notna().to_numpy()
        rows.append(dict(fold=fname,
                         auc_all=roc_auc_score(y_va, p),
                         auc_ev=roc_auc_score(y_va[ev], p[ev]) if y_va[ev].std() > 0 else np.nan,
                         r2_ev=r2_score(va["avail_strict"].to_numpy()[ev], pr[ev])))
        imps.append(pd.Series(clf.feature_importances_, index=X_tr.columns))
    r = pd.DataFrame(rows)
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
    print(f"  [{tag:5s}] {len(cols)+9:3d}열 | 전체 AUC {r['auc_all'].mean():.4f} | "
          f"**평가시간 AUC {r['auc_ev'].mean():.4f}** | R² {r['r2_ev'].mean():+.4f} | "
          f"fold {'/'.join(f'{x:.3f}' for x in r['auc_ev'])}", flush=True)
    return r, imp


def main():
    t0 = time.time()
    print("### exp-092 정지 예측: 도메인 피처 vs 전체 361 ###", flush=True)
    train_df, use_cols = build_train_frame()
    df = train_df.loc[train_df["avail_strict"].notna()].copy()
    df, dcols = add_derived(df)

    blocks = pick_domain(use_cols)
    dom = [c for v in blocks.values() for c in v]
    print(f"\n도메인 블록 ({len(dom)}개):", flush=True)
    for k, v in blocks.items():
        print(f"  {k:<18} {len(v):2d}  {v}", flush=True)
    print(f"\n파생 {len(dcols)}: {dcols}", flush=True)
    print(f"\n(달력 9 + kpx_group 은 전 변형 공통)\n", flush=True)

    # 결함 수정: 1차 dom은 패턴 매칭 탓에 JMA·EC를 통째로 누락했다.
    # 같은 물리량(풍속·기온·기압)을 다른 예보 모델이 본 값이라 도메인상 포함돼야 함.
    jec = [c for c in use_cols if c.startswith(("jgsm_", "jmsm_", "ec_"))]
    print(f"누락 보정 — JMA·EC {len(jec)}개 추가: {jec}\n", flush=True)

    VAR = {"all": use_cols + ["kpx_group"],
           "dom": dom + ["kpx_group"],
           "domx": dom + dcols + ["kpx_group"],
           "dom2": dom + dcols + jec + ["kpx_group"]}
    imps = {}
    for tag, cols in VAR.items():
        _, imps[tag] = run_variant(df, cols, tag)

    print("\n상위 12 피처 (domx):", flush=True)
    for k, v in imps["domx"].head(12).items():
        print(f"    {k:<42} {v:8.1f}", flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
