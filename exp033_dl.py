# ═══════════════════════════════════════════════════════
# exp-033: 딥러닝 트랙 — 멀티태스크 MLP (GBM 앙상블 파트너)
#
# 전략(exp-032 후): GBM 잔차는 이 피처셋에서 비가역(R²<0). DL을 GBM 대체가 아니라
#   **오차 탈상관 앙상블 파트너**로 사용 — NN(매끄러운 함수)은 GBM(축정렬 분할)과
#   다른 오차를 내므로 블렌드가 각각보다 나을 수 있음(LGBM+XGB 블렌드 원리). 블렌드는
#   시드처럼 mild 변경이라 2024→2025 전이 가능성이 aggressive 스냅(exp-031)보다 높음.
#   멀티태스크(공유 trunk + 그룹별 head): 그룹3이 그룹1·2 표현 차용(전이).
#
# dl-modeling 스킬 phase 준수: Phase1(데이터 검증)·Phase2(오버핏)·Phase3(전체학습)·
#   Phase4(에러분석)·Phase5(개선)·Phase6(최종). 이 파일은 --mode로 단계 실행.
# 실행: C:\miniconda3\python.exe -X utf8 exp033_dl.py --mode overfit|cv [--jobs N]
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from exp005_qopt import CAPACITY_KWH, TARGET_COLS, FOLDS_Q, metric
from exp012_feature_v11 import build_v11
from exp020_jma_features import build_jma_frame, JG_COLS, JM_COLS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GROUPS = [1, 2, 3]
GIDX = {1: 0, 2: 1, 3: 2}


def load_data():
    train_df, test_df, col_sets, sample_submission = build_v11()
    jma = build_jma_frame()
    for df in (train_df, test_df):
        pass
    train_df = train_df.merge(jma, on=["forecast_kst_dtm", "kpx_group"], how="left")
    test_df = test_df.merge(jma, on=["forecast_kst_dtm", "kpx_group"], how="left")
    for df in (train_df, test_df):
        df["jgsm_dis"] = (df["jgsm_ws"] - df["ldaps_ws_10m"]).abs()
        df["jmsm_dis"] = (df["jmsm_ws"] - df["ldaps_ws_10m"]).abs()
    cols = col_sets["v10TPS"] + JG_COLS + JM_COLS
    return train_df, test_df, cols, sample_submission


class MTMLP(nn.Module):
    """공유 trunk + 그룹별 head. 그룹 임베딩도 trunk 입력에 concat."""
    def __init__(self, n_feat, width=256, depth=3, p_drop=0.1, emb=8):
        super().__init__()
        self.emb = nn.Embedding(3, emb)
        layers, d_in = [], n_feat + emb
        for i in range(depth):
            layers += [nn.Linear(d_in, width), nn.BatchNorm1d(width), nn.SiLU(), nn.Dropout(p_drop)]
            d_in = width
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(width, 1) for _ in range(3)])

    def forward(self, x, g):
        h = self.trunk(torch.cat([x, self.emb(g)], dim=1))
        # 각 샘플을 그룹 head로 라우팅
        out = torch.zeros(x.shape[0], device=x.device)
        for gi in range(3):
            m = g == gi
            if m.any():
                out[m] = self.heads[gi](h[m]).squeeze(1)
        return out


def make_xy(df, cols, imputer, scaler, fit=False):
    """long-format df → (X, group_idx, cf_target, weight). fit=True면 imputer/scaler 적합."""
    sub = df.loc[df["target"].notna()].copy()
    caps = sub["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    Xraw = sub[cols].to_numpy(dtype=np.float32)
    if fit:
        Xi = imputer.fit_transform(Xraw)
        Xs = scaler.fit_transform(Xi)
    else:
        Xi = imputer.transform(Xraw)
        Xs = scaler.transform(Xi)
    g = sub["kpx_group"].map(GIDX).to_numpy()
    cf = (sub["target"].to_numpy() / caps).astype(np.float32)
    w = np.where(cf < 0.10, 0.05, 1.0).astype(np.float32)
    return (Xs.astype(np.float32), g.astype(np.int64), cf, w, sub, caps)


def huber_weighted(pred, tgt, w, delta=0.05):
    e = pred - tgt
    a = torch.abs(e)
    loss = torch.where(a <= delta, 0.5 * e * e, delta * (a - 0.5 * delta))
    return (loss * w).sum() / w.sum()


def train_model(Xtr, gtr, ytr, wtr, Xva=None, gva=None, yva=None, wva=None,
                width=256, depth=3, p_drop=0.1, lr=1e-3, wd=1e-4, epochs=60,
                bs=1024, seed=0, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MTMLP(Xtr.shape[1], width, depth, p_drop).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt = torch.tensor(Xtr, device=DEVICE); gt = torch.tensor(gtr, device=DEVICE)
    yt = torch.tensor(ytr, device=DEVICE); wt = torch.tensor(wtr, device=DEVICE)
    n = Xt.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            pred = model(Xt[idx], gt[idx])
            loss = huber_weighted(pred, yt[idx], wt[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            msg = f"  ep{ep:3d} train {tot/n:.5f}"
            if Xva is not None:
                vp = predict(model, Xva, gva)
                vl = huber_weighted(torch.tensor(vp), torch.tensor(yva), torch.tensor(wva)).item()
                msg += f" | val {vl:.5f}"
            print(msg, flush=True)
    return model


@torch.no_grad()
def predict(model, X, g, bs=8192):
    model.eval()
    out = []
    Xt = torch.tensor(X, device=DEVICE); gt = torch.tensor(g, device=DEVICE)
    for i in range(0, Xt.shape[0], bs):
        out.append(model(Xt[i:i+bs], gt[i:i+bs]).cpu().numpy())
    return np.concatenate(out)


def overfit_test():
    print(f"### Phase 2: 오버핏 테스트 (device={DEVICE}) ###")
    train_df, _, cols, _ = load_data()
    imp, sca = SimpleImputer(strategy="median"), StandardScaler()
    X, g, y, w, _, _ = make_xy(train_df, cols, imp, sca, fit=True)
    # 100 샘플만, 정규화 제거
    idx = np.random.RandomState(0).choice(len(X), 100, replace=False)
    Xs, gs, ys, ws = X[idx], g[idx], y[idx], np.ones(100, np.float32)
    print(f"피처 {X.shape[1]}개, 100 샘플, dropout=0/wd=0로 오버핏 시도")
    model = train_model(Xs, gs, ys, ws, width=256, depth=3, p_drop=0.0, wd=0.0,
                        lr=1e-3, epochs=300, bs=100, seed=0, verbose=True)
    pred = predict(model, Xs, gs)
    mae = np.abs(pred - ys).mean()
    print(f"\n오버핏 결과: train MAE(CF) = {mae:.5f} (→0 근처면 성공)")
    print("성공" if mae < 0.01 else "실패 — 아키텍처/코드 점검 필요")


def dl_predict_fold(tr_df, va_df, cols, seeds, epochs, width, depth, p_drop, lr, wd):
    """fold train으로 DL 학습(seeds 평균) → va_df 전체(그룹 long) CF 예측 dict{group: cf_array}."""
    imp, sca = SimpleImputer(strategy="median"), StandardScaler()
    Xtr, gtr, ytr, wtr, _, _ = make_xy(tr_df, cols, imp, sca, fit=True)
    preds = []
    for s in range(seeds):
        model = train_model(Xtr, gtr, ytr, wtr, width=width, depth=depth, p_drop=p_drop,
                            lr=lr, wd=wd, epochs=epochs, bs=1024, seed=s, verbose=(s == 0))
        # valid: 그룹별로 예측 (target 유무 무관하게 전 valid 행)
        va = va_df.copy()
        Xva = sca.transform(imp.transform(va[cols].to_numpy(dtype=np.float32)))
        gva = va["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
        preds.append(predict(model, Xva.astype(np.float32), gva))
    return np.mean(preds, axis=0)  # CF, va_df 순서


def run_cv(mode_folds, tag, args):
    import lightgbm as lgb
    import xgboost as xgbm
    from exp005_qopt import (make_sample_weight, apply_rolling_smoothing,
                             apply_bias_correction_v2, lgbm_params, prepare_matrices, metric_single_group)
    from exp009_rated_snap import (estimate_bias_data, bias_map_from_data, COARSE_BINS, RATED_CF, N_CLF_SEEDS)
    from exp011_g3_transfer import build_pooled_matrices

    ALPHA, CLIP_LO, SNAP_THR, SNAP_TO = 0.50, 0.16, 0.30, 0.94
    WSWEEP = {"gbm": 1.0, "b70": 0.7, "b50": 0.5, "b30": 0.3, "dl": 0.0}

    train_df, _, cols, _ = load_data()
    rows_all = {v: {"ans": [], "pred": []} for v in WSWEEP}
    corr_list = []

    for fold in mode_folds:
        tf = time.time()
        tr_mask = train_df["forecast_kst_dtm"] <= fold["train_end"]
        va_mask = (train_df["forecast_kst_dtm"] >= fold["valid_start"]) & (train_df["forecast_kst_dtm"] <= fold["valid_end"])
        tr_all, va_all = train_df.loc[tr_mask], train_df.loc[va_mask]
        tr_pool = tr_all.loc[tr_all["target"].notna()]

        # DL: 전 valid 행 CF 예측 (long)
        va_sorted = va_all.sort_values(["kpx_group", "forecast_kst_dtm"]).copy()
        dl_cf_all = dl_predict_fold(tr_all, va_sorted, cols, args.seeds, args.epochs,
                                    args.width, args.depth, args.drop, args.lr, args.wd)
        va_sorted["dl_cf"] = dl_cf_all

        # GBM base + 편향맵 + 스냅분류기
        bias_data = estimate_bias_data(train_df, cols, fold["train_end"], 500, args.jobs)
        coarse_map = bias_map_from_data(bias_data, {g: COARSE_BINS for g in bias_data})

        va_time_index = sorted(va_all["forecast_kst_dtm"].unique())
        answer_df = pd.DataFrame({"forecast_kst_dtm": va_time_index})
        pred_dfs = {v: pd.DataFrame({"forecast_kst_dtm": va_time_index}) for v in WSWEEP}
        pooled_clf = []

        for group in GROUPS:
            target_col = f"kpx_group_{group}"
            capacity = CAPACITY_KWH[target_col]
            tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
            va_g = va_sorted.loc[va_sorted["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
            answer_df = answer_df.merge(va_g[["forecast_kst_dtm", "target"]].rename(columns={"target": target_col}),
                                        on="forecast_kst_dtm", how="left")
            if len(tr_g) == 0:
                for v in WSWEEP:
                    pred_dfs[v][target_col] = 0.0
                continue
            X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
            y_tr = tr_g["target"]; w_tr = make_sample_weight(y_tr.to_numpy(), capacity, low_weight=0.05)
            pl = np.mean([lgb.LGBMRegressor(**lgbm_params(500, args.jobs, s)).fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(3)], axis=0)
            px = np.mean([xgbm.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, min_child_weight=5,
                          subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=args.jobs, verbosity=0,
                          objective="reg:absoluteerror").fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(3)], axis=0)
            gbm_cf = (0.5 * pl + 0.5 * px) / capacity
            dl_cf = va_g["dl_cf"].to_numpy()

            X_pool_tr, X_pool_va = build_pooled_matrices(tr_pool, va_g, cols)
            if not pooled_clf:
                caps_pool = tr_pool["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
                yA = (tr_pool["target"].to_numpy() >= caps_pool * RATED_CF).astype(int)
                for s in range(N_CLF_SEEDS):
                    c = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, num_leaves=31,
                        min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=args.jobs, verbose=-1)
                    c.fit(X_pool_tr, yA); pooled_clf.append(c)
            pA = np.mean([c.predict_proba(X_pool_va)[:, 1] for c in pooled_clf], axis=0)
            ws = va_g["ldaps_ws_10m"].to_numpy(); actual = va_g["target"].to_numpy()

            # DL↔GBM 오차 상관 (탈상관이면 블렌드 이득)
            ev = actual >= capacity * 0.10
            if ev.sum() > 10:
                eg = gbm_cf[ev] - actual[ev]/capacity; ed = dl_cf[ev] - actual[ev]/capacity
                corr_list.append({"fold": fold["name"], "group": group,
                                  "corr": float(np.corrcoef(eg, ed)[0, 1]),
                                  "gbm_mae": float(np.abs(eg).mean()), "dl_mae": float(np.abs(ed).mean())})

            for v, w in WSWEEP.items():
                base = (w * gbm_cf + (1 - w) * dl_cf) * capacity
                base = apply_bias_correction_v2(base, ws, group, coarse_map)
                base = apply_rolling_smoothing(pd.Series(base, index=va_g.index), 3, ALPHA).to_numpy().copy()
                base[pA >= SNAP_THR] = np.maximum(base[pA >= SNAP_THR], capacity * SNAP_TO)
                base = np.clip(base, capacity * CLIP_LO, capacity)
                vp = va_g[["forecast_kst_dtm"]].copy(); vp[target_col] = base
                pred_dfs[v] = pred_dfs[v].merge(vp, on="forecast_kst_dtm", how="left")

        ans_f = answer_df[TARGET_COLS].fillna(0.0)
        for v in WSWEEP:
            rows_all[v]["ans"].append(ans_f)
            rows_all[v]["pred"].append(pred_dfs[v][TARGET_COLS].fillna(0.0))
        line = {v: metric(ans_f, pred_dfs[v][TARGET_COLS].fillna(0.0))[0] for v in WSWEEP}
        print(f"[{tag}:{fold['name']}] ({time.time()-tf:.0f}s) " + " ".join(f"{v}={line[v]:.4f}" for v in WSWEEP), flush=True)

    print(f"\n--- {tag} pooled ---")
    res = {}
    for v in WSWEEP:
        A = pd.concat(rows_all[v]["ans"], ignore_index=True); P = pd.concat(rows_all[v]["pred"], ignore_index=True)
        s, n, f = metric(A, P); res[v] = s
        print(f"  {v:4s}(w_gbm={WSWEEP[v]:.1f}): Score {s:.4f} | 1-NMAE {n:.4f} | FiCR {f:.4f}  Δgbm {s-res['gbm']:+.4f}" if v != "gbm"
              else f"  {v:4s}(w_gbm={WSWEEP[v]:.1f}): Score {s:.4f} | 1-NMAE {n:.4f} | FiCR {f:.4f}  (기준)")
    if corr_list:
        cdf = pd.DataFrame(corr_list)
        print(f"\n  DL↔GBM 오차 상관 평균 {cdf['corr'].mean():.3f} (낮을수록 블렌드 이득) | "
              f"GBM MAE(CF) {cdf['gbm_mae'].mean():.4f} / DL MAE {cdf['dl_mae'].mean():.4f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["overfit", "cv"], default="overfit")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    if args.mode == "overfit":
        overfit_test()
    else:
        from exp019_multigrid import HOLD_FOLD
        folds = FOLDS_Q[-1:] if args.quick else FOLDS_Q
        print(f"### Phase 3: CV (device={DEVICE}, seeds={args.seeds}, epochs={args.epochs}, "
              f"MLP {args.width}x{args.depth} drop{args.drop}) ###", flush=True)
        cv = run_cv(folds, "CV", args)
        if not args.quick:
            print("\n### 홀드아웃 게이트 ###")
            hd = run_cv([HOLD_FOLD], "HOLD", args)
            print("\n=== 이중 게이트 (Δgbm) ===")
            for v in cv:
                if v != "gbm":
                    print(f"  {v}: ΔCV {cv[v]-cv['gbm']:+.4f} | Δhold {hd[v]-hd['gbm']:+.4f} | "
                          f"both_pos {(cv[v]-cv['gbm']>0) and (hd[v]-hd['gbm']>0)}")
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
