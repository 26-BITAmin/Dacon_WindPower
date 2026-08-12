# ═══════════════════════════════════════════════════════
# wi09: DL early stopping — 계절 블록 검증 (사용자 지시, 2026-08-06)
#
# 설계 (사용자 확정): val = 학습창 **마지막 해**의 3/6/9/12월 11~20일 (연속 10일×4 = 40일,
#   2024만 지정했으나 Q1·홀드 폴드는 학습창에 2024가 없어 '마지막 해'로 일반화).
#   감시 = val q60 손실(학습과 동일 가중) · patience 15 · best 가중치 복원 ·
#   최대 120에폭·Cosine 스케줄 불변. 대상 = 챔피언 dl_cln03 (348피처·384×3).
#
# 셀 3개 (귀속 분리):
#   base : 전체 창·120ep 고정 — dl_cln03 캐시 재사용 (재학습 0)
#   ctrl : 창−블록·120ep 고정 — 블록 희생 비용만
#   es   : 창−블록·early stopping — es−ctrl = 순수 ES 효과
#
# 판정: wi01 관례 (기준 dl_cln03 w0.65, 스냅OFF, 작동점 2, Δn/Δf, 이동)
# 실행(루트에서): C:\miniconda3\python.exe -X utf8 "open (1)/testing/wind_input/wi09_earlystop.py"
# ═══════════════════════════════════════════════════════
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

import numpy as np
import pandas as pd
import torch

from exp005_qopt import CAPACITY_KWH
from exp019_multigrid import HOLD_FOLD
from exp033_dl import GIDX, MTMLP, predict
from exp035_raw_dl import CACHE_DIR, FOLD_LIST, build_inputs, score_from_tidy
from exp039_dl_diverse import DEVICE, DL_CFG, build_slices, make_loss
from exp049_clean import PP_NEW, make_xy_clean, merge_avail, postprocess_flex
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp060_pergroup_hpo import load_cache_full
from exp073_gate import ENS_CACHE

from wi01_bivgate import NOSNAP, SCALES

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HOLD = HOLD_FOLD["name"]
VAL_MONTHS = (3, 6, 9, 12)
VAL_DAYS = range(11, 21)                       # 각 달 11~20일 (연속 10일)
PATIENCE = 15
WS_SWEEP = (0.60, 0.65, 0.70)
DL_STORE = HERE.parent / "wi09_dl_es.parquet"
CSV_OUT = HERE.parent / "wi09_results.csv"
# eval 취합 대상 파일 명시 목록 — glob("wi09_dl_es*") 금지: 태그 불문 병합은 다른 구성
# (시드 수·셀 정의)의 캐시를 조용히 섞는 exp-145형 사고 경로다. 새 태그 파일을 취합에
# 넣으려면 여기에 의도적으로 추가할 것.
EVAL_STORES = ("wi09_dl_es.parquet", "wi09_dl_es_c1.parquet", "wi09_dl_es_c2.parquet",
               "wi09_dl_es_e1.parquet", "wi09_dl_es_e2.parquet")


def val_mask(df):
    """학습창 마지막 해의 3/6/9/12월 11~20일.
    폴드 경계가 다음 해 01-01 00:00 한 시간을 포함하므로 −1h 보정으로 해를 정한다."""
    dt = df["forecast_kst_dtm"]
    last_year = int((dt - pd.Timedelta(hours=1)).dt.year.max())
    m = ((dt.dt.year == last_year) & dt.dt.month.isin(VAL_MONTHS)
         & dt.dt.day.isin(VAL_DAYS))
    assert m.any(), f"val 블록 0행 (last_year={last_year})"
    return m


def train_one_es(Xtr, gtr, ytr, wtr, Xv, gv, yv, wv, cfg, loss_fn, seed, use_es):
    """exp039 train_one + (use_es 시) val q60 감시·patience·best 복원. 스케줄 불변."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MTMLP(Xtr.shape[1], cfg["width"], cfg["depth"], cfg["p_drop"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    Xt = torch.tensor(Xtr, device=DEVICE)
    gt = torch.tensor(gtr, device=DEVICE)
    yt = torch.tensor(ytr, device=DEVICE)
    wt = torch.tensor(wtr, device=DEVICE)
    Xv_ = torch.tensor(Xv, device=DEVICE)
    gv_ = torch.tensor(gv, device=DEVICE)
    yv_ = torch.tensor(yv, device=DEVICE)
    wv_ = torch.tensor(wv, device=DEVICE)
    n, bs = Xt.shape[0], 1024
    best, best_state, bad, best_ep = np.inf, None, 0, cfg["epochs"]
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx], gt[idx]), yt[idx], wt[idx])
            loss.backward()
            opt.step()
        sched.step()
        if use_es:
            model.eval()
            with torch.no_grad():
                vl = float(loss_fn(model(Xv_, gv_), yv_, wv_))
            if vl < best - 1e-5:
                best, bad, best_ep = vl, 0, ep + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    if use_es and best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep


def fold_predict(tr_df, va_df, cols, seeds, use_es, drop_blocks, tag):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    vm = val_mask(tr_df)
    fit_df = tr_df.loc[~vm] if drop_blocks else tr_df
    val_df = tr_df.loc[vm & tr_df["target"].notna()]
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sca = StandardScaler()
    Xtr, gtr, ytr, wtr = make_xy_clean(fit_df, cols, imp, sca, "cln03", fit=True)
    Xv, gv, yv, wv = make_xy_clean(val_df, cols, imp, sca, "cln03", fit=False)
    Xva = sca.transform(imp.transform(va_df[cols].to_numpy(dtype=np.float32))).astype(np.float32)
    gva = va_df["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    loss_fn = make_loss("q60")
    preds, eps = [], []
    for s in range(seeds):
        m, bep = train_one_es(Xtr, gtr, ytr, wtr, Xv, gv, yv, wv, DL_CFG, loss_fn, s, use_es)
        preds.append(predict(m, Xva, gva))
        eps.append(bep)
        del m
        torch.cuda.empty_cache()
    if use_es:
        print(f"    {tag} best-epoch: {eps} (val {len(val_df):,}행)", flush=True)
    return np.mean(preds, axis=0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["full", "eval"], default="full")
    ap.add_argument("--cells", type=str, default="ctrl,es")
    ap.add_argument("--folds", type=str, default="")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    t0 = time.time()
    suffix = f"_{args.tag}" if args.tag else ""
    store = HERE.parent / f"wi09_dl_es{suffix}.parquet"
    run_folds = (tuple(f for f in args.folds.split(",") if f) if args.folds
                 else tuple(fn for fn, _ in FOLD_LIST))
    assert all(f in dict(FOLD_LIST) for f in run_folds), f"폴드 이름 오류: {run_folds}"
    print(f"### wi09: DL early stopping — stage={args.stage} cells={args.cells} "
          f"folds={run_folds} ###", flush=True)
    train_df, _, colsets = build_inputs(["spd"])
    vcols = colsets["spd"]
    train_df = merge_avail(train_df)

    cache, champ_bias = load_cache_full()
    assert merge_cached_cols(cache, ENS_CACHE, ["cf_ens3"])
    cols_new = [f"dl_{c}" for c in args.cells.split(",") if c]
    if args.stage == "eval":
        cols_new = ["dl_ctrl", "dl_es"]
        for fname in EVAL_STORES:
            f = HERE.parent / fname
            got = merge_cached_cols(cache, f, [c for c in cols_new if c not in cache.columns
                                               or cache[c].isna().any()])
            if got:
                print(f"  [eval 취합] {fname} → {got}", flush=True)
        miss = [c for c in cols_new if c not in cache.columns or cache[c].isna().any()]
        assert not miss, f"미완성: {miss}"
    elif store.exists():
        merge_cached_cols(cache, store, cols_new)

    for col, use_es in (() if args.stage == "eval" else
                        tuple((c, c == "dl_es") for c in cols_new)):
        if col in cache.columns and cache[col].notna().all():
            print(f"[{col}] 캐시 재사용", flush=True)
            continue
        out = cache[col].to_numpy().copy() if col in cache.columns else np.full(len(cache), np.nan)
        print(f"[{col}] 재학습 (블록 제외{'+ES' if use_es else ''})", flush=True)
        for fname, fold in FOLD_LIST:
            if fname not in run_folds:
                continue
            sel = (cache["fold"] == fname).to_numpy()
            if not np.isnan(out[sel]).any():
                continue
            tf = time.time()
            tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
            keys = cache.loc[sel, ["group", "forecast_kst_dtm"]]
            va = keys.merge(train_df, left_on=["group", "forecast_kst_dtm"],
                            right_on=["kpx_group", "forecast_kst_dtm"], how="left")
            assert len(va) == len(keys), "정렬 실패"
            out[sel] = fold_predict(tr_all, va, vcols, 3, use_es, True, f"{col}:{fname}")
            cache[col] = out
            save_cols(cache, store, [c for c in cols_new if c in cache.columns])
            print(f"  [{col}:{fname}] {time.time()-tf:.0f}s", flush=True)
        cache[col] = out
    if args.stage == "full":
        save_cols(cache, store, cols_new)
        if args.folds or set(cols_new) != {"dl_ctrl", "dl_es"}:
            print(f"\n부분 실행 완료 — 취합 채점은 --stage eval | "
                  f"총 소요: {(time.time()-t0)/60:.1f}분", flush=True)
            return

    # ── 게이트 ──
    slices = build_slices(cache)
    cap_arr = cache["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{g}"]).to_numpy(float)

    def pp_of(raw, scaled):
        out = np.full(len(cache), np.nan)
        for (fname, group), sub in slices:
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            p = postprocess_flex(raw[sub.index], cap, sub["ws"].to_numpy(),
                                 sub["pA"].to_numpy(), group, champ_bias[fname], **NOSNAP)
            out[sub.index] = np.clip(p * SCALES[group], 0, cap) if scaled else p
        return out

    def sc_of(pred):
        t = pd.DataFrame({"fold": cache["fold"], "group": cache["group"],
                          "forecast_kst_dtm": cache["forecast_kst_dtm"],
                          "target": cache["target"], "pred": pred})
        c = t[t["fold"] != HOLD]
        return (score_from_tidy(c, "pred"), score_from_tidy(t[t["fold"] == HOLD], "pred"),
                score_from_tidy(c[c["fold"] != "fold_2024Q4"], "pred"))

    rows = []
    for scaled in (False, True):
        tag = "제출 작동점" if scaled else "미스케일"
        print("\n" + "=" * 100, flush=True)
        print(f"═══ 스냅 OFF | {tag} | 기준 = dl_cln03(base) w0.65 ═══", flush=True)
        res = {}
        for dl in ["dl_cln03", "dl_ctrl", "dl_es"]:
            for w in WS_SWEEP:
                p = pp_of(w * cache["cf_ens3"].to_numpy() + (1 - w) * cache[dl].to_numpy(),
                          scaled)
                res[(dl, w)] = (sc_of(p), p)
        ref, refp = res[("dl_cln03", 0.65)]
        print(f"  {'셀':<10}{'w':>6}{'CV':>9}{'ΔCV':>9}{'Δhd':>9}{'ΔnoQ4':>9}"
              f"{'Δn':>9}{'Δf':>9}{'이동%cap':>10}", flush=True)
        for dl in ["dl_cln03", "dl_ctrl", "dl_es"]:
            for w in WS_SWEEP:
                s, p = res[(dl, w)]
                mark = "  <-기준" if (dl == "dl_cln03" and w == 0.65) else ""
                print(f"  {dl:<10}{w:>6.2f}{s[0][0]:>9.4f}{s[0][0]-ref[0][0]:>+9.4f}"
                      f"{s[1][0]-ref[1][0]:>+9.4f}{s[2][0]-ref[2][0]:>+9.4f}"
                      f"{s[0][1]-ref[0][1]:>+9.4f}{s[0][2]-ref[0][2]:>+9.4f}"
                      f"{np.nanmean((p-refp)/cap_arr)*100:>+10.3f}{mark}", flush=True)
                rows.append(dict(scaled=scaled, dl=dl, w=w, cv=s[0][0],
                                 d_cv=s[0][0]-ref[0][0], d_hold=s[1][0]-ref[1][0],
                                 d_noq4=s[2][0]-ref[2][0], d_n=s[0][1]-ref[0][1],
                                 d_f=s[0][2]-ref[0][2],
                                 shift_pcap=float(np.nanmean((p-refp)/cap_arr)*100)))
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print(f"\n상세 → {CSV_OUT.name} | 총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
