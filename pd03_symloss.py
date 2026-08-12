# ═══════════════════════════════════════════════════════
# pd03: 트랙 D 3라운드 — base 손실 q60 → 대칭 huber, anws 셀만 (PD-03)
#
# 사용자 지시 (2026-08-11): "1번만 해봐" (커브 진단 후: 상향 편향의 주범 후보 = q60 비대칭)
#
# 변경점 1개: DL base 손실 q60(τ0.60) → hub(대칭 huber, exp039 make_loss("hub")).
#   ES 감시 손실도 hub로 동일 (학습과 동일 손실 감시 관례 유지).
# 동결: anws 입력(멤버 ws 30) · MTMLP 384×3 · ES(val 블록·patience 15·best 복원) · 3시드
#   · GBM 잔차 = ctrl 방식(전 라벨행·고정 트리 448/585 — PD-02에서 ES 불채택) · PP · 스케일
#
# 판정: 트랙 기준 anws(PD-01/PD-02 ctrl) 대비. draw 노이즈 ≤ 0.0005 (PD-02 실측).
#
# 실행(루트에서):
#   C:\miniconda3\python.exe -X utf8 "open (1)/testing/curve_track/pd03_symloss.py"
# ═══════════════════════════════════════════════════════
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "wind_input"))
sys.path.insert(0, str(HERE.parent))

import numpy as np
import pandas as pd
import torch

from exp005_qopt import CAPACITY_KWH
from exp019_multigrid import HOLD_FOLD
from exp033_dl import GIDX, predict
from exp035_raw_dl import FOLD_LIST, score_from_tidy
from exp039_dl_diverse import DL_CFG, build_slices, make_loss
from exp049_clean import KEY as CKEY, make_xy_clean, postprocess_flex
from exp050_lambda_sweep import save_cols
from exp056_hpo import RESULT_JSON as E56_JSON
from exp060_pergroup_hpo import load_cache_full
from exp065_gbm3 import build_train_frame
from exp073_gate import ENS_CACHE

from wi01_bivgate import NOSNAP, SCALES
from wi09_earlystop import train_one_es, val_mask

from pd00_inputs import add_calendar, load_pools
from pd01_gate import (MEM_WS, AN_MEMBERS, WI, fill_from, merge_feats,
                       crossfit_masks, gbm_residual)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HOLD = HOLD_FOLD["name"]
KEY = ["kpx_group", "forecast_kst_dtm"]
STORE = HERE.parent / "pd03_preds.parquet"
DL_STORE = HERE.parent / "pd03_dltrain.parquet"
CSV_OUT = HERE.parent / "pd03_results.csv"
CELL = "anws_hub"


def fold_predict_loss(tr_df, va_df, cols, seeds, loss_kind, tag):
    """wi09.fold_predict와 동일하되 손실 종류만 인자화 (학습·ES 감시 동일 손실)."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    vm = val_mask(tr_df)
    fit_df = tr_df.loc[~vm]
    val_df = tr_df.loc[vm & tr_df["target"].notna()]
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sca = StandardScaler()
    Xtr, gtr, ytr, wtr = make_xy_clean(fit_df, cols, imp, sca, "cln03", fit=True)
    Xv, gv, yv, wv = make_xy_clean(val_df, cols, imp, sca, "cln03", fit=False)
    Xva = sca.transform(imp.transform(va_df[cols].to_numpy(dtype=np.float32))).astype(np.float32)
    gva = va_df["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    loss_fn = make_loss(loss_kind)
    preds, eps = [], []
    for s in range(seeds):
        m, bep = train_one_es(Xtr, gtr, ytr, wtr, Xv, gv, yv, wv, DL_CFG, loss_fn, s, True)
        preds.append(predict(m, Xva, gva))
        eps.append(bep)
        del m
        torch.cuda.empty_cache()
    print(f"    {tag} best-epoch: {eps} (val {len(val_df):,}행)", flush=True)
    return np.mean(preds, axis=0)


def main():
    t0 = time.time()
    seeds = 3
    print(f"### pd03: anws 손실 q60→hub (변경점 1개) ###", flush=True)
    dl_pool, gbm_pool, env_cols = load_pools()
    e56 = json.loads(E56_JSON.read_text())
    train_df, _ = build_train_frame()
    train_df = add_calendar(train_df)
    gbm_cols = gbm_pool + ["pd_dl_cf"]

    cache, champ_bias = load_cache_full()
    fill_from(cache, ENS_CACHE, ["cf_ens3"])
    for f in sorted(WI.glob("wi09_dl_es*.parquet")):
        fill_from(cache, f, ["dl_es"])
    for f in sorted(HERE.parent.glob("pd01_preds*.parquet")):
        fill_from(cache, f, ["pdd_anws", "pdb_anws"])
    for f in sorted(HERE.parent.glob("pd02_preds*.parquet")):
        fill_from(cache, f, ["pdc_anws"])
    fill_from(cache, STORE, [f"pdd_{CELL}", f"pdb_{CELL}"])

    members = pd.read_parquet(AN_MEMBERS)
    for k in range(30):
        members[f"mws{k}"] = np.hypot(members[f"m{k}u"], members[f"m{k}v"])

    fcol, bcol = f"pdd_{CELL}", f"pdb_{CELL}"
    fin = cache[fcol].to_numpy().copy() if fcol in cache.columns else np.full(len(cache), np.nan)
    bas = cache[bcol].to_numpy().copy() if bcol in cache.columns else np.full(len(cache), np.nan)
    dl_saved = pd.read_parquet(DL_STORE) if DL_STORE.exists() else None
    for fname, fold in FOLD_LIST:
        sel = (cache["fold"] == fname).to_numpy()
        if not np.isnan(fin[sel]).any():
            continue
        tf = time.time()
        f_cf = members[(members["fold"] == fname) & (members["split"] == "cf")]
        f_va = members[(members["fold"] == fname) & (members["split"] == "va")]
        wf = merge_feats(train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]],
                         f_cf, MEM_WS)
        keys = cache.loc[sel, ["group", "forecast_kst_dtm"]]
        vf = keys.merge(merge_feats(train_df, f_va, MEM_WS),
                        left_on=["group", "forecast_kst_dtm"], right_on=KEY, how="left")
        assert len(vf) == len(keys), "정렬 실패"
        lab = wf.loc[wf["target"].notna()].copy()

        lab["pd_dl_cf"] = np.nan
        h0 = crossfit_masks(lab)
        for fit_m, q_m, tg in ((h0, ~h0, "cf01"), (~h0, h0, "cf10")):
            p = fold_predict_loss(lab.loc[fit_m], lab.loc[q_m], MEM_WS, seeds, "hub",
                                  f"{CELL}:{fname}:{tg}")
            lab.loc[q_m, "pd_dl_cf"] = p
        add = lab[KEY + ["pd_dl_cf"]].assign(fold=fname, cell=CELL)
        dl_saved = add if dl_saved is None else pd.concat(
            [dl_saved[~((dl_saved["fold"] == fname) & (dl_saved["cell"] == CELL))], add],
            ignore_index=True)
        dl_saved.to_parquet(DL_STORE, index=False)
        dl_va = fold_predict_loss(lab, vf, MEM_WS, seeds, "hub", f"{CELL}:{fname}:va")
        bas[sel] = dl_va

        vf = vf.copy()
        vf["pd_dl_cf"] = dl_va
        res_va = np.full(len(vf), np.nan)
        for g in [1, 2, 3]:
            tr_g = lab.loc[(lab["kpx_group"] == g) & lab["pd_dl_cf"].notna()]
            vmk = (vf["kpx_group"] == g).to_numpy()
            if not len(tr_g) or not vmk.any():
                continue
            cap = CAPACITY_KWH[f"kpx_group_{g}"]
            res_va[vmk] = gbm_residual(tr_g, vf.loc[vmk], gbm_cols, e56, seeds=seeds) / cap
        fin[sel] = dl_va + res_va
        cache[fcol], cache[bcol] = fin, bas
        save_cols(cache, STORE, [fcol, bcol])
        print(f"  [{CELL}:{fname}] {time.time()-tf:.0f}s "
              f"(|잔차보정| 평균 {np.nanmean(np.abs(res_va))*100:.2f}%cap)", flush=True)
    cache[fcol], cache[bcol] = fin, bas
    save_cols(cache, STORE, [fcol, bcol])

    # ── 채점 ──
    slices = build_slices(cache)

    def pp_of(raw, scaled):
        out = np.full(len(cache), np.nan)
        for (fn, group), sub in slices:
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            p = postprocess_flex(raw[sub.index], cap, sub["ws"].to_numpy(),
                                 sub["pA"].to_numpy(), group, champ_bias[fn], **NOSNAP)
            out[sub.index] = np.clip(p * SCALES[group], 0, cap) if scaled else p
        return out

    def sc_of(pred):
        t = pd.DataFrame({"fold": cache["fold"], "group": cache["group"],
                          "forecast_kst_dtm": cache["forecast_kst_dtm"],
                          "target": cache["target"], "pred": pred})
        c = t[t["fold"] != HOLD]
        return score_from_tidy(c, "pred"), score_from_tidy(t[t["fold"] == HOLD], "pred")

    champ_raw = 0.65 * cache["cf_ens3"].to_numpy() + 0.35 * cache["dl_es"].to_numpy()
    rows = []
    for scaled in (False, True):
        tag = "제출 작동점 ×.98/.98/.97" if scaled else "미스케일"
        print("\n" + "=" * 100, flush=True)
        print(f"═══ {tag} | 기준 = anws(q60, PD-01) ═══", flush=True)
        ref = sc_of(pp_of(cache["pdd_anws"].to_numpy(), scaled))
        print(f"  {'행':<18}{'CV':>9}{'Δ기준':>9}{'Δn':>9}{'Δf':>9}{'HOLD':>9}{'base CV':>9}",
              flush=True)
        for name, col, bcol2 in (("champ(참고)", None, None),
                                 ("anws q60 (기준)", "pdd_anws", "pdb_anws"),
                                 ("anws ctrl(PD-02)", "pdc_anws", None),
                                 ("anws_hub", fcol, bcol)):
            raw = champ_raw if col is None else cache[col].to_numpy()
            s = sc_of(pp_of(raw, scaled))
            sb = sc_of(pp_of(cache[bcol2].to_numpy(), scaled)) if bcol2 else None
            print(f"  {name:<18}{s[0][0]:>9.4f}{s[0][0]-ref[0][0]:>+9.4f}"
                  f"{s[0][1]-ref[0][1]:>+9.4f}{s[0][2]-ref[0][2]:>+9.4f}{s[1][0]:>9.4f}"
                  f"{(sb[0][0] if sb else float('nan')):>9.4f}", flush=True)
            rows.append(dict(scaled=scaled, name=name, cv=s[0][0], d=s[0][0]-ref[0][0],
                             d_n=s[0][1]-ref[0][1], d_f=s[0][2]-ref[0][2], hold=s[1][0],
                             base_cv=(sb[0][0] if sb else None)))
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print(f"\n상세 → {CSV_OUT.name} | 총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
