# ═══════════════════════════════════════════════════════
# pd01: 별도 트랙 D 1라운드 게이트 — DL 커브 base 입력 6셀 ablation (PD-01)
#
# 사용자 설계 (2026-08-11):
#   구조 = 잔차형 직렬. "DL이 회귀로 파워 커브를 그리고, 미세한 차이를 GBM이 학습."
#     1단 DL base : 풍속 채널(+환경)만 입력 → CF 커브 회귀 (MTMLP 384×3 · q60 · ES · 3시드)
#     2단 GBM     : 타깃 = 잔차(y − DL·cap), 입력 = DL 출력 + GBM 풀 269(+calendar)
#                   LGB e56(448) + XGB e56(585) 50:50 · 3시드 · 챔피언 가중(저발전 0.05×cln03)
#                   ※ 1라운드는 GBM ES 미적용(고정 트리 수 = ens3 관례) — 명시적 결정
#     최종        : DL + 잔차 → PP(챔피언 편향맵 · NOSNAP · clip · 평활) → (±그룹 스케일)
#
#   6셀 (조작 변인 = DL base 입력만):
#     anuv     AnEn K=30 멤버 u/v 60열            (av00 캐시)
#     anws     AnEn K=30 멤버 ws 크기 30열        (hypot 변환)
#     anuv_env anuv + 환경 92 + calendar 9
#     anws_env anws + 환경 92 + calendar 9
#     gws      GBM 풍속 점예측 1열                 (pd00 캐시)
#     gws_env  gws + 환경 92 + calendar 9
#
# 리크 방지:
#   DL 학습행의 base 피처 = 교차적합분(av00 split cf · pd00 split cf)
#   GBM 잔차 학습에 쓰는 DL 예측 = 그룹별 시간중앙 2분할 교차적합 (자기 적합 잔차 축소 방지)
#   검증행 = 학습창 전체로 학습한 DL/GBM의 예측 (실운영 조건)
#
# 판정: 별도 트랙 규칙 — 1라운드 = 내부 6셀 비교(기준 수립). 챔피언 행은 참고 표기 전용.
#
# 실행(루트에서):
#   C:\miniconda3\python.exe -X utf8 "open (1)/testing/curve_track/pd01_gate.py" --stage full --cells anuv,anws,gws --tag a
#   C:\miniconda3\python.exe -X utf8 "open (1)/testing/curve_track/pd01_gate.py" --stage full --cells anuv_env,anws_env,gws_env --tag b
#   C:\miniconda3\python.exe -X utf8 "open (1)/testing/curve_track/pd01_gate.py" --stage eval
# ═══════════════════════════════════════════════════════
import argparse
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

from exp005_qopt import CAPACITY_KWH, make_sample_weight, prepare_matrices
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import FOLD_LIST, score_from_tidy
from exp039_dl_diverse import build_slices
from exp049_clean import KEY as CKEY, avail_factor, postprocess_flex
from exp050_lambda_sweep import save_cols
from exp056_hpo import RESULT_JSON as E56_JSON
from exp060_pergroup_hpo import load_cache_full
from exp065_gbm3 import build_train_frame
from exp073_gate import ENS_CACHE

from wi01_bivgate import NOSNAP, SCALES
from wi09_earlystop import fold_predict

from pd00_inputs import (GBMWS_PATH, GBMWS_HUB_PATH, QWS_COLS, add_calendar,
                         load_pools, CALENDAR9)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HOLD = HOLD_FOLD["name"]
KEY = ["kpx_group", "forecast_kst_dtm"]
AN_MEMBERS = HERE.parents[1] / "anen_input" / "av00_anen_members.parquet"
WI = HERE.parents[1] / "wind_input"
CSV_OUT = HERE.parent / "pd01_results.csv"
MEM_UV = [f"m{k}{c}" for k in range(30) for c in ("u", "v")]
MEM_WS = [f"mws{k}" for k in range(30)]
ALL_CELLS = ("anuv", "anws", "anuv_env", "anws_env", "gws", "gws_env", "gws_q15",
             "gws_hub")


def fill_from(cache, path, wanted):
    if not path.exists():
        return
    prev = pd.read_parquet(path)
    for c in wanted:
        if c not in prev.columns:
            continue
        m = cache[CKEY].merge(prev[CKEY + [c]], on=CKEY, how="left")
        s = pd.Series(m[c].to_numpy(), index=cache.index)
        cache[c] = s if c not in cache.columns else cache[c].fillna(s)


def cell_cols(cell, env_cols):
    base = {"anuv": MEM_UV, "anws": MEM_WS, "gws": ["ws_gbm"],
            "gws_q15": QWS_COLS, "gws_hub": ["ws_gbm"]}[cell.replace("_env", "")]
    return base + (env_cols + CALENDAR9 if cell.endswith("_env") else [])


def merge_feats(df, feats, cols):
    m = df.merge(feats[KEY + cols], on=KEY, how="left")
    assert len(m) == len(df), "병합 행 증식"
    return m


def crossfit_masks(labeled):
    """그룹별 시간중앙 2분할 마스크 (h0 = 각 그룹 라벨행의 전반부)."""
    h0 = pd.Series(False, index=labeled.index)
    for g in [1, 2, 3]:
        gm = labeled["kpx_group"] == g
        med = labeled.loc[gm, "forecast_kst_dtm"].quantile(0.5)
        h0 |= gm & (labeled["forecast_kst_dtm"] <= med)
    return h0


def gbm_residual(tr_g, va_g, cols, e56, seeds=3, jobs=8):
    """챔피언 GBM 관례(prepare_matrices·e56·가중)로 잔차 회귀. 반환 = va 잔차 예측(kWh)."""
    import lightgbm as lgb
    import xgboost as xgbm
    cap = CAPACITY_KWH[f"kpx_group_{int(tr_g['kpx_group'].iloc[0])}"]
    X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
    y_res = tr_g["target"].to_numpy() - tr_g["pd_dl_cf"].to_numpy() * cap
    w_tr = make_sample_weight(tr_g["target"].to_numpy(), cap, low_weight=0.05) * \
        avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
    pl = np.mean([lgb.LGBMRegressor(n_estimators=e56["lgb_n"], objective="mae", random_state=s,
                                    n_jobs=jobs, verbose=-1, device="gpu", **e56["lgb"])
                  .fit(X_tr, y_res, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
    px = np.mean([xgbm.XGBRegressor(n_estimators=e56["xgb_n"], objective="reg:absoluteerror",
                                    tree_method="hist", device="cuda", random_state=s, n_jobs=jobs,
                                    verbosity=0, **e56["xgb"])
                  .fit(X_tr, y_res, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
    return 0.5 * pl + 0.5 * px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["full", "eval"], default="full")
    ap.add_argument("--cells", type=str, default=",".join(ALL_CELLS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--folds", type=str, default="")
    args = ap.parse_args()
    only_folds = [f for f in args.folds.split(",") if f]
    cells = [c for c in args.cells.split(",") if c]
    assert all(c in ALL_CELLS for c in cells), f"셀 이름 오류: {cells}"
    suffix = f"_{args.tag}" if args.tag else ""
    store = HERE.parent / f"pd01_preds{suffix}.parquet"
    t0 = time.time()
    print(f"### pd01: 트랙 D 1라운드 (잔차형 커브 base) — stage={args.stage} "
          f"cells={cells} ###", flush=True)

    dl_pool, gbm_pool, env_cols = load_pools()
    e56 = json.loads(E56_JSON.read_text())
    train_df, cols352 = build_train_frame()          # 352 + avail_strict 병합됨
    train_df = add_calendar(train_df)
    gbm_cols = gbm_pool + ["pd_dl_cf"]               # calendar는 prepare_matrices가 추가

    cache, champ_bias = load_cache_full()
    fill_from(cache, ENS_CACHE, ["cf_ens3"])
    for f in sorted(WI.glob("wi09_dl_es*.parquet")):
        fill_from(cache, f, ["dl_es"])
    assert cache["cf_ens3"].notna().all() and cache["dl_es"].notna().all()

    fin_all = [f"pdd_{c}" for c in ALL_CELLS]
    base_all = [f"pdb_{c}" for c in ALL_CELLS]
    if args.stage == "eval":
        for f in sorted(HERE.parent.glob("pd01_preds*.parquet")):
            fill_from(cache, f, fin_all + base_all)
        miss = [c for c in fin_all if c not in cache.columns or cache[c].isna().any()]
        assert not miss, f"미완성 셀: {miss}"
    else:
        cols_new = [f"pdd_{c}" for c in cells] + [f"pdb_{c}" for c in cells]
        fill_from(cache, store, cols_new)
        members = pd.read_parquet(AN_MEMBERS)
        for k in range(30):
            members[f"mws{k}"] = np.hypot(members[f"m{k}u"], members[f"m{k}v"])
        gbmws = pd.read_parquet(GBMWS_PATH)
        qfiles = sorted(HERE.parent.glob("pd00_qws*.parquet"))
        qws = (pd.concat([pd.read_parquet(f) for f in qfiles], ignore_index=True)
               .drop_duplicates(KEY + ["fold", "split"]) if qfiles else None)
        if qws is not None:   # 독립 분위수의 행내 교차 제거 (단조 재배열 — Chernozhukov rearrangement)
            qws[QWS_COLS] = np.sort(qws[QWS_COLS].to_numpy(), axis=1)
        print(f"AnEn 멤버 {members.shape} · GBM 풍속 {gbmws.shape} · "
              f"분위수 {qws.shape if qws is not None else '없음'} — "
              f"{time.time()-t0:.0f}s", flush=True)

    for cell in (cells if args.stage == "full" else []):
        fcol, bcol = f"pdd_{cell}", f"pdb_{cell}"
        if fcol in cache.columns and cache[fcol].notna().all():
            print(f"\n[{cell}] 캐시 재사용", flush=True)
            continue
        dcols = cell_cols(cell, env_cols)
        need_an = cell.startswith("an")
        base_feats = [c for c in dcols if c in MEM_UV + MEM_WS + ["ws_gbm"] + QWS_COLS]
        print(f"\n[{cell}] DL 입력 {len(dcols)}열 (base {len(base_feats)}"
              f"{' + env' if cell.endswith('_env') else ''})", flush=True)
        fin = cache[fcol].to_numpy().copy() if fcol in cache.columns else np.full(len(cache), np.nan)
        bas = cache[bcol].to_numpy().copy() if bcol in cache.columns else np.full(len(cache), np.nan)
        for fname, fold in FOLD_LIST:
            if only_folds and fname not in only_folds:
                continue
            sel = (cache["fold"] == fname).to_numpy()
            if not np.isnan(fin[sel]).any():
                continue
            tf = time.time()
            if need_an:
                src = members
            elif cell == "gws_q15":
                src = qws
            elif cell == "gws_hub":
                src = pd.read_parquet(GBMWS_HUB_PATH)
            else:
                src = gbmws
            f_cf = src[(src["fold"] == fname) & (src["split"] == "cf")]
            f_va = src[(src["fold"] == fname) & (src["split"] == "va")]
            wf = merge_feats(
                train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]],
                f_cf, base_feats)
            keys = cache.loc[sel, ["group", "forecast_kst_dtm"]]
            vf = keys.merge(merge_feats(train_df, f_va, base_feats),
                            left_on=["group", "forecast_kst_dtm"],
                            right_on=KEY, how="left")
            assert len(vf) == len(keys), "정렬 실패"
            lab = wf.loc[wf["target"].notna()]
            cov_tr = lab[base_feats[0]].notna().mean()
            cov_va = vf[base_feats[0]].notna().mean()
            assert cov_va > 0.99, f"검증 base 커버 {cov_va:.3f}"

            # ── 1단 DL: 교차적합(잔차 타깃용) + 검증행 ──
            h0 = crossfit_masks(lab)
            lab = lab.copy()
            lab["pd_dl_cf"] = np.nan
            for fit_m, q_m, tg in ((h0, ~h0, "cf01"), (~h0, h0, "cf10")):
                p = fold_predict(lab.loc[fit_m], lab.loc[q_m], dcols, args.seeds,
                                 True, True, f"{cell}:{fname}:{tg}")
                lab.loc[q_m, "pd_dl_cf"] = p
            dl_va = fold_predict(lab, vf, dcols, args.seeds, True, True,
                                 f"{cell}:{fname}:va")
            bas[sel] = dl_va

            # ── 2단 GBM 잔차 (그룹별 · 챔피언 관례) ──
            vf = vf.copy()
            vf["pd_dl_cf"] = dl_va
            res_va = np.full(len(vf), np.nan)
            for g in [1, 2, 3]:
                tr_g = lab.loc[(lab["kpx_group"] == g) & lab["pd_dl_cf"].notna()]
                vm = (vf["kpx_group"] == g).to_numpy()
                if not len(tr_g) or not vm.any():
                    continue
                cap = CAPACITY_KWH[f"kpx_group_{g}"]
                res_va[vm] = gbm_residual(tr_g, vf.loc[vm], gbm_cols, e56,
                                          seeds=args.seeds) / cap
            fin[sel] = dl_va + res_va
            cache[fcol], cache[bcol] = fin, bas
            save_cols(cache, store, [c for c in cols_new if c in cache.columns])
            rmag = np.nanmean(np.abs(res_va)) * 100
            print(f"  [{cell}:{fname}] {time.time()-tf:.0f}s (커버 tr {cov_tr:.3f} / "
                  f"va {cov_va:.3f} | |잔차보정| 평균 {rmag:.2f}%cap)", flush=True)
        cache[fcol], cache[bcol] = fin, bas
    if args.stage == "full":
        save_cols(cache, store, cols_new)
        if set(cells) != set(ALL_CELLS):
            print(f"\n부분 실행 완료 ({cells}) — 취합 채점은 --stage eval | "
                  f"총 소요: {(time.time()-t0)/60:.1f}분", flush=True)
            return

    # ── 채점: PP(챔피언 편향맵·NOSNAP) × 작동점 2개 ──
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
        return score_from_tidy(c, "pred"), score_from_tidy(t[t["fold"] == HOLD], "pred")

    mm = cache["target"].notna().to_numpy()
    champ_raw = 0.65 * cache["cf_ens3"].to_numpy() + 0.35 * cache["dl_es"].to_numpy()
    rows = []
    for scaled in (False, True):
        tag = "제출 작동점 ×.98/.98/.97" if scaled else "미스케일"
        print("\n" + "=" * 108, flush=True)
        print(f"═══ {tag} | 트랙 D 내부 비교 (챔피언 행은 참고 전용) ═══", flush=True)
        ref = sc_of(pp_of(champ_raw, scaled))
        print(f"  {'셀':<20}{'CV':>9}{'Δ참고':>9}{'Δn':>9}{'Δf':>9}{'HOLD':>9}"
              f"{'base단독CV':>11}{'corr(champ)':>12}", flush=True)
        print(f"  {'(참고) champ w65':<20}{ref[0][0]:>9.4f}{'—':>9}{'—':>9}{'—':>9}"
              f"{ref[1][0]:>9.4f}", flush=True)
        for cell in ALL_CELLS:
            fp = pp_of(cache[f"pdd_{cell}"].to_numpy(), scaled)
            bp = pp_of(cache[f"pdb_{cell}"].to_numpy(), scaled)
            s, sb = sc_of(fp), sc_of(bp)
            r = np.corrcoef(cache.loc[mm, f"pdd_{cell}"], champ_raw[mm])[0, 1]
            print(f"  {cell:<20}{s[0][0]:>9.4f}{s[0][0]-ref[0][0]:>+9.4f}"
                  f"{s[0][1]-ref[0][1]:>+9.4f}{s[0][2]-ref[0][2]:>+9.4f}"
                  f"{s[1][0]:>9.4f}{sb[0][0]:>11.4f}{r:>12.4f}", flush=True)
            rows.append(dict(scaled=scaled, cell=cell, cv=s[0][0], hold=s[1][0],
                             base_cv=sb[0][0], d_ref=s[0][0]-ref[0][0],
                             d_n=s[0][1]-ref[0][1], d_f=s[0][2]-ref[0][2],
                             corr_champ=r))

    pd.DataFrame(rows).to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print(f"\n상세 → {CSV_OUT.name} | 총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
