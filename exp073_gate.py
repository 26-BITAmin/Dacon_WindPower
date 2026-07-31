# ═══════════════════════════════════════════════════════
# exp-073 게이트: ECMWF 앙상블(enfo K=16) 집계 피처를 챔피언 GBM base에 투입
#
# 근거:
#   - 파일럿(2024 4계절 60일) 통과: corr(ens16 평균 ws, CF)가 3그룹 전부 HRES 초과
#     (g3 +0.018 최대). K=8이면 이득 절반 소실 → K=16 확정.
#   - FiCR 병목 = base 오차 폭(exp-057) → 앙상블 평균은 D-2 리드타임에서
#     결정론보다 오차가 작은 것이 표준 지식 = 순수 정확도형 입력 개선 후보.
#   - 스프레드 = 불확실성 신호 (기존 피처에 없는 정보 유형).
#
# 피처 변형 2개 (GBM에만 — exp-054 선례, DL 무변경):
#   cf_ens5: +5 = ens_ws, ens_wdsin, ens_wdcos, ens_dis, ens_spread (exp-054 미러 + 스프레드)
#   cf_ens3: +3 = ens_ws, ens_spread, ens_gap(= ens_ws − ec_ws)     (신규 신호 최소 블록)
#
# 게이트: 챔피언 GBM 파라미터(e56)·cln03 가중·3-seed·4-fold+홀드아웃,
#   dl_cln03 블렌드 w{0.65, 0.70}. 판정 Δ = 보드 챔피언(hpo×dC_w65).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp073_gate.py [--check] [--seeds 3] [--jobs 8]
#   --check = 피처 빌드·커버리지·corr 점검만 (GBM 학습 없음 — 부분 수집 상태 스모크용)
# ═══════════════════════════════════════════════════════
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, DATA_DIR
from exp035_raw_dl import FOLD_LIST, CACHE_DIR
from exp049_clean import KEY, PP_NEW
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp056_hpo import RESULT_JSON as E56_JSON, train_gate_fold
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp065_gbm3 import build_train_frame

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ENFO_PARQUET = DATA_DIR / "external" / "enfo_prevruns.parquet"
ENS_CACHE = CACHE_DIR / "exp073_gbm_ens.parquet"
ENS5 = ["ens_ws", "ens_wdsin", "ens_wdcos", "ens_dis", "ens_spread"]
ENS3 = ["ens_ws", "ens_spread", "ens_gap"]


def build_ens_frame(min_members=8):
    """(init_date, step, group, param, member, val) → 시간별 KST 앙상블 집계 피처.
    멤버별 u/v를 시간 보간 후 ws 계산 → 멤버 축 평균/표준편차 (exp-054 보간 방식 계승)."""
    df = pd.read_parquet(ENFO_PARQUET)
    w = df.pivot_table(index=["init_date", "group", "member", "step"],
                       columns="param", values="val").reset_index()
    target_h = np.arange(28, 52, dtype=float)   # D 01:00 ~ D+1 00:00 KST (24시간)
    frames, n_skip = [], 0
    for (init_d, g), sub in w.groupby(["init_date", "group"]):
        us, vs = [], []
        for _, ms in sub.groupby("member"):
            ms = ms.dropna(subset=["10u", "10v"]).sort_values("step")
            if len(ms) < 2:
                continue
            h = ms["step"].to_numpy(dtype=float)
            us.append(np.interp(target_h, h, ms["10u"].to_numpy()))
            vs.append(np.interp(target_h, h, ms["10v"].to_numpy()))
        if len(us) < min_members:
            n_skip += 1
            continue
        U, V = np.asarray(us), np.asarray(vs)
        WS = np.hypot(U, V)
        mu, mv = U.mean(axis=0), V.mean(axis=0)
        wd = np.arctan2(mu, mv)   # 벡터 방향 (사인·코사인만 사용)
        kst = (pd.Timestamp(init_d) + pd.Timedelta(hours=12 + 9)
               + pd.to_timedelta(target_h, unit="h"))
        frames.append(pd.DataFrame({
            "forecast_kst_dtm": kst, "kpx_group": g,
            "ens_ws": WS.mean(axis=0), "ens_spread": WS.std(axis=0, ddof=1),
            "ens_wdsin": np.sin(wd), "ens_wdcos": np.cos(wd), "ens_n": len(us)}))
    out = pd.concat(frames, ignore_index=True) \
            .drop_duplicates(["forecast_kst_dtm", "kpx_group"], keep="last")
    print(f"ens frame: {len(out)}행 | 스킵(멤버<{min_members}) {n_skip}건 | "
          f"멤버수 중앙값 {out['ens_n'].median():.0f}", flush=True)
    return out.drop(columns="ens_n")


def prepare_frame():
    train_df, use_cols = build_train_frame()          # 352 (spd + EC)
    ens = build_ens_frame()
    train_df = train_df.merge(ens, on=["forecast_kst_dtm", "kpx_group"], how="left")
    train_df["ens_dis"] = (train_df["ens_ws"] - train_df["ldaps_ws_10m"]).abs()
    train_df["ens_gap"] = train_df["ens_ws"] - train_df["ec_ws"]
    cov = train_df.groupby(train_df["forecast_kst_dtm"].dt.year)["ens_ws"] \
                  .apply(lambda d: d.notna().mean().round(3))
    print("연도별 enfo 커버리지:\n" + cov.to_string(), flush=True)
    for g in [1, 2, 3]:
        m = train_df["target"].notna() & train_df["ens_ws"].notna() & (train_df["kpx_group"] == g)
        cap = CAPACITY_KWH[f"kpx_group_{g}"]
        cf = train_df.loc[m, "target"] / cap
        ce = np.corrcoef(train_df.loc[m, "ens_ws"], cf)[0, 1]
        ch = np.corrcoef(train_df.loc[m, "ec_ws"], cf)[0, 1]
        print(f"  그룹{g}: corr(ens_ws, CF) {ce:.4f} vs corr(ec_ws, CF) {ch:.4f} "
              f"(라벨행 {m.sum()})", flush=True)
    return train_df, use_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-073 게이트: enfo 앙상블 집계 피처 → 챔피언 GBM base ###", flush=True)

    train_df, use_cols = prepare_frame()
    print(f"입력 준비 완료 — {time.time()-t0:.0f}s", flush=True)
    if args.check:
        print("[--check 모드 종료 — GBM 학습 없음]")
        return
    e56 = json.loads(E56_JSON.read_text())

    cache, champ_bias = load_cache_full()
    want = {"cf_ens5": use_cols + ENS5, "cf_ens3": use_cols + ENS3}
    have = merge_cached_cols(cache, ENS_CACHE, list(want))
    for col, cols in want.items():
        if col in have:
            print(f"[{col}] 캐시 재사용", flush=True)
            continue
        preds = []
        for fname, fold in FOLD_LIST:
            tf = time.time()
            p = train_gate_fold(train_df, cols, fold, args.seeds, args.jobs,
                                e56["lgb"], e56["lgb_n"], e56["xgb"], e56["xgb_n"])
            preds.append(p.rename(columns={"cf_hpo": col}))
            print(f"  [{col}:{fname}] {time.time()-tf:.0f}s", flush=True)
        p = pd.concat(preds, ignore_index=True)
        m = cache[KEY].merge(p, on=KEY, how="left")
        assert m[col].notna().all(), "정렬 실패"
        cache[col] = m[col].to_numpy()
        have.append(col)
        save_cols(cache, ENS_CACHE, have)
        print(f"  캐시 저장 → {ENS_CACHE.name}", flush=True)

    mm = cache["target"].notna()
    for col in want:
        c = np.corrcoef(cache.loc[mm, col], cache.loc[mm, "cf_hpo"])[0, 1]
        shift = (cache[col].to_numpy() - cache["cf_hpo"].to_numpy()).mean() * 100
        print(f"  corr({col}, cf_hpo) = {c:.4f} | 평균 이동 {shift:+.3f}%cap", flush=True)

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} (1-NMAE {ref['cv'][1]:.4f} "
          f"FiCR {ref['cv'][2]:.4f}) | HOLD {ref['hold'][0]:.4f}  (기대 0.6556/0.6535)", flush=True)
    res = {"hpo_solo": ev("cf_hpo", None, 1.0, PP_NEW),
           "ens5_solo": ev("cf_ens5", None, 1.0, PP_NEW),
           "ens3_solo": ev("cf_ens3", None, 1.0, PP_NEW)}
    for w in (0.65, 0.70):
        res[f"ens5_w{int(w*100)}"] = ev("cf_ens5", "dl_cln03", w, PP_NEW)
        res[f"ens3_w{int(w*100)}"] = ev("cf_ens3", "dl_cln03", w, PP_NEW)
    print_table(res, ref)
    for col, tag in (("cf_ens5", "ens5"), ("cf_ens3", "ens3")):
        d_cv = res[f"{tag}_solo"]["cv"][0] - res["hpo_solo"]["cv"][0]
        d_hd = res[f"{tag}_solo"]["hold"][0] - res["hpo_solo"]["hold"][0]
        print(f"GBM 단독 델타 ({tag}−hpo): ΔCV {d_cv:+.4f} / Δhold {d_hd:+.4f}")
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
