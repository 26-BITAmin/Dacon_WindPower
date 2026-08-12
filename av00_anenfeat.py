# ═══════════════════════════════════════════════════════
# av00: 벡터 AnEn(비모수 K=30 멤버) 분포 피처 빌더 — DL 입력용 (anen_input ablation 1/3)
#
# 사용자 지시 (2026-08-08):
#   "BIV이거를 해봤는데 성능이 별로 안좋았음. BIV 대신에 anen_vec(비모수 30멤버)로 다시.
#    BIV를 anen_vec으로 교체만 하면 되는거임."
#   → wi00(BIV-DRN)의 자리를 AN-22 `anen_vec`으로 그대로 교체. 그 외 전부 동결.
#
# 근거 (AN-22, MODEL.md): 동일 행·동일 피처 통제에서 anen_vec은
#   ws MAE 1.1750 · ws CRPS 0.8490 으로 **전 모델 1위**(drn 1.2463/0.9061 대비).
#   단 풍향은 23.9°로 최악 → 발전량(풍속의 함수)에는 BIV가 못 준 채널이 있을 수 있음.
#
# 하는 일: 폴드(Q1~Q4 + 홀드)별로 그룹별 벡터 아날로그 K=30 멤버를 뽑아
#   (1) 원본 멤더 60열(u,v × 30)  → av00_anen_members.parquet  (재요약용 자산)
#   (2) 요약 피처 15열            → av00_anen_feats.parquet    (DL 입력)
#
#   아날로그 구성 = AN-22 `vec_analogs` 그대로: W14[g] 가중 · twin=1 · hw=4 · K=30 ·
#   σ 정규화 거리, 풀 = u_obs 있는 행, 라벨 = 나셀 벡터 (build_scada_vec).
#
# ★ 스태킹 리크 방지 (wi00과 동일 관례 — AnEn에서는 더 치명적):
#   AnEn은 최근접 조회이므로 학습행 자신이 풀에 있으면 거리 0으로 **자기 실측을 복사**한다.
#   DL 학습행용 = 윈도우 시간중앙 2분할 교차적합 (A풀→B구간 / B풀→A구간)
#   검증행용     = 전체 윈도우 풀 (실운영과 동일 조건)
#
# ★ 거리 계산 최적화: an22의 컬럼×시차 outer subtract를 ‖f‖²−2f·a+‖a‖² 항등식(BLAS)으로
#   교체 + 질의행 청킹. 수치 동일성은 --verify로 원본 vec_analogs와 대조 검증.
#
# 실행(루트에서):
#   C:\miniconda3\python.exe -X utf8 "open (1)/testing/anen_input/av00_anenfeat.py" --verify
#   C:\miniconda3\python.exe -X utf8 "open (1)/testing/anen_input/av00_anenfeat.py"
#   폴드 분할 병렬: ... --folds fold_2024Q1,fold_2024Q2 --tag a
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parents[1]))          # testing/ (an14·an22)

import numpy as np
import pandas as pd

from an12_emos import build_scada_vec
from an14_featsel import entry_cols, lag_block
from an14b_final import W14
from exp035_raw_dl import FOLD_LIST
from exp065_gbm3 import build_train_frame

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

K_ANEN = 30                       # 사용자 지정 "비모수 30멤버" (AN-11 CRPS 정점과 일치)
TWIN = 1                          # AN-14/AN-22 규약
HW = 4                            # 시각창 ±4h (AN-14 HW)
CHUNK = 2048                      # 질의행 청크 (거리행렬 메모리 상한)

KEY = ["kpx_group", "forecast_kst_dtm"]
# ── 피처 형태 3종 (BIV b5/b9 대응 + 비모수 고유 채널) ──
A5 = ["an_mu_u", "an_mu_v", "an_sig_u", "an_sig_v", "an_rho"]
A9 = A5 + ["an_ws", "an_dsin", "an_dcos", "an_sigws"]
A15 = A9 + ["an_wq10", "an_wq25", "an_wq50", "an_wq75", "an_wq90", "an_dspread"]
FORMS = {"a5": A5, "a9": A9, "a15": A15}
MEM_COLS = [f"m{k}{c}" for k in range(K_ANEN) for c in ("u", "v")]


# ───────────────────────────────────────────────
# 1) 아날로그 멤버 추출
# ───────────────────────────────────────────────
def analog_members(pool_g, query_g, w_map, K=K_ANEN, twin=TWIN, hw=HW, chunk=CHUNK):
    """AN-22 vec_analogs와 동일 정의 · BLAS 거리 + 청킹. 반환 [Nq, K, 2]."""
    ent = [(e, w) for e, w in w_map.items() if w > 0]
    cols = sum((entry_cols(e) for e, _ in ent), [])
    A = lag_block(pool_g, cols, twin)                     # [Np, V, L]
    F = lag_block(query_g, cols, twin)                    # [Nq, V, L]
    ok = pool_g["u_obs"].notna().to_numpy()
    ti = np.flatnonzero(ok)
    sig = np.nanstd(pool_g[cols].to_numpy(dtype=np.float64), axis=0)

    # 컬럼별 가중 (an22: 쌍 엔트리는 w/2, σ로 정규화)
    wcol, ci = [], 0
    for e, w in ent:
        nc = len(entry_cols(e))
        for k in range(nc):
            wcol.append(w * (0.5 if nc == 2 else 1.0) / max(sig[ci + k], 1e-9))
        ci += nc
    wcol = np.asarray(wcol, dtype=np.float64)

    Ap = np.ascontiguousarray(A[ti])                      # [Nt, V, L]
    An2 = (Ap ** 2).sum(2)                                # [Nt, V]
    h_tr = pool_g["forecast_kst_dtm"].dt.hour.to_numpy()[ti]
    h_va = query_g["forecast_kst_dtm"].dt.hour.to_numpy()
    Y = pool_g.loc[ok, ["u_obs", "v_obs"]].to_numpy()
    out = np.empty((len(query_g), K, 2), dtype=np.float32)

    for s in range(0, len(query_g), chunk):
        Fc = F[s:s + chunk]
        Fn2 = (Fc ** 2).sum(2)
        D = np.zeros((len(Fc), len(ti)), dtype=np.float64)
        for j in range(Ap.shape[1]):
            sq = Fn2[:, j][:, None] + An2[:, j][None, :] \
                - 2.0 * (Fc[:, j, :] @ Ap[:, j, :].T)
            np.maximum(sq, 0.0, out=sq)
            D += wcol[j] * np.sqrt(sq)
        dh = np.abs(h_va[s:s + chunk][:, None] - h_tr[None, :])
        D[np.minimum(dh, 24 - dh) > hw] = np.inf
        assert (np.isfinite(D).sum(1) >= K).all(), "시각창 내 후보가 K 미만"
        idx = np.argpartition(D, K - 1, axis=1)[:, :K]
        out[s:s + chunk] = Y[idx]
    return out


# ───────────────────────────────────────────────
# 2) 멤버 30개 → DL 입력 요약
#    ★ 이 함수가 "비모수 30멤버를 DL이 어떤 언어로 보느냐"를 정하는 자리다.
#      a5  = BIV b5와 1:1 대응 (같은 5개 숫자의 다른 추정기)
#      a9  = b9 대응. 단 an_ws는 |평균벡터|가 아니라 **멤버 풍속의 평균** —
#            |평균벡터|는 an_mu_u/an_mu_v의 결정함수라 새 정보가 0인 반면,
#            E[|V|]는 AN-22에서 실제로 이긴 양이다 (Jensen 갭 = 스프레드 정보).
#      a15 = 비모수 고유 채널. 파워커브가 비선형이라 E[P] = E[f(v)] ≠ f(E[v]) —
#            풍속 분포의 모양(분위수)이 물리적으로 맞는 채널.
# ───────────────────────────────────────────────
def summarize_members(M):
    """M [N, K, 2] → 15열 DataFrame."""
    u, v = M[:, :, 0].astype(np.float64), M[:, :, 1].astype(np.float64)
    mu_u, mu_v = u.mean(1), v.mean(1)
    sd_u, sd_v = u.std(1, ddof=1), v.std(1, ddof=1)
    cov = ((u - mu_u[:, None]) * (v - mu_v[:, None])).sum(1) / (u.shape[1] - 1)
    rho = np.clip(cov / np.maximum(sd_u * sd_v, 1e-9), -1, 1)

    spd = np.hypot(u, v)                                  # 멤버별 풍속
    sp_mean, sp_sd = spd.mean(1), spd.std(1, ddof=1)
    q = np.quantile(spd, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1)

    norm = np.maximum(np.hypot(mu_u, mu_v), 1e-6)         # 평균벡터 방향
    usp = np.maximum(spd, 1e-9)
    R = np.hypot((u / usp).mean(1), (v / usp).mean(1))    # 단위벡터 평균 길이
    return pd.DataFrame({
        "an_mu_u": mu_u, "an_mu_v": mu_v, "an_sig_u": sd_u, "an_sig_v": sd_v,
        "an_rho": rho,
        "an_ws": sp_mean, "an_dsin": mu_v / norm, "an_dcos": mu_u / norm,
        "an_sigws": sp_sd,
        "an_wq10": q[0], "an_wq25": q[1], "an_wq50": q[2], "an_wq75": q[3],
        "an_wq90": q[4], "an_dspread": 1.0 - R,           # 0=완전정렬, 1=완전분산
    })


# ───────────────────────────────────────────────
def anen_block(pool, query, tag):
    """그룹별로 아날로그를 뽑아 query 순서대로 멤버 배열 [Nq, K, 2] 반환."""
    mem = np.full((len(query), K_ANEN, 2), np.nan, dtype=np.float32)
    qpos = pd.MultiIndex.from_frame(query[KEY])
    for g in (1, 2, 3):
        pg = pool[pool["kpx_group"] == g].sort_values(
            "forecast_kst_dtm").reset_index(drop=True)
        qg = query[query["kpx_group"] == g].sort_values(
            "forecast_kst_dtm").reset_index(drop=True)
        if len(qg) == 0:
            continue
        M = analog_members(pg, qg, W14[g])
        pos = qpos.get_indexer(pd.MultiIndex.from_frame(qg[KEY]))
        assert (pos >= 0).all(), f"{tag} g{g}: 질의행 정렬 실패"
        mem[pos] = M
    assert np.isfinite(mem).all(), f"{tag}: 멤버 결측"
    return mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=str, default="")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--verify", action="store_true",
                    help="원본 an22.vec_analogs와 수치 동일성 대조 후 종료")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    feat_out = HERE.parent / f"av00_anen_feats{suffix}.parquet"
    mem_out = HERE.parent / f"av00_anen_members{suffix}.parquet"
    run_folds = (tuple(f for f in args.folds.split(",") if f) if args.folds
                 else tuple(fn for fn, _ in FOLD_LIST))
    assert all(f in dict(FOLD_LIST) for f in run_folds), f"폴드 이름 오류: {run_folds}"

    t0 = time.time()
    print(f"### av00: 벡터 AnEn K={K_ANEN} 분포 피처 (교차적합) — folds={run_folds} ###",
          flush=True)
    train_df, _ = build_train_frame()
    vec = build_scada_vec()
    for g in (1, 2, 3):
        train_df = train_df.merge(
            vec[g].assign(kpx_group=g), on=KEY, how="left", suffixes=("", "_d"))
        for c in ["park_ws", "u_obs", "v_obs"]:
            if f"{c}_d" in train_df:
                train_df[c] = train_df[c].fillna(train_df.pop(f"{c}_d"))
    need = sorted({c for g in (1, 2, 3) for e in W14[g] for c in entry_cols(e)})
    miss = [c for c in need if c not in train_df.columns]
    assert not miss, f"W14 입력 부재: {miss}"
    print(f"프레임 {train_df.shape} | AnEn 입력 {len(need)}컬럼 | "
          f"실측 행 {train_df['u_obs'].notna().sum():,} — {time.time()-t0:.0f}s", flush=True)

    # ── 수치 동일성 검증 (원본 an22 구현과 대조) ──
    if args.verify:
        from an22_shapes import vec_analogs
        fold = dict(FOLD_LIST)["fold_2024Q1"]
        win = train_df[train_df["forecast_kst_dtm"] <= fold["train_end"]]
        obs = win[win["u_obs"].notna()]
        for g in (1, 2, 3):
            pg = obs[obs["kpx_group"] == g].sort_values(
                "forecast_kst_dtm").reset_index(drop=True)
            qg = train_df[(train_df["kpx_group"] == g) &
                          (train_df["forecast_kst_dtm"] >= fold["valid_start"]) &
                          (train_df["forecast_kst_dtm"] <= fold["valid_end"])
                          ].sort_values("forecast_kst_dtm").reset_index(drop=True)
            ref = vec_analogs(pg, qg, W14[g], K=K_ANEN, twin=TWIN, hw=HW)
            new = analog_members(pg, qg, W14[g])
            srt = lambda M: np.sort(M.reshape(len(M), -1), axis=1)
            same = np.isclose(srt(ref), srt(new), atol=1e-6).all(1)
            print(f"  [g{g}] 질의 {len(qg):,} · 멤버집합 일치 {same.mean()*100:.2f}% "
                  f"| 평균풍속 차이 {abs(np.hypot(*ref.transpose(2,0,1)).mean() - np.hypot(*new.transpose(2,0,1)).mean()):.2e}",
                  flush=True)
        print(f"\n### --verify 종료 ({(time.time()-t0)/60:.1f}분) ###")
        return

    done_f = pd.read_parquet(feat_out) if feat_out.exists() else None
    parts_f = [] if done_f is None else [done_f]
    parts_m = [] if not mem_out.exists() else [pd.read_parquet(mem_out)]

    for fname, fold in FOLD_LIST:
        if fname not in run_folds:
            continue
        if done_f is not None and (done_f["fold"] == fname).any():
            print(f"  [{fname}] 캐시 재사용", flush=True)
            continue
        tf = time.time()
        win = train_df[train_df["forecast_kst_dtm"] <= fold["train_end"]]
        va = train_df[(train_df["forecast_kst_dtm"] >= fold["valid_start"]) &
                      (train_df["forecast_kst_dtm"] <= fold["valid_end"])]
        obs = win[win["u_obs"].notna()]
        mid = obs["forecast_kst_dtm"].quantile(0.5)
        A, B = obs[obs["forecast_kst_dtm"] <= mid], obs[obs["forecast_kst_dtm"] > mid]
        win_a = win[win["forecast_kst_dtm"] <= mid]     # ← B풀이 예측
        win_b = win[win["forecast_kst_dtm"] > mid]      # ← A풀이 예측

        blocks_f, blocks_m = [], []
        for pool, query, tag in ((A, win_b, "cf(A→후반)"), (B, win_a, "cf(B→전반)"),
                                 (obs, va, "va(전체→검증)")):
            M = anen_block(pool, query, tag)
            key = query[KEY].reset_index(drop=True)
            split = "va" if tag.startswith("va") else "cf"
            blocks_f.append(pd.concat(
                [key.assign(split=split), summarize_members(M)], axis=1))
            blocks_m.append(pd.concat(
                [key.assign(split=split),
                 pd.DataFrame(M.reshape(len(M), -1), columns=MEM_COLS)], axis=1))
            print(f"    {tag}: 풀 {pool['u_obs'].notna().sum():,} → 예측 {len(query):,} "
                  f"({time.time()-tf:.0f}s 누적)", flush=True)

        out_f = pd.concat(blocks_f, ignore_index=True).assign(fold=fname)
        out_m = pd.concat(blocks_m, ignore_index=True).assign(fold=fname)
        assert not out_f.duplicated(KEY).any(), "행 중복"
        parts_f.append(out_f)
        parts_m.append(out_m)
        pd.concat(parts_f, ignore_index=True).to_parquet(feat_out, index=False)
        pd.concat(parts_m, ignore_index=True).to_parquet(mem_out, index=False)
        print(f"  [{fname}] {len(out_f):,}행 | μws {out_f['an_ws'].mean():.2f} m/s | "
              f"σu/σv {out_f['an_sig_u'].mean():.2f}/{out_f['an_sig_v'].mean():.2f} | "
              f"σws {out_f['an_sigws'].mean():.2f} | 방향분산 {out_f['an_dspread'].mean():.3f} "
              f"| {time.time()-tf:.0f}s", flush=True)

    full = pd.concat(parts_f, ignore_index=True)
    print(f"\n저장: {feat_out.name} {full.shape} · {mem_out.name} "
          f"| 총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
