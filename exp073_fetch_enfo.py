# ═══════════════════════════════════════════════════════
# exp-073 데이터 수집: ECMWF 앙상블(enfo) 10u/10v — D-2 12z, 멤버 서브샘플 (K=16)
#
# 배경: exp-054는 oper(HRES)만 수집. D-2 리드타임에서 앙상블 평균은 결정론보다
#   오차가 작은 것이 표준 지식 — FiCR 병목(base 오차 폭, exp-057)을 직접 좁히는
#   입력 개선 후보. 스프레드 = 예보 불확실성 부가 신호. 규정 = exp-054와 동일.
#
# ⚠ 구조 변경 (2026-07-30): eccodes DLL이 애플리케이션 제어 정책에 차단됨
#   (exp-054 당시엔 동작 — 정책 변경 추정). GRIB이 CCSDS/AEC(템플릿 5.42)라 순수
#   파이썬 디코드 불가 → **Windows = HTTP 수집(raw 저장) / WSL = eccodes 디코드** 분리:
#   --stage fetch  (Windows): raw 메시지 → external/enfo_raw/{일자}.bin + .json 매니페스트
#   (디코드는 exp073_decode_wsl.py를 WSL python3로 실행 — eccodes 2.45 apt 설치 확인)
#   --stage pack   (Windows): WSL이 쓴 external/enfo_decoded/{일자}.csv → parquet 취합
#   --stage analyze(Windows): 파일럿 신호 점검 — corr(ens평균 ws, CF) vs HRES(ec_ws)
#
# 실행: C:\miniconda3\python.exe -X utf8 exp073_fetch_enfo.py --stage fetch --pilot
#       wsl -e python3 exp073_decode_wsl.py            (프로젝트 루트에서)
#       C:\miniconda3\python.exe -X utf8 exp073_fetch_enfo.py --stage pack --pilot
# ═══════════════════════════════════════════════════════
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import pandas as pd

from exp005_qopt import DATA_DIR
from exp019_multigrid import _centroids
from exp054_fetch_ecmwf import http_get   # truststore inject 포함

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

S3 = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
STEPS = list(range(27, 52, 3))
PARAMS = ("10u", "10v")
RAW_DIR = DATA_DIR / "external" / "enfo_raw"
DEC_DIR = DATA_DIR / "external" / "enfo_decoded"


def pilot_targets():
    out = []
    for m0 in (1, 4, 7, 10):            # 2024년 4계절 × 15일
        out += [date(2024, m0, 5) + timedelta(days=i) for i in range(15)]
    return out


def match_entry(e, k_members):
    if e.get("levtype") != "sfc" or e.get("param") not in PARAMS:
        return None
    if e.get("type") == "cf":
        return 0
    if e.get("type") == "pf" and int(e.get("number", 999)) < k_members:
        return int(e["number"])
    return None


def fetch_day(init_d, k_members):
    """raw 메시지를 {일자}.bin에 이어 쓰고 매니페스트 반환 (디코드 없음)"""
    d = init_d.strftime("%Y%m%d")
    roots = (["12z/0p4-beta/enfo", "12z/ifs/0p4-beta/enfo", "12z/ifs/0p25/enfo"]
             if init_d < date(2024, 2, 1) else
             ["12z/ifs/0p25/enfo", "12z/ifs/0p4-beta/enfo", "12z/0p4-beta/enfo"])
    root = None
    idx0 = None
    for cand in roots:
        idx0 = http_get(f"{S3}/{d}/{cand}/{d}120000-27h-enfo-ef.index", retries=2)
        if idx0:
            root = cand
            break
    if root is None:
        return False, ["no-index"]
    manifest, fails, blobs = [], [], []
    pos = 0
    for s in STEPS:
        raw = idx0 if s == 27 else http_get(f"{S3}/{d}/{root}/{d}120000-{s}h-enfo-ef.index")
        if raw is None:
            fails.append(f"{s}-index")
            continue
        entries = {}
        for line in raw.decode("utf-8", "ignore").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            mem = match_entry(e, k_members)
            if mem is not None:
                entries[(e["param"], mem)] = (int(e["_offset"]), int(e["_length"]))
        gurl = f"{S3}/{d}/{root}/{d}120000-{s}h-enfo-ef.grib2"
        for (p, mem), rng in sorted(entries.items()):
            b = http_get(gurl, rng=rng)
            if b is None:
                fails.append(f"{s}-{p}-m{mem}")
            else:
                blobs.append(b)
                manifest.append(dict(step=s, param=p, member=mem, offset=pos, length=len(b)))
                pos += len(b)
    if not manifest:
        return False, fails
    (RAW_DIR / f"{d}.bin").write_bytes(b"".join(blobs))
    (RAW_DIR / f"{d}.json").write_text(json.dumps(dict(init_date=d, fails=fails,
                                                       items=manifest)))
    return True, fails


def stage_fetch(args):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cent = _centroids()
    (RAW_DIR / "centroids.json").write_text(json.dumps(
        {str(g): [float(cent.loc[g, "lat"]), float(cent.loc[g, "lon"])] for g in [1, 2, 3]}))
    if args.pilot:
        targets = pilot_targets()
    else:
        d0, d1 = date.fromisoformat(args.start), date.fromisoformat(args.end)
        targets = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    inits = sorted({t - timedelta(days=2) for t in targets})
    done = {p.stem for p in RAW_DIR.glob("*.json")} | \
           {p.stem for p in DEC_DIR.glob("*.csv")} if DEC_DIR.exists() else \
           {p.stem for p in RAW_DIR.glob("*.json")}
    todo = [d for d in inits if d.strftime("%Y%m%d") not in done]
    print(f"수집 대상 {len(todo)}/{len(inits)}일 × K={args.members} (raw 저장, 디코드는 WSL)",
          flush=True)
    t0 = time.time()
    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_day, d, args.members): d for d in todo}
        for fut in as_completed(futs):
            ok, fails = fut.result()
            n += 1
            if n % 10 == 0:
                el = time.time() - t0
                print(f"  진행 {n}/{len(todo)} ({el/60:.1f}분, {n/max(el,1)*3600:.0f}일/시간)",
                      flush=True)
    (RAW_DIR / "FETCH_DONE").write_text(str(time.time()))
    print(f"fetch 완료 {n}일 — {(time.time()-t0)/60:.1f}분. 다음: WSL 디코드", flush=True)


def stage_pack(args):
    files = sorted(DEC_DIR.glob("*.csv"))
    assert files, "WSL 디코드 산출물 없음 — exp073_decode_wsl.py를 먼저 실행"
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) \
           .drop_duplicates(["init_date", "step", "group", "param", "member"])
    df["init_date"] = pd.to_datetime(df["init_date"], format="%Y%m%d")
    out = DATA_DIR / "external" / ("enfo_prevruns_pilot.parquet" if args.pilot
                                   else "enfo_prevruns.parquet")
    df.to_parquet(out, index=False)
    print(f"pack: {len(files)}일 → {out.name} ({len(df)}행)", flush=True)


def stage_analyze(args):
    from exp065_gbm3 import build_train_frame
    pil = pd.read_parquet(DATA_DIR / "external" / "enfo_prevruns_pilot.parquet")
    w = pil.pivot_table(index=["init_date", "step", "group", "member"],
                        columns="param", values="val").reset_index()
    w["ws"] = np.hypot(w["10u"], w["10v"])
    agg = w.groupby(["init_date", "step", "group"])["ws"] \
           .agg(ens_ws="mean", ens_spread="std", n="count").reset_index()
    # valid KST = init 12z UTC + step + 9h
    agg["forecast_kst_dtm"] = agg["init_date"] + pd.to_timedelta(agg["step"] + 12 + 9, unit="h")
    train_df, _ = build_train_frame()
    sub = train_df.loc[train_df["target"].notna(),
                       ["forecast_kst_dtm", "kpx_group", "target", "ldaps_ws_10m", "ec_ws"]] \
                  .rename(columns={"kpx_group": "group"})
    m = agg.merge(sub, on=["forecast_kst_dtm", "group"], how="inner")
    from exp005_qopt import CAPACITY_KWH
    m["cf"] = m["target"] / m["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"])
    print(f"매칭 {len(m)}행 (3h 간격), 멤버수 중앙값 {m['n'].median():.0f}", flush=True)
    for g in (1, 2, 3):
        s = m[m["group"] == g]
        ce = np.corrcoef(s["ens_ws"], s["cf"])[0, 1]
        ch = np.corrcoef(s["ec_ws"], s["cf"])[0, 1]
        cl = np.corrcoef(s["ldaps_ws_10m"], s["cf"])[0, 1]
        chh = np.corrcoef(s["ens_ws"], s["ec_ws"])[0, 1]
        print(f"  g{g}: corr(CF) — ens16 {ce:.4f} vs HRES {ch:.4f} vs LDAPS {cl:.4f} | "
              f"corr(ens, HRES) {chh:.4f} | spread 평균 {s['ens_spread'].mean():.2f}m/s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["fetch", "pack", "analyze"], default="fetch")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--members", type=int, default=16)
    ap.add_argument("--start", default="2023-01-20")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.stage == "fetch":
        stage_fetch(args)
    elif args.stage == "pack":
        stage_pack(args)
    else:
        stage_analyze(args)


if __name__ == "__main__":
    main()
