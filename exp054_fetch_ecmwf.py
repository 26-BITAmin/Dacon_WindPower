# ═══════════════════════════════════════════════════════
# exp-054 데이터 수집: ECMWF open-data 아카이브 (AWS S3 ecmwf-forecasts) — IFS 예보
#
# 규정 적합성 (exp-048 프로브에서 남은 "기관 원본 아카이브" 경로):
#   - 사용 런: 타깃 KST일 D에 대해 **D-2일 12z UTC** 초기화 oper 런.
#     S3 LastModified 실측 18:45 UTC(+6h45m) = D-1 03:45 KST 공개 → 예측기준시점
#     (전날 13:00 KST) 이전 발행 완료. 재분석·실측 아님 (LDAPS/GFS와 동일 성격의 예보).
#   - 라이선스: ECMWF open data = CC-BY 4.0.
#   - 커버리지 (2026-07-28 프로브): 2023-01-18 ~ 현재 (2022 없음 — 그룹1/2는 1년 결측.
#     선례: exp-020 JMA MSM도 2023+ 부분 커버로 LB 전이 성공).
#   - 변수: 10u/10v (sfc). 100m 풍속은 2023 아카이브에 없어 제외 (전 기간 일관 변수만).
#
# 방식: 스텝별 .index(JSON lines)에서 10u/10v 메시지의 offset/length를 읽어
#   바이트 레인지 GET (~0.6MB/메시지, 전체 GRIB 50MB 대비 1/40) → eccodes로 디코드 →
#   그룹 centroid 이중선형 보간값만 저장. 해상도 시대(0p4→0p25) 무관하게 지점값으로 통일.
#   스텝 27~51h(3시간 간격 9개) = KST D일 01~24시 커버 (시간별 보간은 피처 단계에서).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp054_fetch_ecmwf.py [--start 2023-01-20] [--end 2025-12-31] [--smoke]
# 출력: open (1)/external/ecmwf_prevruns.parquet (init_date, step, group, u10, v10)
# ═══════════════════════════════════════════════════════
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np
import pandas as pd
import truststore
truststore.inject_into_ssl()   # 이 머신은 로컬 TLS 가로채기 → Windows 인증서 저장소 필요
import requests

from exp005_qopt import DATA_DIR
from exp019_multigrid import _centroids

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

S3 = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
OUT = DATA_DIR / "external" / "ecmwf_prevruns.parquet"
MISS = DATA_DIR / "external" / "ecmwf_missing.json"
STEPS = list(range(27, 52, 3))          # 27~51h
PARAMS = ("10u", "10v")                  # --plevel 시 925hPa u/v로 교체
PLEVEL = False
ROOT_CANDIDATES = ["12z/ifs/0p25/oper", "12z/ifs/0p4-beta/oper", "12z/0p4-beta/oper"]


def match_entry(e):
    """index 라인이 수집 대상인지 (sfc 10u/10v 또는 pl 925 u/v)"""
    if PLEVEL:
        return e.get("levtype") == "pl" and str(e.get("levelist")) == "925" and e.get("param") in ("u", "v")
    return e.get("levtype") == "sfc" and e.get("param") in ("10u", "10v")

SESS = requests.Session()
SESS.headers["User-Agent"] = "baram2026-research"


def http_get(url, rng=None, retries=3):
    for a in range(retries):
        try:
            h = {"Range": f"bytes={rng[0]}-{rng[0]+rng[1]-1}"} if rng else {}
            r = SESS.get(url, headers=h, timeout=60)
            if r.status_code in (200, 206):
                return r.content
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 * (a + 1))
    return None


def bilinear(msg_bytes, pts):
    """GRIB 메시지에서 각 (lat, lon) 지점의 이중선형 보간값. 반환: list[float]"""
    import eccodes as ec
    gid = ec.codes_new_from_message(msg_bytes)
    try:
        ni = ec.codes_get(gid, "Ni")
        nj = ec.codes_get(gid, "Nj")
        lat0 = ec.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
        lon0 = ec.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
        dlat = ec.codes_get(gid, "jDirectionIncrementInDegrees")
        dlon = ec.codes_get(gid, "iDirectionIncrementInDegrees")
        jpos = ec.codes_get(gid, "jScansPositively")
        vals = ec.codes_get_values(gid).reshape(nj, ni)
    finally:
        ec.codes_release(gid)
    slat = dlat if jpos else -dlat
    out = []
    for lat, lon in pts:
        fi = ((lon - lon0) % 360.0) / dlon
        fj = (lat - lat0) / slat
        i0, j0 = int(np.floor(fi)), int(np.floor(fj))
        wi, wj = fi - i0, fj - j0
        j0 = min(max(j0, 0), nj - 2)
        v = ((1 - wi) * (1 - wj) * vals[j0, i0 % ni] + wi * (1 - wj) * vals[j0, (i0 + 1) % ni]
             + (1 - wi) * wj * vals[j0 + 1, i0 % ni] + wi * wj * vals[j0 + 1, (i0 + 1) % ni])
        out.append(float(v))
    return out


def fetch_day(init_d, root_hint=None):
    """한 초기화일의 (step, param)별 메시지 bytes 수집. 반환: (root, {(step,param): bytes}, 실패목록)"""
    d = init_d.strftime("%Y%m%d")
    # 시대별 경로 우선순위: 2024-02 이전은 0p4-beta가 먼저 (404 낭비 제거)
    if init_d < date(2024, 2, 1):
        roots = ["12z/0p4-beta/oper", "12z/ifs/0p4-beta/oper", "12z/ifs/0p25/oper"]
    else:
        roots = list(ROOT_CANDIDATES)
    if root_hint:
        roots = [root_hint] + [r for r in roots if r != root_hint]
    root = None
    idx0 = None
    for cand in roots:
        idx0 = http_get(f"{S3}/{d}/{cand}/{d}120000-27h-oper-fc.index", retries=2)
        if idx0:
            root = cand
            break
    if root is None:
        return None, {}, [(s, p) for s in STEPS for p in PARAMS]
    msgs, fails = {}, []
    for s in STEPS:
        raw = idx0 if s == 27 else http_get(f"{S3}/{d}/{root}/{d}120000-{s}h-oper-fc.index")
        if raw is None:
            fails += [(s, p) for p in PARAMS]
            continue
        entries = {}
        for line in raw.decode("utf-8", "ignore").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if match_entry(e):
                key = f"925{e['param']}" if PLEVEL else e["param"]
                entries[key] = (int(e["_offset"]), int(e["_length"]))
        gurl = f"{S3}/{d}/{root}/{d}120000-{s}h-oper-fc.grib2"
        for p in PARAMS:
            if p not in entries:
                fails.append((s, p))
                continue
            b = http_get(gurl, rng=entries[p])
            if b is None:
                fails.append((s, p))
            else:
                msgs[(s, p)] = b
    return root, msgs, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-20")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--smoke", action="store_true", help="3일 샘플만 (경로 3시대 검증)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--plevel", action="store_true", help="925hPa u/v 수집 (별도 파일)")
    args = ap.parse_args()
    t0 = time.time()
    global PARAMS, OUT, MISS, PLEVEL
    if args.plevel:
        PLEVEL = True
        PARAMS = ("925u", "925v")
        OUT = OUT.with_name("ecmwf_prevruns_925.parquet")
        MISS = MISS.with_name("ecmwf_missing_925.json")

    cent = _centroids()
    pts = [(float(cent.loc[g, "lat"]), float(cent.loc[g, "lon"])) for g in [1, 2, 3]]
    print(f"centroids: {pts}", flush=True)

    # 타깃 KST일 → 초기화일(D-2). 저장 키는 초기화일.
    if args.smoke:
        targets = [date(2023, 2, 3), date(2024, 6, 3), date(2025, 6, 3)]
    else:
        d0, d1 = date.fromisoformat(args.start), date.fromisoformat(args.end)
        targets = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    inits = sorted({t - timedelta(days=2) for t in targets})

    done = set()
    frames = []
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        frames.append(prev)
        cnt = prev.groupby("init_date")["step"].count()
        done = set(pd.to_datetime(cnt[cnt >= len(STEPS) * 3 * 2].index).date)  # 9스텝×3그룹×2파라미터 완결
        print(f"기존 캐시: {len(prev)}행, 완결 초기화일 {len(done)}개", flush=True)
    todo = [d for d in inits if d not in done]
    print(f"수집 대상 초기화일: {len(todo)}개 (전체 {len(inits)})", flush=True)

    missing = json.loads(MISS.read_text()) if MISS.exists() else {}
    new_rows = []
    n_ok = n_fail = 0

    def flush(force=False):
        nonlocal new_rows, frames
        if new_rows and (force or len(new_rows) >= 27 * 25):
            frames.append(pd.DataFrame(new_rows))
            allf = pd.concat(frames, ignore_index=True).drop_duplicates(["init_date", "step", "group", "param"])
            allf.to_parquet(OUT, index=False)
            frames = [allf]
            new_rows = []

    from concurrent.futures import as_completed
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_day, init_d): init_d for init_d in todo}
        for fut in as_completed(futs):
            init_d = futs[fut]
            root, msgs, fails = fut.result()
            if fails:
                missing[init_d.isoformat()] = [f"{s}-{p}" for s, p in fails]
            for (s, p) in sorted(msgs):
                try:
                    v3 = bilinear(msgs[(s, p)], pts)
                except Exception:
                    missing.setdefault(init_d.isoformat(), []).append(f"{s}-{p}:decode")
                    continue
                for gi, g in enumerate([1, 2, 3]):
                    new_rows.append(dict(init_date=pd.Timestamp(init_d), step=s, group=g,
                                         param=p, val=v3[gi]))
            n_ok += 1 if not fails else 0
            n_fail += 1 if fails else 0
            flush()
            if (n_ok + n_fail) % 50 == 0:
                el = time.time() - t0
                rate = (n_ok + n_fail) / max(el, 1) * 60
                print(f"  진행 {n_ok+n_fail}/{len(todo)} (완전 {n_ok} / 결손 {n_fail}) "
                      f"{el/60:.1f}분, {rate:.0f}일/분", flush=True)
    flush(force=True)
    MISS.write_text(json.dumps(missing, indent=1))

    allf = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(allf):
        # OUT은 long 유지 (재개 호환). wide는 피처용 별도 파일.
        wide = allf.pivot_table(index=["init_date", "step", "group"], columns="param",
                                values="val").reset_index()
        wide = wide.rename(columns={"10u": "u10", "10v": "v10", "925u": "u925", "925v": "v925"})
        wname = "ecmwf_prevruns_925_wide.parquet" if PLEVEL else "ecmwf_prevruns_wide.parquet"
        wide.to_parquet(OUT.with_name(wname), index=False)
        uc, vc = ("u925", "v925") if PLEVEL else ("u10", "v10")
        ws = np.hypot(wide[uc], wide[vc])
        print(f"\n저장: long {len(allf)}행 / wide {len(wide)}행 | ws10 범위 "
              f"{ws.min():.1f}~{ws.max():.1f} m/s (평균 {ws.mean():.2f}) | 결손일 {len(missing)}", flush=True)
    print(f"총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
