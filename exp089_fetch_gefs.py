#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════
# exp-089 데이터 수집: NOAA GEFS 앙상블 10m 바람 — D-2 12z, 평균·스프레드
#
# 배경: exp-073(ECMWF enfo)이 "D-2 리드타임에서 앙상블 평균 < 결정론 오차"로 LB를 갱신
#   (+0.00014)했으나 corr(cf_ens3, cf_hpo) 0.9996으로 대부분 중복이었다. GEFS는
#   **커버리지 2021+**(exp-048에서 다른 모델이 전부 탈락한 사유가 2024+ 커버리지였음)라
#   train 전 구간이 되는 유일한 잔여 앙상블 소스.
#
# ★ enfo 대비 결정적 비용 절감: NOAA가 **앙상블 평균(geavg)·표준편차(gespr)를 미리 계산**해
#   올려둔다 → 멤버 16개를 받을 필요가 없다 (enfo는 이 때문에 19시간이 걸렸음).
#   또 packing이 CCSDS가 아니라 표준 complex+공간차분(템플릿 3)이고, WSL에
#   requests·numpy·eccodes가 전부 있어 **fetch와 디코드를 한 프로세스에서** 처리한다
#   (exp-073의 Windows fetch → WSL decode 2단계 분리와 그로 인한 디코더 무반응 회피).
#
# 규정: exp-054/073과 동일 — 타깃일 D에 대해 **D-2 12z 런**만 사용(발표 완료가 D-1 02시 KST
#   경으로 예보 가용시점 D-1 13:00보다 앞섬). 공개·무료(NOAA Open Data, 재분석 아님).
#
# 산출: gefs_decoded/{init}.csv  —  init,step,group,u_avg,v_avg,u_spr,v_spr  (9스텝 × 3그룹)
#   시간 보간(3h → 1h)과 피처 구성은 Windows 쪽 pack 단계(exp089_gefs_features.py)에서.
#
# 실행 (WSL): wsl -e python3 exp089_fetch_gefs.py --smoke
#             wsl -e python3 exp089_fetch_gefs.py --start 2022-01-01 --end 2025-12-31
# ═══════════════════════════════════════════════════════
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import requests

BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
ROOT = "/mnt/d/projects/풍력 예측/open (1)/external"
OUT_DIR = os.path.join(ROOT, "gefs_decoded")
PTS_JSON = os.path.join(ROOT, "gefs_points.json")
STEPS = list(range(27, 52, 3))          # D-2 12z → 타깃일 01~24시 KST (exp-073과 동일)
FAMS = {"avg": "geavg", "spr": "gespr"}  # 컨트롤(gec00)은 미수집 — 결정론 GFS 0.25°가 이미 피처에 있음
WANT = (":UGRD:10 m above ground:", ":VGRD:10 m above ground:")

SESS = requests.Session()
SESS.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=64, max_retries=0))


def http_get(url, rng=None, retries=3):
    for a in range(retries):
        try:
            h = {"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}
            r = SESS.get(url, headers=h, timeout=60)
            if r.status_code in (200, 206):
                return r.content
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (a + 1))
    return None


def parse_idx(txt):
    """.idx → [(offset, 설명)] (파일 내 순서 유지). 다음 항목 offset이 현 메시지의 끝."""
    rows = []
    for line in txt.splitlines():
        p = line.split(":", 2)
        if len(p) < 3:
            continue
        try:
            rows.append((int(p[1]), ":" + p[2]))
        except ValueError:
            continue
    return rows


def bilinear(msg, pts):
    """GRIB 메시지에서 (lat, lon) 지점들의 이중선형 보간값 (exp-054 구현과 동일 규약)."""
    import eccodes as ec
    gid = ec.codes_new_from_message(msg)
    try:
        ni = ec.codes_get(gid, "Ni"); nj = ec.codes_get(gid, "Nj")
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
        out.append(float((1 - wi) * (1 - wj) * vals[j0, i0 % ni]
                         + wi * (1 - wj) * vals[j0, (i0 + 1) % ni]
                         + (1 - wi) * wj * vals[j0 + 1, i0 % ni]
                         + wi * wj * vals[j0 + 1, (i0 + 1) % ni]))
    return out


def fetch_step(d, fam_file, step, pts):
    """(u, v) 각 지점값. 실패 시 None."""
    stem = f"{BASE}/gefs.{d}/12/atmos/pgrb2ap5/{fam_file}.t12z.pgrb2a.0p50.f{step:03d}"
    raw = http_get(stem + ".idx", retries=2)
    if raw is None:
        return None
    rows = parse_idx(raw.decode("utf-8", "ignore"))
    got = {}
    for k, want in enumerate(WANT):
        hit = next((i for i, (_, desc) in enumerate(rows) if want in desc), None)
        if hit is None:
            return None
        start = rows[hit][0]
        end = rows[hit + 1][0] - 1 if hit + 1 < len(rows) else start + 4_000_000
        msg = http_get(stem, rng=(start, end))
        if msg is None:
            return None
        got[k] = bilinear(msg, pts)
    return got[0], got[1]


def do_day(target_d, pts, groups):
    """타깃일 D → init = D-2. CSV 한 개 생성. 반환 (성공여부, 실패메모)"""
    init_d = target_d - timedelta(days=2)
    d = init_d.strftime("%Y%m%d")
    path = os.path.join(OUT_DIR, f"{d}.csv")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True, "skip"
    rows, fails = [], []
    for step in STEPS:
        vals = {}
        for tag, famfile in FAMS.items():
            r = fetch_step(d, famfile, step, pts)
            if r is None:
                fails.append(f"{step}{tag}")
                break
            vals[tag] = r
        if len(vals) != len(FAMS):
            continue
        for gi, g in enumerate(groups):
            rows.append([d, step, g,
                         vals["avg"][0][gi], vals["avg"][1][gi],
                         vals["spr"][0][gi], vals["spr"][1][gi]])
    if not rows:
        return False, ",".join(fails[:3]) or "empty"
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["init", "step", "group", "u_avg", "v_avg", "u_spr", "v_spr"])
        w.writerows(rows)
    os.replace(tmp, path)                      # 원자적 — 중단돼도 반쪽 파일 없음
    return True, f"{len(rows)}행" + (f" (결손 {len(fails)})" if fails else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="20일만 — 처리율 측정")
    args = ap.parse_args()

    import json
    pts_map = json.loads(open(PTS_JSON).read())
    groups = sorted(int(k) for k in pts_map)
    pts = [tuple(pts_map[str(g)]) for g in groups]
    os.makedirs(OUT_DIR, exist_ok=True)

    s = date.fromisoformat(args.start); e = date.fromisoformat(args.end)
    days = [s + timedelta(days=i) for i in range((e - s).days + 1)]
    if args.smoke:
        days = days[:10] + days[len(days) // 2:len(days) // 2 + 10]
    print(f"### exp-089 GEFS 수집 ### 타깃일 {len(days)}일 "
          f"({args.start} ~ {args.end}) | 스텝 {len(STEPS)} × 계열 {len(FAMS)} | "
          f"워커 {args.workers}", flush=True)

    t0 = time.time()
    ok = miss = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(do_day, d, pts, groups): d for d in days}
        for n, fu in enumerate(as_completed(futs), 1):
            good, note = fu.result()
            ok += good; miss += (not good)
            if not good:
                print(f"  [실패] {futs[fu]} — {note}", flush=True)
            if n % 25 == 0 or n == len(days):
                el = (time.time() - t0) / 60
                rate = n / el if el else 0
                print(f"  진행 {n}/{len(days)} ({el:.1f}분, {rate*60:.0f}일/시간) "
                      f"| 성공 {ok} 실패 {miss} | 남은 예상 "
                      f"{(len(days)-n)/rate/60 if rate else 0:.1f}시간", flush=True)
    print(f"\n완료: 성공 {ok} / 실패 {miss} — {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
