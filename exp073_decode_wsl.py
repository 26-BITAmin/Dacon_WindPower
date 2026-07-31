# ═══════════════════════════════════════════════════════
# exp-073 디코드 워커 — WSL(python3 + apt eccodes 2.45)에서 실행
#   Windows의 eccodes DLL 차단(애플리케이션 제어 정책) 우회: 수집(Windows)과
#   디코드(WSL)를 분리. 의존성 = eccodes + numpy만 (pandas 불필요, CSV 출력).
#
# 입력:  open (1)/external/enfo_raw/{일자}.bin + .json (+ centroids.json)
# 출력:  open (1)/external/enfo_decoded/{일자}.csv  (init_date,step,group,param,member,val)
# 옵션:  --delete-raw  (디코드 성공한 날의 .bin 삭제 — 전 기간 수집 시 디스크 절약)
#        --watch       (수집과 병렬 실행: 새 파일을 폴링, fetch 완료 마커까지 대기)
#
# 실행 (프로젝트 루트에서):  wsl -e python3 exp073_decode_wsl.py [--delete-raw] [--watch]
# ═══════════════════════════════════════════════════════
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import eccodes as ec

BASE = Path(__file__).parent / "open (1)" / "external"
RAW = BASE / "enfo_raw"
DEC = BASE / "enfo_decoded"


def bilinear(msg_bytes, pts):
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


def decode_day(day, pts, delete_raw):
    if (DEC / f"{day}.csv").exists():      # 경합 내성: 다른 인스턴스가 먼저 처리
        return 0
    try:
        man = json.loads((RAW / f"{day}.json").read_text())
        blob = (RAW / f"{day}.bin").read_bytes()
    except FileNotFoundError:
        return 0
    rows = []
    for it in man["items"]:
        b = blob[it["offset"]:it["offset"] + it["length"]]
        try:
            v3 = bilinear(b, pts)
        except Exception as e:
            print(f"  {day} {it['step']}-{it['param']}-m{it['member']} 디코드 실패: {e}",
                  flush=True)
            continue
        for gi, g in enumerate([1, 2, 3]):
            rows.append(f"{day},{it['step']},{g},{it['param']},{it['member']},{v3[gi]:.4f}")
    if not rows:
        return 0
    tmp = DEC / f"{day}.csv.tmp"
    tmp.write_text("init_date,step,group,param,member,val\n" + "\n".join(rows) + "\n")
    tmp.rename(DEC / f"{day}.csv")
    if delete_raw:
        (RAW / f"{day}.bin").unlink(missing_ok=True)
    return len(rows)


def pending_days():
    done = {p.stem for p in DEC.glob("*.csv")}
    return sorted(p.stem for p in RAW.glob("*.json")
                  if p.stem != "centroids" and p.stem not in done
                  and (RAW / f"{p.stem}.bin").exists())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete-raw", action="store_true")
    ap.add_argument("--watch", action="store_true")
    args = ap.parse_args()
    DEC.mkdir(parents=True, exist_ok=True)
    cent = json.loads((RAW / "centroids.json").read_text())
    pts = [tuple(cent[str(g)]) for g in [1, 2, 3]]
    t0 = time.time()
    n_days = 0
    idle = 0
    while True:
        days = pending_days()
        if not days:
            if not args.watch:
                break
            if (RAW / "FETCH_DONE").exists():
                days = pending_days()
                if not days:
                    break
            idle += 1
            if idle > 240:      # 2시간 무입력이면 종료
                print("watch 타임아웃 — 종료", flush=True)
                break
            time.sleep(30)
            continue
        idle = 0
        for day in days:
            n = decode_day(day, pts, args.delete_raw)
            n_days += 1
            if n_days % 20 == 0:
                print(f"  디코드 {n_days}일째 ({(time.time()-t0)/60:.1f}분)", flush=True)
    print(f"디코드 완료: {n_days}일, {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
