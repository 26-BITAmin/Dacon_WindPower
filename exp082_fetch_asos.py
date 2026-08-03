# ═══════════════════════════════════════════════════════
# exp-082 데이터 수집: 기상청 API허브 지상관측(ASOS) 시간자료
#
# 규정 적합성 (대회 규칙 Data Leakage 조항):
#   "기상 관측자료 등 실측 데이터는 예측기준시점 전에 이미 관측·공개·확정된 과거 자료에
#    한해 사용할 수 있습니다." → 피처는 타깃일 D에 대해 관측시각 ≤ D-1 12:00만 사용
#    (예보 data_available_kst_dtm 13:00보다 1시간 앞). 재분석·사후보정 아님(원시 지상관측).
#   공개성: apihub.kma.go.kr 누구나 가입·신청 가능 / 라이선스: 공공누리(출처표시).
#
# 엔드포인트 제약: kma_sfctm2.php는 tm 단일시각만 지원(tm1/tm2 무시) → 1시간 = 1콜,
#   대신 stn=0으로 전 지점(~95개) 일괄 수신. kma_sfctm3.php(31일 기간조회)는 미신청 상태.
#
# 인증키: open (1)/external/kma_apihub_key.txt 에서 읽음 (코드 하드코딩 금지 — 팀 repo 푸시 대비).
#   환경변수 KMA_APIHUB_KEY 가 있으면 그쪽 우선.
#
# 출력: open (1)/external/kma_asos/{YYYYMM}.csv  (재개 가능 — 완성된 월은 건너뜀)
# 실행: C:\miniconda3\python.exe -X utf8 exp082_fetch_asos.py [--workers 8] [--start 2021-12-30] [--end 2025-12-31]
# ═══════════════════════════════════════════════════════
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd
import truststore
truststore.inject_into_ssl()   # 이 머신은 로컬 TLS 가로채기 → Windows 인증서 저장소 필요 (exp-054 선례)
import requests

from exp005_qopt import DATA_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php"
OUT_DIR = DATA_DIR / "external" / "kma_asos"
KEY_FILE = DATA_DIR / "external" / "kma_apihub_key.txt"
META = DATA_DIR / "external" / "kma_asos_meta.json"
# 헤더: YYMMDDHHMI STN WD WS GST_WD GST_WS GST_TM PA PS PT PR TA TD HM PV RN ...
KEEP = {0: "tm", 1: "stn", 2: "wd", 3: "ws", 5: "gst_ws", 7: "pa", 11: "ta", 13: "hm"}
MISSING = {"-9", "-9.0", "-99", "-99.9", "-999"}


def get_key():
    k = os.environ.get("KMA_APIHUB_KEY")
    if k:
        return k.strip()
    assert KEY_FILE.exists(), f"인증키 파일 없음: {KEY_FILE}"
    return KEY_FILE.read_text(encoding="utf-8").strip()


SESS = requests.Session()

# ⚠ 속도 제한 (2026-08-01 실측 교훈): workers 8 · 무제한으로 ~2만 콜 발사 후
#   apihub.kma.go.kr가 IP 단위로 연결 차단됨(http 000, 타 호스트는 정상).
#   토큰버킷으로 초당 요청수를 강제한다. 기본 3 req/s.
_RATE_LOCK = __import__("threading").Lock()
_LAST = [0.0]
_MIN_INTERVAL = [1.0 / 3.0]


def _throttle():
    with _RATE_LOCK:
        now = time.monotonic()
        wait = _LAST[0] + _MIN_INTERVAL[0] - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _LAST[0] = now


def http_get(url, retries=3):
    for i in range(retries):
        _throttle()
        try:
            r = SESS.get(url, timeout=40)
            if r.status_code == 200:
                return r.content
        except requests.RequestException:
            pass
        if i < retries - 1:
            time.sleep(2.0 * (i + 1))
    return None


def fetch_hour(dt, key):
    tm = dt.strftime("%Y%m%d%H%M")
    raw = http_get(f"{BASE}?tm={tm}&stn=0&authKey={key}")
    if raw is None:
        return tm, None
    txt = raw.decode("euc-kr", "replace")
    if '"status"' in txt[:200]:          # JSON 에러 응답
        return tm, None
    rows = []
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 14:
            continue
        rec = {}
        for i, name in KEEP.items():
            v = p[i]
            rec[name] = None if v in MISSING else v
        rows.append(rec)
    return tm, rows


BASE3 = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"
# 발전단지 인근·고상관 상위 지점 (exp082_inventory.py 실측: 일평균 ws ↔ 일평균 CF)
CORE_STNS = [100, 105, 104, 216, 217, 272, 137, 276]


def parse_rows(txt):
    rows = []
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 14:
            continue
        rows.append({name: (None if p[i] in MISSING else p[i]) for i, name in KEEP.items()})
    return rows


def fetch_period(stn, d0, d1, key):
    """kma_sfctm3.php — 한 지점의 기간(≤31일) 일괄 수신. 승인 시 콜 수가 ~60배 감소."""
    url = (f"{BASE3}?tm1={d0.strftime('%Y%m%d%H%M')}&tm2={d1.strftime('%Y%m%d%H%M')}"
           f"&stn={stn}&authKey={key}")
    raw = http_get(url)
    if raw is None:
        return None
    txt = raw.decode("euc-kr", "replace")
    if '"status"' in txt[:200]:
        return None
    return parse_rows(txt)


def stage_period(args, key):
    """기간조회 모드 — 월 × 지점 단위. sfctm3 활용신청 승인 후 사용."""
    stns = [int(s) for s in args.stns.split(",")] if args.stns else CORE_STNS
    d0 = datetime.fromisoformat(args.start)
    d1 = datetime.fromisoformat(args.end) + timedelta(hours=23)
    months = month_range(d0, d1)
    # 시간별 모드로 이미 확보한 월은 건너뜀 (불필요한 재수신 방지)
    todo = [m for m in sorted(months)
            if not (OUT_DIR / f"{m}.period.csv").exists() and not (OUT_DIR / f"{m}.csv").exists()]
    print(f"[기간조회] {len(todo)}개월 × 지점인수 {len(stns)}개 = {len(todo)*len(stns)}콜 "
          f"(시간별 모드였다면 {sum(len(months[m]) for m in todo)}콜)", flush=True)
    t0 = time.time()
    for mi, m in enumerate(todo, 1):
        hours = months[m]
        h0, h1 = hours[0], hours[-1]
        rows, fails = [], []
        for stn in stns:
            r = fetch_period(stn, h0, h1, key)
            if r is None:
                fails.append(stn)
            else:
                rows.extend(r)
        if fails or not rows:
            print(f"  [{m}] 실패 지점인수 {fails} — 저장 보류", flush=True)
            if len(fails) == len(stns):
                print("  전량 실패 — 중단 (권한/차단 확인 필요)", flush=True)
                return False
            continue
        df = pd.DataFrame(rows)
        # 응답 절단 검증: 받은 고유 시각이 그 달 시각 수와 일치해야 함
        got_h = df["tm"].nunique()
        if got_h < len(hours):
            print(f"  [{m}] 시각 부족 {got_h}/{len(hours)} — 응답 절단 의심, 저장 보류", flush=True)
            continue
        tmp = OUT_DIR / f"{m}.period.csv.tmp"
        df.to_csv(tmp, index=False, encoding="utf-8")
        tmp.replace(OUT_DIR / f"{m}.period.csv")
        print(f"  [{m}] {len(df)}행 · 지점 {df['stn'].nunique()} · 시각 {got_h}/{len(hours)} "
              f"({mi}/{len(todo)}, {(time.time()-t0)/60:.1f}분)", flush=True)
    print(f"[기간조회] 완료 — {(time.time()-t0)/60:.1f}분", flush=True)
    return True


def month_range(d0, d1):
    """월별 (라벨, 시각리스트)"""
    out = {}
    d = d0
    while d <= d1:
        out.setdefault(d.strftime("%Y%m"), []).append(d)
        d += timedelta(hours=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rate", type=float, default=3.0, help="초당 최대 요청 수 (차단 방지)")
    ap.add_argument("--mode", choices=["hour", "period"], default="hour",
                    help="hour=sfctm2(1시각 1콜·전지점) / period=sfctm3(31일 1콜·지점별, 활용신청 필요)")
    ap.add_argument("--stns", default="", help="period 모드 지점 목록 (쉼표구분, 기본 CORE_STNS)")
    ap.add_argument("--start", default="2021-12-30")
    ap.add_argument("--end", default="2025-12-31")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _MIN_INTERVAL[0] = 1.0 / max(args.rate, 0.1)
    key = get_key()
    if args.mode == "period":
        stage_period(args, key)
        return
    d0 = datetime.fromisoformat(args.start)
    d1 = datetime.fromisoformat(args.end) + timedelta(hours=23)
    months = month_range(d0, d1)
    todo_months = [m for m in sorted(months) if not (OUT_DIR / f"{m}.csv").exists()]
    total_h = sum(len(months[m]) for m in todo_months)
    print(f"수집 대상 {len(todo_months)}/{len(months)}개월 ({total_h}시각) — workers {args.workers}",
          flush=True)
    t0 = time.time()
    done_h = 0
    for mi, m in enumerate(todo_months, 1):
        hours = months[m]
        rows, fails = [], []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_hour, h, key): h for h in hours}
            for fut in as_completed(futs):
                tm, r = fut.result()
                if r is None:
                    fails.append(tm)
                else:
                    rows.extend(r)
        if fails:
            # 결손 월을 완료로 표시하면 재개 시 영구 누락 → 저장하지 않고 다음 런에서 재시도
            print(f"  [{m}] 실패 {len(fails)}/{len(hours)} — 저장 보류(다음 런 재시도)", flush=True)
            if len(fails) > len(hours) * 0.5:
                print("  연결 차단 정황 — 수집 중단", flush=True)
                break
            continue
        df = pd.DataFrame(rows)
        tmp = OUT_DIR / f"{m}.csv.tmp"
        df.to_csv(tmp, index=False, encoding="utf-8")
        tmp.replace(OUT_DIR / f"{m}.csv")
        done_h += len(hours)
        el = time.time() - t0
        eta = (total_h - done_h) / max(done_h / el, 1e-9) / 60
        print(f"  [{m}] {len(df)}행 · 지점 {df['stn'].nunique()} · 실패 {len(fails)} "
              f"({mi}/{len(todo_months)}, {el/60:.1f}분 경과, ETA {eta:.0f}분)", flush=True)
    META.write_text(json.dumps({
        "source": "기상청 API허브 지상관측(ASOS) 시간자료 (kma_sfctm2.php)",
        "url": "https://apihub.kma.go.kr/apiList.do?seqApi=2",
        "license": "공공누리 출처표시 (기상청)",
        "fetched_at": datetime.now().isoformat(),
        "range": [args.start, args.end],
        "columns": list(KEEP.values()),
        "lead_compliance_note": ("피처는 타깃일 D에 대해 관측시각 ≤ D-1 12:00만 사용 — "
                                 "대회 규칙 '예측기준시점 전에 관측·공개·확정된 과거 자료' 조항 준수. "
                                 "재분석·사후보정 자료 아님(원시 지상관측)."),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n수집 완료 — {(time.time()-t0)/60:.1f}분, 메타 → {META.name}", flush=True)


if __name__ == "__main__":
    main()
