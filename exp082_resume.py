# ═══════════════════════════════════════════════════════
# exp-082 재개 감시기 — apihub.kma.go.kr IP 차단 해제 대기 후 저속 수집 재개
#
# 배경 (2026-08-01): workers 8·무제한으로 ~2만 콜 발사 → apihub가 IP 단위 연결 차단
#   (http 000, 타 호스트 정상 = API 레벨 429가 아니라 WAF 차단). 27/48개월 확보 후 중단.
#   → 10분 주기로 연결 프로브, 살아나면 --rate 3 --workers 2로 재개 (남은 21개월 ≈ 85분).
#
# 종료: 전 월 확보 시 0 / 12시간 초과 시 3
# 로그: exp082_resume_log.txt
# 실행: C:\miniconda3\python.exe -X utf8 exp082_resume.py
# ═══════════════════════════════════════════════════════
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import truststore
truststore.inject_into_ssl()
import requests

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "open (1)" / "external" / "kma_asos"
KEY_FILE = ROOT / "open (1)" / "external" / "kma_apihub_key.txt"
LOG = ROOT / "exp082_resume_log.txt"
FETCH_LOG = ROOT / "exp082_fetch_log.txt"
PY = r"C:\miniconda3\python.exe"
PROBE_EVERY = 600          # 10분
MAX_HOURS = 12


def log(msg):
    line = f"[{datetime.now():%m-%d %H:%M}] {msg}"
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def needed_months():
    d0, d1 = datetime(2021, 12, 1), datetime(2025, 12, 1)
    out, d = [], d0
    while d <= d1:
        out.append(d.strftime("%Y%m"))
        d = (d.replace(day=28) + timedelta(days=8)).replace(day=1)
    return [m for m in out if not (OUT_DIR / f"{m}.csv").exists()]


def alive(key):
    """단발 프로브 — 살아있으면 True"""
    try:
        r = requests.get("https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php"
                         f"?tm=202401010100&stn=216&authKey={key}", timeout=25)
        return r.status_code == 200 and '"status"' not in r.text[:200]
    except requests.RequestException:
        return False


def main():
    key = KEY_FILE.read_text(encoding="utf-8").strip()
    start = time.time()
    log(f"재개 감시 시작 — 남은 {len(needed_months())}개월, {PROBE_EVERY//60}분 주기 프로브")
    while True:
        miss = needed_months()
        if not miss:
            log("전 월 확보 완료")
            sys.exit(0)
        if time.time() - start > MAX_HOURS * 3600:
            log(f"{MAX_HOURS}시간 초과 — 종료 (남은 {len(miss)}개월)")
            sys.exit(3)
        if alive(key):
            log(f"연결 복구 확인 — 저속 수집 재개 (남은 {len(miss)}개월, rate 3/s)")
            with FETCH_LOG.open("ab") as f:
                subprocess.run([PY, "-X", "utf8", "exp082_fetch_asos.py",
                                "--start", "2021-12-30", "--end", "2025-12-31",
                                "--workers", "2", "--rate", "3"],
                               cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
            left = needed_months()
            log(f"수집 런 종료 — 남은 {len(left)}개월")
            if left:
                log("잔여분 존재(재차단 정황) — 30분 대기 후 재시도")
                time.sleep(1800)
        else:
            log(f"아직 차단 중 (남은 {len(miss)}개월) — {PROBE_EVERY//60}분 후 재프로브")
            time.sleep(PROBE_EVERY)


if __name__ == "__main__":
    main()
