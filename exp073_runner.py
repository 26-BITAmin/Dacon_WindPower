# ═══════════════════════════════════════════════════════
# exp-073 자동 러너 — train 구간(init 2023-01-18~2024-12-29) enfo 커버리지 완료 감시
#   → 완료 시 pack + 게이트 자동 실행 (스케줄 세션에서 백그라운드 가동)
#
# 종료 코드: 0 = 게이트 완료 / 3 = 13시간 타임아웃 (게이트 미실행) / 그 외 = 게이트 오류
# 로그: exp073_runner_log.txt (감시), exp073_gate_log.txt (pack + 게이트 출력)
#
# 실행: C:\miniconda3\python.exe -X utf8 exp073_runner.py
# ═══════════════════════════════════════════════════════
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent
EXT = ROOT / "open (1)" / "external"
DEC = EXT / "enfo_decoded"
RAW = EXT / "enfo_raw"
PY = r"C:\miniconda3\python.exe"
RUNNER_LOG = ROOT / "exp073_runner_log.txt"
GATE_LOG = ROOT / "exp073_gate_log.txt"
ENS_CACHE = EXT / "exp073_gbm_ens.parquet"


def log(msg):
    line = f"[{datetime.now():%m-%d %H:%M}] {msg}"
    with RUNNER_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def train_days():
    d0, d1 = date(2023, 1, 18), date(2024, 12, 29)
    return [(d0 + timedelta(days=i)).strftime("%Y%m%d")
            for i in range((d1 - d0).days + 1)]


def status():
    dec = {p.stem for p in DEC.glob("*.csv")}
    raw = {p.stem for p in RAW.glob("*.json")}
    miss = [x for x in train_days() if x not in dec]
    pending = [x for x in miss if x in raw and (RAW / f"{x}.bin").exists()]
    return miss, pending, len(dec)


def drain_decoder():
    log("디코더 무반응 — WSL 원샷 드레인 실행")
    with GATE_LOG.open("ab") as f:
        subprocess.run(["wsl", "-e", "python3", "exp073_decode_wsl.py", "--delete-raw"],
                       cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, timeout=3600)


def main():
    start = time.time()
    last_change = time.time()
    prev_n = -1
    log(f"러너 시작 — train 구간 결손 감시 (5분 주기)")
    while True:
        miss, pending, n_csv = status()
        if n_csv != prev_n:
            prev_n = n_csv
            last_change = time.time()
        idle_min = (time.time() - last_change) / 60
        log(f"결손 {len(miss)} | raw대기 {len(pending)} | 디코드누계 {n_csv} | idle {idle_min:.0f}분")
        if not miss:
            log("train 구간 커버리지 완료")
            break
        if pending and idle_min > 20:
            try:
                drain_decoder()
            except Exception as e:
                log(f"드레인 실패: {e}")
            last_change = time.time()
            continue
        if len(miss) <= 15 and idle_min > 45:
            log(f"진행 정지 + 결손 {len(miss)}일(≤15) — 결손 허용하고 진행 (GBM은 NaN 처리)")
            break
        if time.time() - start > 13 * 3600:
            log("13시간 타임아웃 — 게이트 미실행 종료")
            sys.exit(3)
        time.sleep(300)

    if ENS_CACHE.exists():
        ENS_CACHE.unlink()
        log("이전 ens 캐시 삭제 (부분 데이터 오염 방지)")
    log("pack + 게이트 시작 → exp073_gate_log.txt")
    with GATE_LOG.open("ab") as f:
        subprocess.run([PY, "-X", "utf8", "exp073_fetch_enfo.py", "--stage", "pack"],
                       cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
        rc = subprocess.run([PY, "-X", "utf8", "exp073_gate.py"],
                            cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"게이트 종료 (rc={rc})")
    sys.exit(rc)


if __name__ == "__main__":
    main()
