"""Build the Colab notebook for the external-data no-snap champion track.

The model implementation is derived from Sinyoung's self-contained notebook at
commit c8855bd.  This generator applies the team-agreed paths, reproducibility
guards, missing-label handling, external-data manifests, and final output names.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "external_champion_train.ipynb"
SOURCE_REV = "c8855bd"
SOURCE_PATH = "simplified.ipynb"


def lines(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def markdown(text: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": lines(text),
    }


def replace_required(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"source patch not found: {label}")
    return text.replace(old, new)


raw = subprocess.check_output(
    ["git", "show", f"{SOURCE_REV}:{SOURCE_PATH}"], cwd=ROOT
)
source_nb = json.loads(raw)
source_cells = source_nb["cells"]
if len(source_cells) != 16:
    raise RuntimeError(f"unexpected source cell count: {len(source_cells)}")


setup = r'''
#@title 1. Colab, Drive, 실행 설정
import os, sys, json, hashlib, random, re, time, pickle, platform, subprocess
from datetime import datetime, timezone
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "lightgbm", "xgboost",
         "torch", "tqdm", "pyarrow", "openpyxl", "eccodes"],
        check=True,
    )

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgbm
from tqdm.auto import tqdm

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DATA_DIR = Path("/content/drive/MyDrive/BARAM2026/data")
    RUN_ROOT = Path("/content/drive/MyDrive/BARAM2026/hyejin_external_champion")
else:
    DATA_DIR = Path.cwd() / "data"
    RUN_ROOT = Path.cwd() / "runs" / "hyejin_external_champion"

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
EXTERNAL_DIR = DATA_DIR / "external"
CACHE_DIR = RUN_ROOT / "cache"
for path in [RUN_ROOT, EXTERNAL_DIR, CACHE_DIR]:
    path.mkdir(exist_ok=True, parents=True)

# Open-Meteo 무료 API의 비상업 이용 조건을 팀에서 확인한 뒤 True로 바꿉니다.
ACK_OPEN_METEO_TERMS = False  #@param {type:"boolean"}
FORCE_JMA_FETCH = False
FORCE_ECMWF_FETCH = False
FINAL_GBM_SEEDS = 3
FINAL_DL_SEEDS = 3

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print("device:", DEVICE, "/", GPU_NAME)
if IN_COLAB and "A100" not in GPU_NAME.upper():
    print("[주의] A100 런타임인지 확인하세요.")

TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPACITY_KWH = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}
GROUP_SCALES = {1: 0.98, 2: 0.98, 3: 0.97}

REQUIRED = [
    "train/train_labels.csv", "train/ldaps_train.csv", "train/gfs_train.csv",
    "train/scada_vestas_train.csv", "train/scada_unison_train.csv",
    "test/ldaps_test.csv", "test/gfs_test.csv", "info.xlsx", "sample_submission.csv",
]
missing = [rel for rel in REQUIRED if not (DATA_DIR / rel).exists()]
assert not missing, "누락된 공식 파일:\n" + "\n".join(missing)

def sha256_file(path, block=4 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(block)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

print("data    :", DATA_DIR)
print("external:", EXTERNAL_DIR)
print("output  :", RUN_ROOT)
'''


cells = [
    markdown(
        r'''
## BARAM 2026 - 외부 데이터 no-snap 챔피언 학습

이 노트북은 공식 LDAPS/GFS와 JMA GSM/MSM, ECMWF IFS HRES를 사용해 검증하고 최종 submission을 생성합니다.

- JMA: Open-Meteo Previous Runs의 `previous_day2`만 사용
- ECMWF: D-2 12z IFS HRES 10m 바람 사용
- GBM: LightGBM과 XGBoost 50:50
- DL: q60 MTMLP 384×3, 계절 블록 early stopping
- 최종 혼합: GBM:DL = 0.65:0.35
- 후처리: bias, rolling 0.40, clip 0.16, snap 없음

외부 데이터는 처음 한 번만 수집하며 Google Drive에 저장된 캐시를 다음 실행부터 재사용합니다. `F1_ADD_VEER`는 gate에서 기각되어 포함하지 않습니다.
''',
        "external_intro",
    ),
    markdown(
        r'''
### 실행 전 확인

1. Colab A100 런타임을 선택합니다.
2. 공식 데이터가 `/content/drive/MyDrive/BARAM2026/data` 아래에 있어야 합니다.
3. Open-Meteo 이용 조건을 팀에서 확인한 뒤 `ACK_OPEN_METEO_TERMS=True`로 바꿉니다.
4. 전체 셀을 순서대로 실행합니다.

JMA와 ECMWF 파일을 수동으로 내려받을 필요는 없습니다. 수집이 중단되면 ECMWF는 Drive 체크포인트에서 이어서 진행합니다.
''',
        "external_run_guide",
    ),
    {
        "cell_type": "code",
        "id": "external_setup",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": lines(setup),
    },
]


for source_index in range(1, 13):
    cell = source_cells[source_index]
    if cell["cell_type"] != "code":
        raise RuntimeError(f"source cell {source_index} is not code")
    text = "".join(cell.get("source", []))

    if source_index == 5:
        text = replace_required(
            text,
            'JMA_PATH = DATA_DIR / "external"',
            "JMA_PATH = EXTERNAL_DIR",
            "JMA path",
        )
        text = replace_required(
            text,
            'LEADS = ["previous_day2", "previous_day1"]',
            'LEADS = ["previous_day2"]',
            "JMA lead",
        )
        text = replace_required(
            text,
            '''def fetch_jma_prevruns(info, force=False):
    out_path = JMA_PATH / "jma_prevruns.csv"
    if not force and out_path.exists():
        return pd.read_csv(out_path, encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"])
''',
            '''def fetch_jma_prevruns(info, force=False):
    force = force or FORCE_JMA_FETCH
    out_path = JMA_PATH / "jma_prevruns.csv"
    if not ACK_OPEN_METEO_TERMS:
        raise RuntimeError(
            "Open-Meteo 이용 조건을 팀에서 확인한 뒤 1번 셀의 "
            "ACK_OPEN_METEO_TERMS=True로 바꾸세요."
        )
    if not force and out_path.exists():
        print(f"JMA 캐시 로드: {out_path}")
        return pd.read_csv(out_path, encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"])
''',
            "JMA terms guard",
        )
        text = replace_required(
            text,
            '''    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path} ({len(out)}행)")
''',
            '''    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    jma_meta = {
        "source": "Open-Meteo Previous Runs API",
        "source_url": "https://open-meteo.com/en/docs/previous-runs-api",
        "terms_url": "https://open-meteo.com/en/terms",
        "lead": "previous_day2",
        "models": MODELS,
        "variables": HOURLY_VARS,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (JMA_PATH / "jma_fetch_meta.json").write_text(
        json.dumps(jma_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved: {out_path} ({len(out)}행)")
''',
            "JMA metadata",
        )

    if source_index == 6:
        text = replace_required(
            text,
            "예측기준시점(전날 13시 KST)",
            "예측기준시점(전날 14시 KST)",
            "cutoff comment",
        )
        text = replace_required(
            text,
            'EC_PATH = DATA_DIR / "external"',
            "EC_PATH = EXTERNAL_DIR",
            "ECMWF path",
        )
        text = replace_required(
            text,
            '''def fetch_ecmwf_prevruns(info, start=EC_START, end=EC_END, workers=20, force=False):
    # 체크포인트''',
            '''def fetch_ecmwf_prevruns(info, start=EC_START, end=EC_END, workers=20, force=False):
    force = force or FORCE_ECMWF_FETCH
    # 체크포인트''',
            "ECMWF force flag",
        )
        text = replace_required(
            text,
            '''    out = pd.DataFrame(all_rows)
    out.to_parquet(EC_RAW_PARQUET, index=False)
    print(f"ECMWF raw 수집 완료: {len(out)}행, {out['init_date'].nunique()}/{len(inits)}일 ({(time.time()-t0)/60:.1f}분)", flush=True)
''',
            '''    out = pd.DataFrame(all_rows)
    out.to_parquet(EC_RAW_PARQUET, index=False)
    ec_meta = {
        "source": "ECMWF Open Data AWS archive",
        "source_url": "https://www.ecmwf.int/en/forecasts/datasets/open-data",
        "run": "D-2 12z IFS HRES",
        "variables": ["10u", "10v"],
        "date_range": [str(start), str(end)],
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (EC_PATH / "ecmwf_fetch_meta.json").write_text(
        json.dumps(ec_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"ECMWF raw 수집 완료: {len(out)}행, {out['init_date'].nunique()}/{len(inits)}일 ({(time.time()-t0)/60:.1f}분)", flush=True)
''',
            "ECMWF metadata",
        )

    if source_index == 10:
        text = replace_required(
            text,
            '''        va_all = train_df.loc[(train_df["forecast_kst_dtm"] >= fold["valid_start"]) & (train_df["forecast_kst_dtm"] <= fold["valid_end"])]
''',
            '''        va_all = train_df.loc[
            (train_df["forecast_kst_dtm"] >= fold["valid_start"])
            & (train_df["forecast_kst_dtm"] <= fold["valid_end"])
            & train_df["target"].notna()
        ]
''',
            "GBM missing validation targets",
        )
        text = replace_required(
            text,
            '''        va_all = train_df.loc[(train_df["forecast_kst_dtm"] >= fold["valid_start"]) & (train_df["forecast_kst_dtm"] <= fold["valid_end"])].sort_values("forecast_kst_dtm")
''',
            '''        va_all = train_df.loc[
            (train_df["forecast_kst_dtm"] >= fold["valid_start"])
            & (train_df["forecast_kst_dtm"] <= fold["valid_end"])
            & train_df["target"].notna()
        ].sort_values("forecast_kst_dtm")
''',
            "DL missing validation targets",
        )

    if source_index == 11:
        text = replace_required(
            text,
            '''# ── 데이터 준비 ──
train_df_tps, test_df_tps, cols_tps, sample_submission = build_v11(build_datasets()[4])
# 주의: info를 따로 받으려면 build_datasets()를 한 번 더 호출하는 게 안전합니다.
_, _, _, sample_submission, info = build_datasets()
train_df_tps, test_df_tps, cols_tps, _ = build_v11(info)
''',
            '''# 데이터 준비
_, _, _, sample_submission, info = build_datasets()
train_df_tps, test_df_tps, cols_tps, _ = build_v11(info)
''',
            "duplicate dataset build",
        )
        text = replace_required(
            text,
            '''cols_gbm = cols_gspd + EC_COLS

# ── null importance 스크리닝''',
            '''cols_gbm = cols_gspd + EC_COLS

external_files = []
for external_path in [JMA_PATH / "jma_prevruns.csv", EC_RAW_PARQUET]:
    if external_path.exists():
        external_files.append({
            "path": str(external_path),
            "bytes": external_path.stat().st_size,
            "sha256": sha256_file(external_path),
        })
external_manifest = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "prediction_cutoff": "target day D uses information public by D-1 14:00 KST",
    "jma": {
        "source": "Open-Meteo Previous Runs API",
        "lead": "previous_day2",
        "license_and_terms": "CC-BY 4.0 data; API usage subject to Open-Meteo terms",
    },
    "ecmwf": {
        "source": "ECMWF Open Data AWS archive",
        "run": "D-2 12z IFS HRES",
        "variables": ["10u", "10v"],
        "license": "CC-BY-4.0 with attribution",
    },
    "files": external_files,
}
(RUN_ROOT / "external_manifest.json").write_text(
    json.dumps(external_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)

# null importance 스크리닝''',
            "external manifest",
        )
        text = text.replace(
            'GBM_CACHE_PATH = CACHE_DIR / "gbm_gspd_cache.parquet"',
            'GBM_CACHE_PATH = CACHE_DIR / "external_champion_gbm_v1.parquet"',
        )
        text = text.replace(
            'BIAS_CACHE_PATH = CACHE_DIR / "gbm_gspd_bias.pkl"',
            'BIAS_CACHE_PATH = CACHE_DIR / "external_champion_bias_v1.pkl"',
        )
        text = text.replace(
            'DL_CACHE_PATH = CACHE_DIR / "dl_q60_cache.parquet"',
            'DL_CACHE_PATH = CACHE_DIR / "external_champion_dl_v1.parquet"',
        )
        text = text.replace(
            'ckpt_path=CACHE_DIR / "gbm_gspd_ckpt.pkl"',
            'ckpt_path=CACHE_DIR / "external_champion_gbm_ckpt_v1.pkl"',
        )
        text = text.replace(
            'ckpt_path=CACHE_DIR / "dl_q60_ckpt.parquet"',
            'ckpt_path=CACHE_DIR / "external_champion_dl_ckpt_v1.parquet"',
        )
        text = replace_required(
            text,
            '''cv_score, hold_score = score_simplified(cache, bias_maps, w=0.65)
print(f"CV Score={cv_score[0]:.4f} | HOLD Score={hold_score[0]:.4f}")''',
            '''cv_score, hold_score = score_simplified(cache, bias_maps, w=0.65)
print(f"CV Score={cv_score[0]:.4f} | HOLD Score={hold_score[0]:.4f}")
validation_result = {
    "cv": {"Score": float(cv_score[0]), "1-NMAE": float(cv_score[1]), "FiCR": float(cv_score[2])},
    "hold": {"Score": float(hold_score[0]), "1-NMAE": float(hold_score[1]), "FiCR": float(hold_score[2])},
    "feature_counts": {"gbm": len(cols_gbm), "dl": len(cols_gspd)},
}
(RUN_ROOT / "validation_result.json").write_text(
    json.dumps(validation_result, ensure_ascii=False, indent=2), encoding="utf-8"
)
''',
            "validation result",
        )

    if source_index == 12:
        text = text.replace("range(3)", "range(FINAL_GBM_SEEDS)", 2)
        text = replace_required(
            text,
            "dl_cf_te = dl_fold_predict(tr_g, te_g, cols_gspd, seeds=3, cfg=DL_CFG)",
            "dl_cf_te = dl_fold_predict(tr_g, te_g, cols_gspd, seeds=FINAL_DL_SEEDS, cfg=DL_CFG)",
            "final DL seeds",
        )
        text = replace_required(
            text,
            '''output_path = DATA_DIR / "simplified_nosnap_ec_submit.csv"
submission_final.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"Saved: {output_path}")''',
            '''for col in TARGET_COLS:
    submission_final[col] = submission_final[col].clip(0, CAPACITY_KWH[col])

output_path = RUN_ROOT / "submission_external_nosnap_ec.csv"
submission_final.to_csv(output_path, index=False, encoding="utf-8-sig")

submission_scaled = submission_final.copy()
for group in [1, 2, 3]:
    col = f"kpx_group_{group}"
    submission_scaled[col] = (
        submission_scaled[col] * GROUP_SCALES[group]
    ).clip(0, CAPACITY_KWH[col])
scaled_output_path = RUN_ROOT / "submission_external_nosnap_ec_g3s97.csv"
submission_scaled.to_csv(scaled_output_path, index=False, encoding="utf-8-sig")
print(f"Saved: {output_path}")
print(f"Saved: {scaled_output_path}")
''',
            "submission outputs",
        )

    text = text.replace("⚠️", "").replace("⚠", "").replace("★", "").replace("←", "")
    cells.append(
        {
            "cell_type": "code",
            "id": f"external_source_{source_index:02d}",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": lines(text),
        }
    )


verification = r'''
#@title 15. submission 형식 검증
for submission_path in [output_path, scaled_output_path]:
    sub = pd.read_csv(submission_path, encoding="utf-8-sig")
    print("\n", submission_path.name, sub.shape)
    assert len(sub) == len(sample_submission)
    assert sub["forecast_id"].equals(sample_submission["forecast_id"])
    assert sub[TARGET_COLS].notna().all().all()
    for col in TARGET_COLS:
        assert sub[col].between(0, CAPACITY_KWH[col]).all()
        print(col, "min:", sub[col].min(), "max:", sub[col].max())
print("[완료] submission 두 파일 형식 검증 통과")
'''
cells.append(
    {
        "cell_type": "code",
        "id": "external_submission_check",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": lines(verification),
    }
)


cells.append(
    markdown(
        r'''
### 생성 파일

`/content/drive/MyDrive/BARAM2026/hyejin_external_champion`에 다음 파일이 저장됩니다.

- `validation_result.json`
- `external_manifest.json`
- `submission_external_nosnap_ec.csv`
- `submission_external_nosnap_ec_g3s97.csv`
- `cache/` 아래 fold별 GBM/DL 체크포인트

두 submission을 바로 모두 제출하지 말고, 검증 결과와 팀의 기존 챔피언 구성을 비교해 최종 후보를 정합니다.
''',
        "external_outputs",
    )
)


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": OUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
        "source_provenance": {
            "repository_revision": SOURCE_REV,
            "repository_path": SOURCE_PATH,
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT)
