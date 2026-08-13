#!/usr/bin/env python3
"""Build the official-data-only no-snap feature ablation notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "nosnap_dl_ablation.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


cells = [
    md(
        r"""
        ## BARAM 2026 — 공식 데이터 전용 no-snap 피처 ablation

        이 노트북은 원우의 예측 캐시나 외부 데이터를 받지 않고, 대회 공식 데이터만으로 강한 독립 control을 처음부터 학습합니다.

        그대로 고정하는 규칙:

        - 멀티태스크 MLP `384×3`, q60 Huber loss
        - 학습창 마지막 해의 3·6·9·12월 11~20일 내부 validation
        - 최대 120 epoch, patience 15, min delta `1e-5`, best weight 복원
        - GBM:DL=`0.65:0.35`
        - bias correction → rolling `alpha=.40` → clip `0.16` → 그룹 scale `(.98,.98,.97)`
        - rated snap은 사용하지 않음

        팀 코드에서 확인해 반영한 부분:

        - 원우 `wi09_earlystop.py`: 계절 블록 early stopping 규칙
        - 신영 `simplified.ipynb` 최신본: SCADA 종료시각 `ceil` 정렬과 결측/확정정지 분리

        신영의 ECMWF 외부예보와 null-importance 제거 목록은 이 노트북의 공식 데이터 전용 조건 및 피처 공간과 다르므로 직접 가져오지 않습니다.

        비교하는 DL 입력:

        - `C0_OFFICIAL_BASE`: 공식 LDAPS/GFS level·물리·시간·공간편차 입력
        - `F0_REMOVE_REDUNDANT`: C0의 상수·완전중복 입력 제거
        - `F1_ADD_VEER`: F0 + 10m↔100m 풍향 변화
        - `F2_ADD_GROUP_DIRECTION`: F0 + 발전단지×풍향 circular interaction
        - `F3_ADD_ALONG_GRADIENT`: F0 + LDAPS 풍속 공간경사의 바람축 투영

        이 점수는 외부 JMA·ECMWF·ENFO를 포함한 원우 챔피언 점수의 재현값이 아닙니다. 같은 split에서 `후보−control`의 상대 변화로 피처를 선별하는 독립 실험입니다.
        """
    ),
    md(
        r"""
        ### 0. 필요한 폴더

        Colab에서는 공식 데이터를 Drive의 `BARAM2026/data`에 아래 구조로 둡니다. 외부 데이터와 원우 캐시는 필요하지 않습니다.

        ```text
        data/
        ├── train/
        │   ├── train_labels.csv
        │   ├── ldaps_train.csv
        │   ├── gfs_train.csv
        │   ├── scada_vestas_train.csv
        │   └── scada_unison_train.csv
        ├── test/
        │   ├── ldaps_test.csv
        │   └── gfs_test.csv
        ├── info.xlsx
        └── sample_submission.csv
        ```

        처음 실행하면 공식 원본으로 피처와 GBM 예측을 생성해 혜진의 Drive에 저장합니다. 이후 같은 설정은 혜진이 만든 캐시를 재사용합니다.
        """
    ),
    code(
        r"""
        #@title 1. Colab/Drive 경로와 실행 설정
        import os, sys, json, hashlib, random, time, platform, re
        from pathlib import Path

        IN_COLAB = "google.colab" in sys.modules
        if IN_COLAB:
            from google.colab import drive
            drive.mount("/content/drive")

        if IN_COLAB:
            DATA_DIR = Path("/content/drive/MyDrive/BARAM2026/data")
            RUN_ROOT = Path("/content/drive/MyDrive/BARAM2026/hyejin_official_ablation")
        else:
            DATA_DIR = Path.cwd() / "data"
            RUN_ROOT = Path.cwd() / "runs" / "hyejin_official_ablation"

        RUN_MODE = "gate"  #@param ["smoke", "gate"]
        ACTIVE_EXPERIMENTS = "F0_REMOVE_REDUNDANT,F1_ADD_VEER" #@param {type:"string"}
        REBUILD_FEATURES = False
        FORCE_GBM = False
        FORCE_DL = False
        EXACT_FP32 = True
        SCADA_WEIGHT_VERSION = "ceil_missinghalf_v1"

        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        print("data   :", DATA_DIR)
        print("output :", RUN_ROOT)
        print("mode   :", RUN_MODE)
        """
    ),
    code(
        r"""
        #@title 2. 의존성, GPU, 재현성 설정
        import subprocess
        if IN_COLAB:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "lightgbm", "pyarrow", "openpyxl", "joblib"],
                check=True,
            )

        import numpy as np
        import pandas as pd
        import sklearn
        import torch
        import torch.nn as nn
        import lightgbm as lgb
        import joblib
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print("device:", DEVICE, "/", GPU_NAME)
        if IN_COLAB and "A100" not in GPU_NAME.upper():
            print("[주의] A100이 아닙니다. Colab 런타임 유형을 확인하세요.")

        torch.backends.cuda.matmul.allow_tf32 = not EXACT_FP32
        torch.backends.cudnn.allow_tf32 = not EXACT_FP32
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest" if EXACT_FP32 else "high")

        versions = {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "lightgbm": lgb.__version__,
            "gpu": GPU_NAME,
        }
        print(json.dumps(versions, ensure_ascii=False, indent=2))
        """
    ),
    code(
        r"""
        #@title 3. 공식 파일 감사와 실행 manifest
        REQUIRED = [
            "train/train_labels.csv", "train/ldaps_train.csv", "train/gfs_train.csv",
            "train/scada_vestas_train.csv", "train/scada_unison_train.csv",
            "test/ldaps_test.csv", "test/gfs_test.csv", "info.xlsx", "sample_submission.csv",
        ]

        def sha256_file(path, block=4 * 1024 * 1024):
            h = hashlib.sha256()
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(block)
                    if not chunk:
                        break
                    h.update(chunk)
            return h.hexdigest()

        missing = [rel for rel in REQUIRED if not (DATA_DIR / rel).exists()]
        assert not missing, "누락된 공식 파일:\n" + "\n".join(missing)

        asset_rows = []
        for rel in REQUIRED:
            p = DATA_DIR / rel
            asset_rows.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p)})
        asset_manifest = pd.DataFrame(asset_rows)
        display(asset_manifest.assign(size_mb=asset_manifest.bytes / 2**20))
        asset_manifest.to_json(
            RUN_ROOT / "official_asset_manifest.json", orient="records", force_ascii=False, indent=2
        )
        print("[확인] 외부 파일 0개, 공식 파일 manifest 저장 완료")
        """
    ),
    code(
        r"""
        #@title 4. 공통 상수, metric, fold 정의
        TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
        CAPACITY_KWH = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}
        GROUPS = [1, 2, 3]
        GIDX = {1: 0, 2: 1, 3: 2}

        FOLDS = [
            {"name": "fold_2024Q1", "train_end": pd.Timestamp("2024-01-01 00:00:00"),
             "valid_start": pd.Timestamp("2024-01-01 01:00:00"), "valid_end": pd.Timestamp("2024-04-01 00:00:00")},
            {"name": "fold_2024Q2", "train_end": pd.Timestamp("2024-04-01 00:00:00"),
             "valid_start": pd.Timestamp("2024-04-01 01:00:00"), "valid_end": pd.Timestamp("2024-07-01 00:00:00")},
            {"name": "fold_2024Q3", "train_end": pd.Timestamp("2024-07-01 00:00:00"),
             "valid_start": pd.Timestamp("2024-07-01 01:00:00"), "valid_end": pd.Timestamp("2024-10-01 00:00:00")},
            {"name": "fold_2024Q4", "train_end": pd.Timestamp("2024-10-01 00:00:00"),
             "valid_start": pd.Timestamp("2024-10-01 01:00:00"), "valid_end": pd.Timestamp("2025-01-01 00:00:00")},
        ]

        if RUN_MODE == "smoke":
            ACTIVE_FOLDS = [FOLDS[-1]]
            GBM_SEEDS, DL_SEEDS, GBM_TREES = [0], [0], 250
        else:
            ACTIVE_FOLDS = FOLDS
            GBM_SEEDS, DL_SEEDS, GBM_TREES = [0, 1, 2], [0, 1, 2], 500

        MAX_EPOCHS = 120
        ES_PATIENCE = 15
        ES_MIN_DELTA = 1e-5
        VAL_MONTHS = (3, 6, 9, 12)
        VAL_DAYS = tuple(range(11, 21))

        def metric_single_group(actual, forecast, capacity):
            actual = np.asarray(actual, dtype=float)
            forecast = np.asarray(forecast, dtype=float)
            valid = actual >= capacity * .10
            if not valid.any():
                return np.nan, np.nan, 0
            a, p = actual[valid], forecast[valid]
            error_rate = np.abs(p - a) / capacity
            nmae = error_rate.mean()
            unit_price = np.select([error_rate <= .06, error_rate <= .08], [4.0, 3.0], default=0.0)
            ficr = np.sum(a * unit_price) / np.sum(a * 4.0)
            return nmae, ficr, int(valid.sum())

        def metric_wide(answer, pred):
            nmae_rows, ficr_rows = [], []
            for col in TARGET_COLS:
                nmae, ficr, _ = metric_single_group(answer[col], pred[col], CAPACITY_KWH[col])
                nmae_rows.append(nmae); ficr_rows.append(ficr)
            one_minus_nmae = 1 - np.mean(nmae_rows)
            ficr = np.mean(ficr_rows)
            return .5 * one_minus_nmae + .5 * ficr, one_minus_nmae, ficr

        print("folds:", [f["name"] for f in ACTIVE_FOLDS])
        print("GBM seeds:", GBM_SEEDS, "/ DL seeds:", DL_SEEDS)
        """
    ),
    code(
        r"""
        #@title 5. 공식 데이터 피처 빌더
        def dms_to_decimal(value):
            pattern = r"(\d+)°(\d+)'([\d.]+)\"([NS])\s+(\d+)°(\d+)'([\d.]+)\"([EW])"
            match = re.match(pattern, str(value).strip())
            if not match:
                raise ValueError(f"좌표 파싱 실패: {value}")
            d1, m1, s1, h1, d2, m2, s2, h2 = match.groups()
            lat = float(d1) + float(m1) / 60 + float(s1) / 3600
            lon = float(d2) + float(m2) / 60 + float(s2) / 3600
            return (-lat if h1 == "S" else lat), (-lon if h2 == "W" else lon)

        def haversine(lat1, lon1, lat2, lon2):
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
            return 6371.0 * 2 * np.arcsin(np.sqrt(a))

        def load_group_centroids():
            raw = pd.read_excel(DATA_DIR / "info.xlsx", sheet_name="info", header=None)
            info = raw.iloc[4:21].copy(); info.columns = raw.iloc[3]
            info = info.reset_index(drop=True).rename(columns={"좌표(Google)": "coord", "KPX그룹": "group"})
            info["group"] = pd.to_numeric(info["group"], errors="coerce").ffill().astype(int)
            coords = info["coord"].map(dms_to_decimal)
            info["lat"] = coords.map(lambda x: x[0]); info["lon"] = coords.map(lambda x: x[1])
            return info.groupby("group")[["lat", "lon"]].mean()

        def nearest_grid_map(weather, centroids):
            grids = weather[["grid_id", "latitude", "longitude"]].drop_duplicates().set_index("grid_id")
            mapping = {}
            for group in GROUPS:
                distance = haversine(
                    centroids.loc[group, "lat"], centroids.loc[group, "lon"],
                    grids["latitude"].to_numpy(), grids["longitude"].to_numpy(),
                )
                mapping[group] = int(grids.index[np.argmin(distance)])
            return mapping

        def nearest_level_frame(weather, mapping, prefix):
            excluded = {"forecast_kst_dtm", "data_available_kst_dtm", "grid_id", "latitude", "longitude"}
            value_cols = [c for c in weather.columns if c not in excluded]
            frames = []
            for group, grid_id in mapping.items():
                sub = weather.loc[weather.grid_id.eq(grid_id), ["forecast_kst_dtm", *value_cols]].copy()
                sub = sub.rename(columns={c: f"{prefix}_{c}" for c in value_cols})
                sub["kpx_group"] = group
                frames.append(sub)
            return pd.concat(frames, ignore_index=True), [f"{prefix}_{c}" for c in value_cols]

        LD_SPATIAL = {
            "ws10": lambda d: np.hypot(d["heightAboveGround_10_10u"], d["heightAboveGround_10_10v"]),
            "u10": lambda d: d["heightAboveGround_10_10u"],
            "v10": lambda d: d["heightAboveGround_10_10v"],
            "ws50max": lambda d: np.hypot(d["heightAboveGround_50_50MUmax"], d["heightAboveGround_50_50MVmax"]),
            "wsbl": lambda d: np.hypot(d["heightAboveGround_5_XBLWS"], d["heightAboveGround_5_YBLWS"]),
            "t2": lambda d: d["heightAboveGround_2_t"],
            "sp": lambda d: d["surface_0_sp"],
            "blh": lambda d: d["etc_0_blh"],
        }
        GF_SPATIAL = {
            "ws10": lambda d: np.hypot(d["heightAboveGround_10_10u"], d["heightAboveGround_10_10v"]),
            "ws100": lambda d: np.hypot(d["heightAboveGround_100_100u"], d["heightAboveGround_100_100v"]),
            "u100": lambda d: d["heightAboveGround_100_100u"],
            "v100": lambda d: d["heightAboveGround_100_100v"],
            "gust": lambda d: d["surface_0_gust"],
            "ws850": lambda d: np.hypot(d["isobaricInhPa_850_u"], d["isobaricInhPa_850_v"]),
            "t850": lambda d: d["isobaricInhPa_850_t"],
            "prmsl": lambda d: d["meanSea_0_prmsl"],
        }

        def spatial_deviation_frame(weather, prefix, definitions, add_ws10_gradient=False):
            grid_meta = weather[["grid_id", "latitude", "longitude"]].drop_duplicates().sort_values("grid_id")
            grid_ids = grid_meta.grid_id.to_numpy()
            master_time = pd.Index(np.sort(weather.forecast_kst_dtm.unique()), name="forecast_kst_dtm")
            output_columns = {}
            gradient_cols = []

            lat = grid_meta.latitude.to_numpy(); lon = grid_meta.longitude.to_numpy()
            plane = np.column_stack([
                np.ones(len(grid_ids)),
                (lat - lat.mean()) * 111.0,
                (lon - lon.mean()) * 111.0 * np.cos(np.deg2rad(lat.mean())),
            ])
            plane_pinv = np.linalg.pinv(plane)

            for name, fn in definitions.items():
                values = weather[["forecast_kst_dtm", "grid_id"]].copy()
                values["value"] = np.asarray(fn(weather), dtype=np.float64)
                matrix = values.pivot_table(
                    index="forecast_kst_dtm", columns="grid_id", values="value", aggfunc="mean"
                ).reindex(index=master_time, columns=grid_ids)
                row_mean = matrix.mean(axis=1)
                matrix = matrix.apply(lambda col: col.fillna(row_mean))
                array = matrix.to_numpy()
                mean = np.nanmean(array, axis=1)
                output_columns[f"sd{prefix}_{name}_mean"] = mean
                for index, grid_id in enumerate(grid_ids):
                    output_columns[f"sd{prefix}_{name}_g{int(grid_id)}"] = array[:, index] - mean
                if add_ws10_gradient and name == "ws10":
                    coef = array @ plane_pinv.T
                    output_columns[f"grad{prefix}_ws10_dlat"] = coef[:, 1]
                    output_columns[f"grad{prefix}_ws10_dlon"] = coef[:, 2]
                    gradient_cols = [f"grad{prefix}_ws10_dlat", f"grad{prefix}_ws10_dlon"]
            out = pd.DataFrame(output_columns, index=master_time).reset_index()
            return out, [c for c in output_columns if c not in gradient_cols], gradient_cols

        def add_level_physics_time_features(df):
            df = df.sort_values(["kpx_group", "forecast_kst_dtm"]).copy()

            wind_pairs = {
                "ldaps_10m": ("ldaps_heightAboveGround_10_10u", "ldaps_heightAboveGround_10_10v"),
                "gfs_10m": ("gfs_heightAboveGround_10_10u", "gfs_heightAboveGround_10_10v"),
                "gfs_80m": ("gfs_heightAboveGround_80_u", "gfs_heightAboveGround_80_v"),
                "gfs_100m": ("gfs_heightAboveGround_100_100u", "gfs_heightAboveGround_100_100v"),
                "gfs_850": ("gfs_isobaricInhPa_850_u", "gfs_isobaricInhPa_850_v"),
            }
            derived = []
            for tag, (u_col, v_col) in wind_pairs.items():
                ws = np.hypot(df[u_col], df[v_col])
                df[f"{tag}_ws"] = ws
                df[f"{tag}_wd_sin"] = df[v_col] / (ws + 1e-6)
                df[f"{tag}_wd_cos"] = df[u_col] / (ws + 1e-6)
                derived += [f"{tag}_ws", f"{tag}_wd_sin", f"{tag}_wd_cos"]

            dt = df.forecast_kst_dtm
            df["cal_hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
            df["cal_hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
            df["cal_year_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
            df["cal_year_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)
            derived += ["cal_hour_sin", "cal_hour_cos", "cal_year_sin", "cal_year_cos"]

            rho = df["ldaps_surface_0_sp"] / (287.05 * df["ldaps_heightAboveGround_2_t"])
            df["phy_density"] = rho
            df["phy_ws10_cube"] = df["ldaps_10m_ws"] ** 3
            df["phy_ws100_cube"] = df["gfs_100m_ws"] ** 3
            df["phy_rho_ws100_cube"] = rho * df["phy_ws100_cube"]
            df["phy_power_curve"] = np.clip(df["gfs_100m_ws"], 0, 12) ** 3 / 12 ** 3
            ws50max = np.hypot(df["ldaps_heightAboveGround_50_50MUmax"], df["ldaps_heightAboveGround_50_50MVmax"])
            ws50min = np.hypot(df["ldaps_heightAboveGround_50_50MUmin"], df["ldaps_heightAboveGround_50_50MVmin"])
            df["phy_ws50max"] = ws50max
            df["phy_gust_range"] = ws50max - ws50min
            df["phy_gust_over_ws"] = ws50max / (df["ldaps_10m_ws"] + .5)
            df["phy_gfs_gust_factor"] = df["gfs_surface_0_gust"] / (df["gfs_10m_ws"] + .5)
            df["phy_shear_100_10"] = df["gfs_100m_ws"] - df["gfs_10m_ws"]
            df["phy_shear_850_100"] = df["gfs_850_ws"] - df["gfs_100m_ws"]
            df["phy_ws_disagreement"] = np.abs(df["ldaps_10m_ws"] - df["gfs_10m_ws"])
            df["phy_wd_agreement"] = (
                df["ldaps_10m_wd_sin"] * df["gfs_10m_wd_sin"]
                + df["ldaps_10m_wd_cos"] * df["gfs_10m_wd_cos"]
            )
            derived += [c for c in df if c.startswith("phy_")]

            block = (df.forecast_kst_dtm - pd.Timedelta(hours=1)).dt.date
            df["_block"] = block
            for source in ["ldaps_10m_ws", "gfs_100m_ws"]:
                tag = "l10" if source.startswith("ldaps") else "g100"
                grouped = df.groupby(["kpx_group", "_block"], sort=False)[source]
                for window in [3, 7]:
                    df[f"tmp_{tag}_r{window}_mean"] = grouped.transform(
                        lambda x: x.rolling(window, center=True, min_periods=1).mean()
                    )
                    df[f"tmp_{tag}_r{window}_std"] = grouped.transform(
                        lambda x: x.rolling(window, center=True, min_periods=1).std()
                    )
                    derived += [f"tmp_{tag}_r{window}_mean", f"tmp_{tag}_r{window}_std"]
                for offset in [1, 3]:
                    df[f"tmp_{tag}_prev{offset}"] = grouped.shift(offset).fillna(df[source])
                    df[f"tmp_{tag}_next{offset}"] = grouped.shift(-offset).fillna(df[source])
                    derived += [f"tmp_{tag}_prev{offset}", f"tmp_{tag}_next{offset}"]
            day_group = df.groupby(["kpx_group", "_block"], sort=False)["ldaps_10m_ws"]
            for stat in ["mean", "max", "std"]:
                name = f"tmp_day_{stat}"
                df[name] = day_group.transform(stat); derived.append(name)
            return df.drop(columns="_block"), list(dict.fromkeys(derived))

        def add_candidate_features(df):
            df = df.copy()
            df["f1_veer_sin"] = (
                df["gfs_100m_wd_sin"] * df["gfs_10m_wd_cos"]
                - df["gfs_100m_wd_cos"] * df["gfs_10m_wd_sin"]
            )
            df["f1_veer_cos"] = (
                df["gfs_100m_wd_cos"] * df["gfs_10m_wd_cos"]
                + df["gfs_100m_wd_sin"] * df["gfs_10m_wd_sin"]
            )
            f1_cols = ["f1_veer_sin", "f1_veer_cos"]

            f2_cols = []
            for group in GROUPS:
                flag = df.kpx_group.eq(group).astype(np.float32)
                for trig in ["sin", "cos"]:
                    name = f"f2_g{group}_wd_{trig}"
                    df[name] = flag * df[f"gfs_100m_wd_{trig}"]
                    f2_cols.append(name)

            east = df["gfs_100m_wd_cos"]
            north = df["gfs_100m_wd_sin"]
            grad_east = df["gradL_ws10_dlon"]
            grad_north = df["gradL_ws10_dlat"]
            df["f3_grad_along"] = grad_east * east + grad_north * north
            df["f3_grad_cross"] = -grad_east * north + grad_north * east
            df["f3_grad_magnitude"] = np.hypot(grad_east, grad_north)
            f3_cols = ["f3_grad_along", "f3_grad_cross", "f3_grad_magnitude"]
            return df, {"f1": f1_cols, "f2": f2_cols, "f3": f3_cols}

        def build_official_train_features():
            labels = pd.read_csv(DATA_DIR / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
            labels = labels.melt(id_vars="kst_dtm", value_vars=TARGET_COLS, var_name="target_col", value_name="target")
            labels["kpx_group"] = labels.target_col.str.extract(r"(\d+)").astype(int)
            labels = labels.rename(columns={"kst_dtm": "forecast_kst_dtm"}).drop(columns="target_col")

            ldaps = pd.read_csv(DATA_DIR / "train/ldaps_train.csv", encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"])
            gfs = pd.read_csv(DATA_DIR / "train/gfs_train.csv", encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"])
            centroids = load_group_centroids()
            ld_map = nearest_grid_map(ldaps, centroids); gf_map = nearest_grid_map(gfs, centroids)
            print("nearest LDAPS:", ld_map, "/ GFS:", gf_map)

            ld_level, ld_cols = nearest_level_frame(ldaps, ld_map, "ldaps")
            gf_level, gf_cols = nearest_level_frame(gfs, gf_map, "gfs")
            frame = labels.merge(ld_level, on=["forecast_kst_dtm", "kpx_group"], how="left", validate="many_to_one")
            frame = frame.merge(gf_level, on=["forecast_kst_dtm", "kpx_group"], how="left", validate="many_to_one")
            frame, derived_cols = add_level_physics_time_features(frame)

            ld_spatial, ld_spatial_cols, grad_cols = spatial_deviation_frame(
                ldaps, "L", LD_SPATIAL, add_ws10_gradient=True
            )
            gf_spatial, gf_spatial_cols, _ = spatial_deviation_frame(gfs, "G", GF_SPATIAL)
            frame = frame.merge(ld_spatial, on="forecast_kst_dtm", how="left", validate="many_to_one")
            frame = frame.merge(gf_spatial, on="forecast_kst_dtm", how="left", validate="many_to_one")
            frame, candidates = add_candidate_features(frame)

            base_cols = list(dict.fromkeys(ld_cols + gf_cols + derived_cols + ld_spatial_cols + gf_spatial_cols))
            assert not set(base_cols) & set(sum(candidates.values(), []))
            assert not set(grad_cols) & set(base_cols)
            assert frame[base_cols].select_dtypes(exclude="number").empty
            return frame, base_cols, candidates, {"ldaps": ld_map, "gfs": gf_map}
        """
    ),
    code(
        r"""
        #@title 6. 공식 피처 생성 또는 혜진 캐시 로드
        FEATURE_CACHE = RUN_ROOT / "official_train_features_v1.parquet"
        FEATURE_META = RUN_ROOT / "official_train_features_v1.json"
        feature_input_paths = ["train/train_labels.csv", "train/ldaps_train.csv", "train/gfs_train.csv", "info.xlsx"]
        current_feature_inputs = {
            row.path: row.sha256 for row in asset_manifest.itertuples() if row.path in feature_input_paths
        }

        cache_metadata = (
            json.loads(FEATURE_META.read_text(encoding="utf-8"))
            if FEATURE_META.exists() else {}
        )
        feature_cache_valid = (
            FEATURE_CACHE.exists()
            and cache_metadata.get("input_sha256") == current_feature_inputs
        )

        if feature_cache_valid and not REBUILD_FEATURES:
            train_df = pd.read_parquet(FEATURE_CACHE)
            train_df["forecast_kst_dtm"] = pd.to_datetime(train_df.forecast_kst_dtm)
            feature_meta = cache_metadata
            base_cols = feature_meta["base_cols"]
            candidate_cols = feature_meta["candidate_cols"]
            print("[확인] 혜진 피처 캐시 로드")
        else:
            t0 = time.time()
            train_df, base_cols, candidate_cols, grid_maps = build_official_train_features()
            train_df.to_parquet(FEATURE_CACHE, index=False)
            feature_meta = {
                "base_cols": base_cols, "candidate_cols": candidate_cols,
                "grid_maps": grid_maps, "created_at": pd.Timestamp.now().isoformat(),
                "input_sha256": current_feature_inputs,
            }
            FEATURE_META.write_text(json.dumps(feature_meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[확인] 공식 피처 생성 및 캐시 저장: {(time.time()-t0)/60:.1f}분")

        print("train shape:", train_df.shape)
        print("base features:", len(base_cols), "/ candidates:", {k: len(v) for k, v in candidate_cols.items()})
        display(train_df.groupby("kpx_group")["target"].agg(rows="size", labels="count"))
        """
    ),
    code(
        r"""
        #@title 7. 공식 SCADA에서 정렬 보정 cln03 가용성 가중 생성
        def build_scada_availability():
            specs = [("vestas", range(1, 7), 1), ("vestas", range(7, 13), 2), ("unison", range(1, 6), 3)]
            loaded = {
                maker: pd.read_csv(
                    DATA_DIR / f"train/scada_{maker}_train.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"]
                ) for maker in ["vestas", "unison"]
            }
            parts = []
            for maker, turbines, group in specs:
                data = loaded[maker].copy()
                # SCADA 10분 값과 발전량 라벨은 구간 종료시각 기준이다.
                # 로컬 확인과 신영 최신 코드 모두 ceil 정렬에서 그룹3 라벨 상관이 약 .996~.997이었다.
                data["hour"] = data.kst_dtm.dt.ceil("h")
                power_cols = [f"{maker}_wtg{i:02d}_power_kw10m" for i in turbines]
                wind_cols = [f"{maker}_wtg{i:02d}_ws" for i in turbines]
                median_ws = data[wind_cols].median(axis=1)
                confirmed_down, missing_down = [], []
                for power_col, wind_col in zip(power_cols, wind_cols):
                    effective_ws = data[wind_col].fillna(median_ws)
                    windy = effective_ws.ge(5)
                    missing = data[power_col].isna()
                    dead = (~data[power_col].gt(1)) & windy & ~missing
                    missing_windy = missing & windy
                    dead_count = dead.groupby(data.hour).sum()
                    missing_count = missing_windy.groupby(data.hour).sum()
                    slots = data.groupby("hour")[power_col].size()
                    threshold = np.maximum(slots - 1, 5)
                    confirmed_down.append(dead_count.ge(threshold).astype(float))
                    missing_down.append(missing_count.ge(threshold).astype(float))
                # 명시적 무출력은 확정 정지, 수집 결측은 불확실성을 인정해 절반만 반영한다.
                unavailable = sum(confirmed_down) + .5 * sum(missing_down)
                availability_value = np.clip(1 - unavailable.to_numpy() / len(list(turbines)), 0, 1)
                parts.append(pd.DataFrame({
                    "forecast_kst_dtm": unavailable.index,
                    "kpx_group": group,
                    "avail_strict": availability_value,
                }))
            return pd.concat(parts, ignore_index=True)

        def availability_factor(values):
            values = np.nan_to_num(np.asarray(values, dtype=float), nan=1.0)
            return np.where(values < .9, .05, np.where(values < .999, .3, 1.0)).astype(np.float32)

        availability = build_scada_availability()
        train_df = train_df.merge(
            availability, on=["forecast_kst_dtm", "kpx_group"], how="left", validate="many_to_one"
        )
        assert len(train_df) == 3 * train_df.forecast_kst_dtm.nunique()
        display(availability.groupby("kpx_group").avail_strict.agg(
            hours="size", mean="mean", full=lambda x: x.ge(.999).mean(), below90=lambda x: x.lt(.9).mean()
        ).round(4))
        print(f"[확인] SCADA weight={SCADA_WEIGHT_VERSION}; train-only 가중이며 입력 피처에는 포함하지 않습니다.")
        print("[확인] UNISON 양자화 이후 0출력을 임의로 완화하는 규칙은 팀 실험에서 악화되어 사용하지 않습니다.")
        """
    ),
    code(
        r"""
        #@title 8. F0 제거 목록과 실험 피처셋 확정
        numeric = train_df[base_cols].replace([np.inf, -np.inf], np.nan)
        constant_cols = [c for c in base_cols if numeric[c].nunique(dropna=False) <= 1]
        duplicate_of = {}
        buckets = {}
        for col in [c for c in base_cols if c not in constant_cols]:
            signature = int(pd.util.hash_pandas_object(numeric[col], index=False).sum())
            matched = False
            for previous in buckets.get(signature, []):
                if numeric[col].equals(numeric[previous]):
                    duplicate_of[col] = previous; matched = True; break
            if not matched:
                buckets.setdefault(signature, []).append(col)

        clean_cols = [c for c in base_cols if c not in constant_cols and c not in duplicate_of]
        feature_sets = {
            "C0_OFFICIAL_BASE": base_cols,
            "F0_REMOVE_REDUNDANT": clean_cols,
            "F1_ADD_VEER": clean_cols + candidate_cols["f1"],
            "F2_ADD_GROUP_DIRECTION": clean_cols + candidate_cols["f2"],
            "F3_ADD_ALONG_GRADIENT": clean_cols + candidate_cols["f3"],
        }
        active_names = [name.strip() for name in ACTIVE_EXPERIMENTS.split(",") if name.strip()]
        unknown = [name for name in active_names if name not in feature_sets]
        assert not unknown, f"정의되지 않은 실험: {unknown}"
        assert "F0_REMOVE_REDUNDANT" in active_names, "비교 기준 F0_REMOVE_REDUNDANT를 포함하세요."

        feature_manifest = {
            "constant_cols": constant_cols, "duplicate_of": duplicate_of,
            "feature_sets": feature_sets, "active_experiments": active_names,
        }
        (RUN_ROOT / "feature_ablation_manifest.json").write_text(
            json.dumps(feature_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("constant:", constant_cols)
        print("exact duplicate:", duplicate_of)
        print({name: len(cols) for name, cols in feature_sets.items()})
        print("active:", active_names)
        """
    ),
    code(
        r"""
        #@title 9. 고정 GBM과 fold별 bias map을 처음부터 학습
        def sample_weights(frame):
            caps = frame.kpx_group.map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
            cf = frame.target.to_numpy() / caps
            return (np.where(cf < .10, .05, 1.0) * availability_factor(frame.avail_strict)).astype(np.float32)

        def make_lgbm(seed, trees):
            return lgb.LGBMRegressor(
                objective="mae", n_estimators=trees, learning_rate=.03,
                max_depth=6, num_leaves=31, min_child_samples=20,
                subsample=.8, subsample_freq=1, colsample_bytree=.8,
                random_state=seed, n_jobs=-1, verbosity=-1,
            )

        def fit_bias_map(fold, cols):
            result = {}
            for group in GROUPS:
                data = train_df.loc[
                    train_df.kpx_group.eq(group) & train_df.target.notna()
                    & train_df.forecast_kst_dtm.le(fold["train_end"])
                ].sort_values("forecast_kst_dtm").copy()
                first_date = data.forecast_kst_dtm.min()
                available_days = max((fold["train_end"] - first_date).days, 1)
                holdout_days = min(int(available_days * .25), 180)
                if available_days - holdout_days < 90:
                    holdout_days = max(0, available_days - 90)
                split_time = fold["train_end"] - pd.Timedelta(days=holdout_days)
                fit = data.loc[data.forecast_kst_dtm.le(split_time)]
                check = data.loc[data.forecast_kst_dtm.gt(split_time)]
                if fit.empty or check.empty:
                    result[group] = []
                    continue
                imputer = SimpleImputer(strategy="median", keep_empty_features=True)
                X_fit = imputer.fit_transform(fit[cols].replace([np.inf, -np.inf], np.nan))
                X_check = imputer.transform(check[cols].replace([np.inf, -np.inf], np.nan))
                model = make_lgbm(260806 + group, min(GBM_TREES, 300))
                model.fit(X_fit, fit.target, sample_weight=sample_weights(fit))
                pred = model.predict(X_check)
                cap = CAPACITY_KWH[f"kpx_group_{group}"]
                valid = check.target.to_numpy() >= cap * .10
                ws = check.ldaps_10m_ws.to_numpy()
                error = (pred - check.target.to_numpy()) / cap
                rows = []
                for left, right in [(0, 11), (11, 13), (13, 100)]:
                    mask = valid & (ws > left) & (ws <= right)
                    rows.append({"left": left, "right": right, "bias": float(np.mean(error[mask])) if mask.any() else 0.0})
                result[group] = rows
            return result

        gbm_feature_hash = hashlib.sha256(
            json.dumps(clean_cols, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:10]
        cache_tag = f"{RUN_MODE}_gbm{len(GBM_SEEDS)}x{GBM_TREES}_{gbm_feature_hash}_{SCADA_WEIGHT_VERSION}_v2"
        GBM_CACHE = RUN_ROOT / f"fixed_gbm_{cache_tag}.parquet"
        BIAS_CACHE = RUN_ROOT / f"bias_maps_{cache_tag}.joblib"

        fixed_oof = pd.DataFrame()
        bias_maps = {}
        if GBM_CACHE.exists() and BIAS_CACHE.exists() and not FORCE_GBM:
            fixed_oof = pd.read_parquet(GBM_CACHE)
            fixed_oof["forecast_kst_dtm"] = pd.to_datetime(fixed_oof.forecast_kst_dtm)
            bias_maps = joblib.load(BIAS_CACHE)
            print(f"[확인] 혜진 고정 GBM 체크포인트 로드: {sorted(fixed_oof.fold.unique())}")

        required_folds = {fold["name"] for fold in ACTIVE_FOLDS}
        completed_folds = set(fixed_oof.fold.unique()) if not fixed_oof.empty else set()
        if FORCE_GBM or not required_folds.issubset(completed_folds):
            rows = [] if FORCE_GBM or fixed_oof.empty else [fixed_oof]
            if FORCE_GBM:
                bias_maps = {}; completed_folds = set()
            gbm_cols = clean_cols
            for fold in ACTIVE_FOLDS:
                if fold["name"] in completed_folds:
                    continue
                t0 = time.time(); bias_maps[fold["name"]] = fit_bias_map(fold, gbm_cols)
                train_window = train_df.loc[
                    train_df.forecast_kst_dtm.le(fold["train_end"]) & train_df.target.notna()
                ]
                valid_window = train_df.loc[
                    train_df.forecast_kst_dtm.between(fold["valid_start"], fold["valid_end"])
                    & train_df.target.notna()
                ]
                for group in GROUPS:
                    tr = train_window.loc[train_window.kpx_group.eq(group)]
                    va = valid_window.loc[valid_window.kpx_group.eq(group)].sort_values("forecast_kst_dtm")
                    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
                    Xtr = imputer.fit_transform(tr[gbm_cols].replace([np.inf, -np.inf], np.nan))
                    Xva = imputer.transform(va[gbm_cols].replace([np.inf, -np.inf], np.nan))
                    predictions = []
                    for seed in GBM_SEEDS:
                        model = make_lgbm(seed, GBM_TREES)
                        model.fit(Xtr, tr.target, sample_weight=sample_weights(tr))
                        predictions.append(model.predict(Xva))
                    cap = CAPACITY_KWH[f"kpx_group_{group}"]
                    rows.append(pd.DataFrame({
                        "fold": fold["name"], "group": group,
                        "forecast_kst_dtm": va.forecast_kst_dtm.to_numpy(),
                        "target": va.target.to_numpy(), "ws": va.ldaps_10m_ws.to_numpy(),
                        "gbm_cf": np.mean(predictions, axis=0) / cap,
                    }))
                print(f"  {fold['name']} GBM+bias complete: {(time.time()-t0)/60:.1f}분")
                fixed_oof = pd.concat(rows, ignore_index=True)
                fixed_oof.to_parquet(GBM_CACHE, index=False)
                joblib.dump(bias_maps, BIAS_CACHE)
                print(f"  [체크포인트] {fold['name']} 저장")

        fixed_oof = fixed_oof.loc[fixed_oof.fold.isin(required_folds)].copy().reset_index(drop=True)
        missing_oof_targets = int(fixed_oof.target.isna().sum())
        if missing_oof_targets:
            print(f"[확인] 공식 라벨 결측 OOF {missing_oof_targets}행 제외")
            fixed_oof = fixed_oof.loc[fixed_oof.target.notna()].copy().reset_index(drop=True)
        assert set(fixed_oof.fold.unique()) == required_folds

        assert fixed_oof.gbm_cf.notna().all()
        display(fixed_oof.groupby(["fold", "group"]).size().unstack(fill_value=0))
        """
    ),
    code(
        r"""
        #@title 10. 멀티태스크 MLP와 exp-132 early stopping
        def routed_heads(hidden, group_index, heads):
            output = torch.zeros(hidden.shape[0], device=hidden.device, dtype=hidden.dtype)
            for group in range(3):
                mask = group_index.eq(group)
                if mask.any():
                    output[mask] = heads[group](hidden[mask]).squeeze(1)
            return output

        class MTMLP(nn.Module):
            def __init__(self, n_features, width=384, depth=3, dropout=.15, embedding=8):
                super().__init__()
                self.embedding = nn.Embedding(3, embedding)
                layers, input_width = [], n_features + embedding
                for _ in range(depth):
                    layers += [nn.Linear(input_width, width), nn.BatchNorm1d(width), nn.SiLU(), nn.Dropout(dropout)]
                    input_width = width
                self.trunk = nn.Sequential(*layers)
                self.heads = nn.ModuleList([nn.Linear(width, 1) for _ in range(3)])

            def forward(self, features, group_index):
                hidden = self.trunk(torch.cat([features, self.embedding(group_index)], dim=1))
                return routed_heads(hidden, group_index, self.heads)

        def q60_huber(prediction, target, weight, delta=.05, tau=.60):
            error = prediction - target
            absolute = error.abs()
            huber = torch.where(absolute <= delta, .5 * error.square(), delta * (absolute - .5 * delta))
            asymmetric = torch.where(error < 0, 2 * tau, 2 * (1 - tau))
            return (huber * asymmetric * weight).sum() / weight.sum()

        def seasonal_validation_mask(frame):
            dt = frame.forecast_kst_dtm
            last_year = int((dt - pd.Timedelta(hours=1)).dt.year.max())
            mask = dt.dt.year.eq(last_year) & dt.dt.month.isin(VAL_MONTHS) & dt.dt.day.isin(VAL_DAYS)
            assert mask.any(), f"seasonal validation 0 rows: {last_year}"
            return mask, last_year

        def seed_all(seed):
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        def xy(frame, cols, imputer, scaler, fit):
            raw = frame[cols].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
            transformed = scaler.fit_transform(imputer.fit_transform(raw)) if fit else scaler.transform(imputer.transform(raw))
            group_index = frame.kpx_group.map(GIDX).to_numpy(np.int64)
            caps = frame.kpx_group.map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
            target = (frame.target.to_numpy() / caps).astype(np.float32)
            weight = sample_weights(frame)
            return transformed.astype(np.float32), group_index, target, weight

        def prepare_dl_fold(fold, cols):
            full_train = train_df.loc[
                train_df.forecast_kst_dtm.le(fold["train_end"]) & train_df.target.notna()
            ].copy()
            validation_mask, validation_year = seasonal_validation_mask(full_train)
            fit = full_train.loc[~validation_mask]
            early_validation = full_train.loc[validation_mask]

            keys = fixed_oof.loc[fixed_oof.fold.eq(fold["name"]), ["group", "forecast_kst_dtm"]].copy()
            keys["order"] = np.arange(len(keys))
            outer_validation = keys.merge(
                train_df, left_on=["group", "forecast_kst_dtm"],
                right_on=["kpx_group", "forecast_kst_dtm"], how="left", validate="one_to_one"
            ).sort_values("order")
            assert outer_validation.target.notna().all()

            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            scaler = StandardScaler()
            Xtr, gtr, ytr, wtr = xy(fit, cols, imputer, scaler, True)
            Xes, ges, yes, wes = xy(early_validation, cols, imputer, scaler, False)
            Xout = scaler.transform(imputer.transform(
                outer_validation[cols].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
            )).astype(np.float32)
            gout = outer_validation.kpx_group.map(GIDX).to_numpy(np.int64)
            metadata = {
                "validation_year": validation_year,
                "validation_rows": int(len(early_validation)),
                "validation_days": int(early_validation.forecast_kst_dtm.dt.normalize().nunique()),
                "train_rows": int(len(fit)),
            }
            return Xtr, gtr, ytr, wtr, Xes, ges, yes, wes, Xout, gout, metadata

        @torch.no_grad()
        def predict_batches(model, features, group_index, batch_size=16384):
            model.eval(); predictions = []
            for start in range(0, len(features), batch_size):
                x = torch.as_tensor(features[start:start+batch_size], device=DEVICE)
                g = torch.as_tensor(group_index[start:start+batch_size], device=DEVICE)
                predictions.append(model(x, g).float().cpu().numpy())
            return np.concatenate(predictions)

        def fit_predict_seed(data, seed):
            Xtr, gtr, ytr, wtr, Xes, ges, yes, wes, Xout, gout = data
            seed_all(seed)
            model = MTMLP(Xtr.shape[1]).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
            Xt = torch.as_tensor(Xtr, device=DEVICE); gt = torch.as_tensor(gtr, device=DEVICE)
            yt = torch.as_tensor(ytr, device=DEVICE); wt = torch.as_tensor(wtr, device=DEVICE)
            Xv = torch.as_tensor(Xes, device=DEVICE); gv = torch.as_tensor(ges, device=DEVICE)
            yv = torch.as_tensor(yes, device=DEVICE); wv = torch.as_tensor(wes, device=DEVICE)
            best_loss, best_state, best_epoch, bad = np.inf, None, 0, 0
            batch_size = 1024
            for epoch in range(MAX_EPOCHS):
                model.train(); order = torch.randperm(len(Xt), device=DEVICE); train_total = 0.0
                for start in range(0, len(Xt), batch_size):
                    index = order[start:start+batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    loss = q60_huber(model(Xt[index], gt[index]), yt[index], wt[index])
                    loss.backward(); optimizer.step()
                    train_total += float(loss.detach()) * len(index)
                scheduler.step(); model.eval()
                with torch.no_grad():
                    validation_loss = float(q60_huber(model(Xv, gv), yv, wv))
                if validation_loss < best_loss - ES_MIN_DELTA:
                    best_loss, best_epoch, bad = validation_loss, epoch + 1, 0
                    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                else:
                    bad += 1
                if epoch == 0 or (epoch + 1) % 10 == 0 or bad >= ES_PATIENCE:
                    print(f"      ep {epoch+1:3d} train={train_total/len(Xt):.6f} "
                          f"val={validation_loss:.6f} best={best_epoch} bad={bad}")
                if bad >= ES_PATIENCE:
                    break
            assert best_state is not None
            model.load_state_dict(best_state)
            prediction = predict_batches(model, Xout, gout)
            del model, Xt, gt, yt, wt, Xv, gv, yv, wv, best_state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return prediction, best_epoch, best_loss
        """
    ),
    code(
        r"""
        #@title 11. DL 피처 ablation 실행과 자체 예측 저장
        RUN_DIR = RUN_ROOT / f"{RUN_MODE}_official_feature_ablation_{SCADA_WEIGHT_VERSION}_v2"
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        run_config = {
            "mode": RUN_MODE, "active_folds": [f["name"] for f in ACTIVE_FOLDS],
            "gbm_seeds": GBM_SEEDS, "dl_seeds": DL_SEEDS, "gbm_trees": GBM_TREES,
            "experiments": active_names, "feature_counts": {n: len(feature_sets[n]) for n in active_names},
            "early_stopping": {"max_epochs": MAX_EPOCHS, "patience": ES_PATIENCE,
                               "min_delta": ES_MIN_DELTA, "months": VAL_MONTHS, "days": VAL_DAYS},
            "fixed": {"gbm_dl": [.65, .35], "alpha": .40, "clip_lo": .16,
                      "scales": {"1": .98, "2": .98, "3": .97}, "snap": False,
                      "loss": "q60_huber", "sample_weight": "low_cf+cln03",
                      "scada_weight_version": SCADA_WEIGHT_VERSION},
            "versions": versions,
        }
        (RUN_DIR / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(run_config, ensure_ascii=False, indent=2))

        dl_predictions, epoch_rows = {}, []
        for experiment in active_names:
            cols = feature_sets[experiment]
            all_fold_predictions = []
            print(f"\n===== {experiment}: {len(cols)} features =====")
            for fold in ACTIVE_FOLDS:
                t0 = time.time()
                prepared = prepare_dl_fold(fold, cols)
                arrays, split_metadata = prepared[:-1], prepared[-1]
                print(f"  {fold['name']} inner validation: {split_metadata}")
                seed_predictions = []
                for seed in DL_SEEDS:
                    signature_payload = {
                        "experiment": experiment, "columns": cols, "fold": fold["name"], "seed": seed,
                        "max_epochs": MAX_EPOCHS, "patience": ES_PATIENCE, "min_delta": ES_MIN_DELTA,
                        "loss": "q60_huber", "architecture": "MTMLP_384x3_drop015",
                        "scada_weight_version": SCADA_WEIGHT_VERSION,
                    }
                    signature = hashlib.sha256(
                        json.dumps(signature_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
                    ).hexdigest()[:10]
                    prediction_file = RUN_DIR / f"{experiment}__{fold['name']}__seed{seed}__{signature}.npy"
                    metadata_file = RUN_DIR / f"{experiment}__{fold['name']}__seed{seed}__{signature}.json"
                    if prediction_file.exists() and metadata_file.exists() and not FORCE_DL:
                        prediction = np.load(prediction_file)
                        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                        print(f"  seed{seed}: cache, best_epoch={metadata['best_epoch']}")
                    else:
                        print(f"  seed{seed}: train")
                        prediction, best_epoch, best_loss = fit_predict_seed(arrays, seed)
                        metadata = {
                            **split_metadata, "experiment": experiment, "fold": fold["name"], "seed": seed,
                            "best_epoch": best_epoch, "best_validation_q60": best_loss,
                        }
                        np.save(prediction_file, prediction)
                        metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
                    expected = fixed_oof.fold.eq(fold["name"]).sum()
                    assert len(prediction) == expected
                    seed_predictions.append(prediction); epoch_rows.append(metadata)
                fold_prediction = np.mean(seed_predictions, axis=0)
                fold_keys = fixed_oof.loc[fixed_oof.fold.eq(fold["name"]), ["fold", "group", "forecast_kst_dtm"]].copy()
                fold_keys["dl_cf"] = fold_prediction
                all_fold_predictions.append(fold_keys)
                print(f"  {fold['name']} complete: {(time.time()-t0)/60:.1f}분")
                del prepared, arrays
            dl_predictions[experiment] = pd.concat(all_fold_predictions, ignore_index=True)
            dl_predictions[experiment].to_parquet(RUN_DIR / f"oof_{experiment}.parquet", index=False)

        epoch_table = pd.DataFrame(epoch_rows).drop_duplicates(["experiment", "fold", "seed"], keep="last")
        epoch_table.to_csv(RUN_DIR / "best_epochs.csv", index=False)
        display(epoch_table.pivot_table(index="fold", columns="experiment", values="best_epoch", aggfunc=list))
        print("[완료] DL OOF 저장:", RUN_DIR)
        """
    ),
    code(
        r"""
        #@title 12. no-snap 평가와 F0 대비 판정
        W_GBM = .65
        ALPHA = .40
        CLIP_LO = .16
        SCALES = {1: .98, 2: .98, 3: .97}

        def apply_bias(prediction, ws, group, bias_map):
            output = np.asarray(prediction, dtype=float).copy()
            for row in bias_map.get(int(group), []):
                mask = (ws > row["left"]) & (ws <= row["right"])
                output[mask] *= 1 - row["bias"]
            return output

        def postprocess_nosnap(cf, ws, group, bias_map):
            cap = CAPACITY_KWH[f"kpx_group_{int(group)}"]
            power = apply_bias(np.asarray(cf) * cap, np.asarray(ws), group, bias_map)
            rolling = pd.Series(power).rolling(3, center=True, min_periods=1).mean().to_numpy()
            power = ALPHA * power + (1 - ALPHA) * rolling
            power = np.clip(power, cap * CLIP_LO, cap)
            return np.clip(power * SCALES[int(group)], 0, cap)

        def prediction_tidy(experiment):
            dl = dl_predictions[experiment]
            frame = fixed_oof.merge(dl, on=["fold", "group", "forecast_kst_dtm"], how="left", validate="one_to_one")
            parts = []
            for (fold, group), sub in frame.groupby(["fold", "group"], sort=False):
                sub = sub.sort_values("forecast_kst_dtm").copy()
                blend = W_GBM * sub.gbm_cf.to_numpy() + (1 - W_GBM) * sub.dl_cf.to_numpy()
                sub["pred"] = postprocess_nosnap(blend, sub.ws.to_numpy(), group, bias_maps[fold])
                sub["experiment"] = experiment
                parts.append(sub)
            return pd.concat(parts, ignore_index=True)

        def score_tidy(frame):
            answer_parts, prediction_parts = [], []
            for fold, sub in frame.groupby("fold", sort=False):
                answer = sub.pivot(index="forecast_kst_dtm", columns="group", values="target")
                prediction = sub.pivot(index="forecast_kst_dtm", columns="group", values="pred")
                answer.columns = [f"kpx_group_{int(col)}" for col in answer.columns]
                prediction.columns = [f"kpx_group_{int(col)}" for col in prediction.columns]
                answer_parts.append(answer.reindex(columns=TARGET_COLS).fillna(0))
                prediction_parts.append(prediction.reindex(columns=TARGET_COLS).fillna(0))
            return metric_wide(pd.concat(answer_parts), pd.concat(prediction_parts))

        all_tidy = pd.concat([prediction_tidy(name) for name in active_names], ignore_index=True)
        all_tidy.to_parquet(RUN_DIR / "all_predictions_nosnap.parquet", index=False)
        summary_rows, fold_rows, group_rows = [], [], []
        for experiment, frame in all_tidy.groupby("experiment", sort=False):
            score, one_minus_nmae, ficr = score_tidy(frame)
            summary_rows.append({"experiment": experiment, "Score": score, "1-NMAE": one_minus_nmae, "FiCR": ficr})
            for fold, sub in frame.groupby("fold"):
                fs, fn, ff = score_tidy(sub)
                fold_rows.append({"experiment": experiment, "fold": fold, "Score": fs, "1-NMAE": fn, "FiCR": ff})
            for group, sub in frame.groupby("group"):
                cap = CAPACITY_KWH[f"kpx_group_{int(group)}"]
                nmae, group_ficr, count = metric_single_group(sub.target, sub.pred, cap)
                group_rows.append({"experiment": experiment, "group": group,
                                   "Score": .5 * (1 - nmae) + .5 * group_ficr,
                                   "1-NMAE": 1 - nmae, "FiCR": group_ficr, "n_eval": count})

        summary = pd.DataFrame(summary_rows).set_index("experiment")
        reference = "F0_REMOVE_REDUNDANT"
        for column in ["Score", "1-NMAE", "FiCR"]:
            summary[f"delta_{column}_vs_F0"] = summary[column] - summary.loc[reference, column]
        fold_table = pd.DataFrame(fold_rows)
        reference_folds = fold_table.loc[fold_table.experiment.eq(reference)].set_index("fold").Score
        fold_table["delta_Score_vs_F0"] = fold_table.apply(
            lambda row: row.Score - reference_folds.get(row.fold, np.nan), axis=1
        )

        print("\n=== pooled ===")
        display(summary.sort_values("Score", ascending=False).round(6))
        print("\n=== fold ===")
        display(fold_table.sort_values(["fold", "Score"], ascending=[True, False]).round(6))
        print("\n=== group ===")
        display(pd.DataFrame(group_rows).sort_values(["group", "Score"], ascending=[True, False]).round(6))
        summary.reset_index().to_csv(RUN_DIR / "summary.csv", index=False)
        fold_table.to_csv(RUN_DIR / "fold_scores.csv", index=False)

        if RUN_MODE == "smoke":
            print("\n[주의] smoke는 Q4/1-seed 배관 점검입니다. 가장 좋은 피처 1~2개만 F0와 함께 gate로 올리세요.")
        else:
            print("\n판정 기준: F0 대비 pooled +0.0005 이상, 4개 fold 중 3개 이상 비음수, 두 세부지표의 큰 희생 없음.")
            for experiment in active_names:
                if experiment in {"C0_OFFICIAL_BASE", reference}:
                    continue
                row = summary.loc[experiment]
                fold_delta = fold_table.loc[fold_table.experiment.eq(experiment), "delta_Score_vs_F0"]
                promote = (
                    row["delta_Score_vs_F0"] >= .0005
                    and (fold_delta >= -.0002).sum() >= 3
                    and row["delta_1-NMAE_vs_F0"] >= -.0002
                    and row["delta_FiCR_vs_F0"] >= -.0002
                )
                print(f"  {experiment}: {'PROMOTE' if promote else '보류/기각'}")
        """
    ),
    md(
        r"""
        ### 13. 실행 순서

        Smoke 결과에서 `F1_ADD_VEER`가 F0보다 `+0.006658` 높아 gate 후보로 선정되었습니다.

        1. 기본값인 `RUN_MODE='gate'`로 실행합니다.
        2. `F0_REMOVE_REDUNDANT`와 `F1_ADD_VEER`를 `4-fold × 3-seed`로 비교합니다.
        3. pooled 개선, fold 일관성, 두 세부지표를 모두 통과한 경우에만 챔피언 파이프라인에 제안합니다.

        `C0−F0`는 중복 제거 효과이고, `F1/F2/F3−F0`가 각 신규 피처의 순수 효과입니다. 서로 다른 신규 피처를 한 실험에 묶지 않았기 때문에 개선 원인을 설명할 수 있습니다.

        117m 풍속은 100m 풍속과 거의 같은 순위 신호였으므로 이번 1차 실험에서는 제외했습니다. F1~F3 중 승자가 나온 뒤 별도 실험으로 추가합니다.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": OUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT)
