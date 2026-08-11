#!/usr/bin/env python3
"""Build a data-understanding and feature-audit notebook for BARAM 2026."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_feature.ipynb"


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(s).strip() + "\n"}


def code(s: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": dedent(s).strip() + "\n"}


cells = [
    md(
        """
        ## BARAM 2026 — 01. 데이터 이해와 Feature Audit

        이 노트북은 모델을 학습하지 않습니다. 먼저 다음 질문에 답합니다.

        1. 그룹별 발전량 분포와 계절·시간·연도 변화는 어떤가?
        2. LDAPS/GFS의 시간축, 발표 시점, 격자 구조와 train→test shift는 어떤가?
        3. SCADA는 어떤 품질 문제와 가용성 신호를 갖는가?
        4. 현재 348개 입력에 이미 들어간 물리 피처와, 아직 실험할 가치가 있는 피처는 무엇인가?
        5. 무엇을 추가하기 전에 무엇을 제거해야 하는가?

        결과물은 `feature_audit/`에 CSV와 그림으로 저장합니다. 여기서 선정한 **한 개 feature family씩** exp-132 early-stopping control에 붙이는 것이 다음 단계입니다.
        """
    ),
    code(
        """
        #@title 0. 경로·라이브러리 설정
        import sys, subprocess, re, json, math, warnings
        from pathlib import Path

        IN_COLAB = "google.colab" in sys.modules
        if IN_COLAB:
            from google.colab import drive
            drive.mount("/content/drive")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "openpyxl", "pyarrow", "matplotlib", "seaborn"], check=True)
            DATA_DIR = Path("/content/drive/MyDrive/BARAM2026/wonwoo_bundle")
            OUT_DIR = Path("/content/drive/MyDrive/BARAM2026/hyejin_feature_audit")
        else:
            DATA_DIR = Path("data")
            OUT_DIR = Path("outputs/feature_audit")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from IPython.display import display

        sns.set_theme(style="whitegrid")
        pd.set_option("display.max_columns", 120)
        pd.set_option("display.width", 180)
        warnings.filterwarnings("ignore", category=FutureWarning)

        REQUIRED = [
            "train/train_labels.csv", "train/ldaps_train.csv", "train/gfs_train.csv",
            "train/scada_vestas_train.csv", "train/scada_unison_train.csv",
            "test/ldaps_test.csv", "test/gfs_test.csv", "info.xlsx", "sample_submission.csv",
        ]
        missing = [x for x in REQUIRED if not (DATA_DIR / x).exists()]
        assert not missing, f"누락 파일: {missing}"
        print("DATA:", DATA_DIR.resolve())
        print("OUT :", OUT_DIR.resolve())
        """
    ),
    md(
        """
        ### 1. 메타데이터와 타깃

        `00:00` 행은 이전 24시간 예보 블록의 마지막 시각입니다. 연도·월 단위 분석에서는 `kst_dtm - 1시간`으로 block date를 정의해 `2025-01-01 00:00` 한 행이 2025 학습 데이터처럼 오해되지 않게 합니다.
        """
    ),
    code(
        """
        #@title 1-1. 터빈·그룹 메타데이터
        def dms_to_decimal(text):
            parts = re.findall(r"(\\d+)°(\\d+).([\\d.]+).([NSEW])", str(text))
            assert len(parts) == 2, text
            out = []
            for deg, minute, sec, direction in parts:
                value = float(deg) + float(minute) / 60 + float(sec) / 3600
                out.append(-value if direction in "SW" else value)
            return out

        raw_info = pd.read_excel(DATA_DIR / "info.xlsx", sheet_name="info", header=None)
        turbine = raw_info.iloc[4:21].copy()
        turbine.columns = raw_info.iloc[3]
        turbine = turbine.reset_index(drop=True).rename(columns={
            "단계":"stage", "명칭":"site", "제작사":"maker", "모델명":"model",
            "호기":"turbine_no", "좌표(Google)":"coord", "KPX그룹":"group",
            "Hub Height(m)":"hub_height", "Rotor Diameter(m)":"rotor_diameter",
            "설비용량(MW)":"capacity_mw", "그룹설비용량(MW)":"group_capacity_mw",
        })
        turbine["group"] = turbine["group"].ffill().astype(int)
        coords = turbine["coord"].map(dms_to_decimal)
        turbine["lat"] = coords.map(lambda x: x[0]); turbine["lon"] = coords.map(lambda x: x[1])
        group_meta = turbine.groupby("group").agg(
            turbines=("turbine_no", "size"), maker=("maker", "first"), model=("model", "first"),
            hub_height=("hub_height", "first"), rotor_diameter=("rotor_diameter", "first"),
            capacity_mw=("capacity_mw", "sum"), lat=("lat", "mean"), lon=("lon", "mean"),
        )
        display(group_meta)
        assert group_meta.capacity_mw.round(1).tolist() == [21.6, 21.6, 21.0]
        group_meta.reset_index().to_csv(OUT_DIR / "group_metadata.csv", index=False)
        """
    ),
    code(
        """
        #@title 1-2. 발전량 분포·연도 drift·계절·시간 패턴
        CAP = {1: 21600.0, 2: 21600.0, 3: 21000.0}
        labels = pd.read_csv(DATA_DIR / "train/train_labels.csv", encoding="utf-8-sig",
                             parse_dates=["kst_dtm"])
        labels["block_dtm"] = labels["kst_dtm"] - pd.Timedelta(hours=1)
        target_long = labels.melt(id_vars=["kst_dtm", "block_dtm"], var_name="group_col", value_name="target")
        target_long["group"] = target_long.group_col.str.extract(r"(\\d+)").astype(int)
        target_long["cf"] = target_long.target / target_long.group.map(CAP)
        target_long["year"] = target_long.block_dtm.dt.year
        target_long["month"] = target_long.block_dtm.dt.month
        target_long["hour"] = target_long.kst_dtm.dt.hour

        def q(s, p): return s.quantile(p)
        target_summary = (target_long.dropna(subset=["target"]).groupby(["group", "year"])
            .agg(n=("cf","size"), cf_mean=("cf","mean"), cf_std=("cf","std"),
                 p10=("cf",lambda x:q(x,.1)), p50=("cf",lambda x:q(x,.5)), p90=("cf",lambda x:q(x,.9)),
                 below_eval=("cf",lambda x:(x<.10).mean()), rated=("cf",lambda x:(x>=.90).mean()))
            .reset_index())
        display(target_summary.round(4))
        target_summary.to_csv(OUT_DIR / "target_summary.csv", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
        sns.lineplot(data=target_long, x="month", y="cf", hue="group", marker="o", errorbar=None, ax=axes[0])
        axes[0].set_title("Monthly mean capacity factor")
        sns.lineplot(data=target_long, x="hour", y="cf", hue="group", marker="o", errorbar=None, ax=axes[1])
        axes[1].set_title("Hourly mean capacity factor")
        fig.tight_layout(); fig.savefig(OUT_DIR / "target_month_hour.png", dpi=150); plt.show()

        print("핵심 확인:")
        print("- group3 label은 2023년부터만 존재:", target_long[target_long.group.eq(3)].groupby("year").target.count().to_dict())
        print("- 2024년 저발전(<10%cap) 비율:",
              target_summary[target_summary.year.eq(2024)].set_index("group").below_eval.round(3).to_dict())
        """
    ),
    md(
        """
        ### 2. 기상 예보 무결성과 train→test shift

        여기서는 아직 “좋은 피처”를 고르지 않습니다. 먼저 키 중복, 격자 수, 발표 시점, 결측, 상수 컬럼, 2025 분포 이동을 확인합니다. 표준화 평균 차이 `shift_z=(test_mean-train_mean)/train_std`는 모델 성능이 아니라 shift 경보입니다.
        """
    ),
    code(
        """
        #@title 2-1. LDAPS/GFS 원본 감사
        def audit_weather(source, expected_grids):
            train = pd.read_csv(DATA_DIR / f"train/{source}_train.csv", encoding="utf-8-sig")
            test = pd.read_csv(DATA_DIR / f"test/{source}_test.csv", encoding="utf-8-sig")
            for df in (train, test):
                df["forecast_kst_dtm"] = pd.to_datetime(df.forecast_kst_dtm)
                df["data_available_kst_dtm"] = pd.to_datetime(df.data_available_kst_dtm)
            key = ["forecast_kst_dtm", "grid_id"]
            assert train.duplicated(key).sum() == 0 and test.duplicated(key).sum() == 0
            grids_per_time = train.groupby("forecast_kst_dtm").grid_id.nunique()
            assert grids_per_time.min() == grids_per_time.max() == expected_grids

            cutoff = train.forecast_kst_dtm.dt.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
            available_ok = train.data_available_kst_dtm.le(cutoff)
            lag_h = (train.forecast_kst_dtm - train.data_available_kst_dtm).dt.total_seconds() / 3600
            numeric = [c for c in train.select_dtypes("number") if c not in {"grid_id","latitude","longitude"}]
            constants = [c for c in numeric if train[c].nunique(dropna=False) <= 1]
            shifts = []
            for c in numeric:
                sd = train[c].std()
                z = (test[c].mean() - train[c].mean()) / sd if sd and np.isfinite(sd) else np.nan
                shifts.append({"source":source, "feature":c, "shift_z":z,
                               "train_mean":train[c].mean(), "test_mean":test[c].mean(),
                               "train_missing":train[c].isna().mean(), "test_missing":test[c].isna().mean(),
                               "constant":c in constants})
            overview = {
                "source":source, "train_rows":len(train), "test_rows":len(test),
                "train_times":train.forecast_kst_dtm.nunique(), "test_times":test.forecast_kst_dtm.nunique(),
                "grids":train.grid_id.nunique(), "duplicate_keys":int(train.duplicated(key).sum()),
                "available_before_cutoff":float(available_ok.mean()),
                "lag_min_h":lag_h.min(), "lag_median_h":lag_h.median(), "lag_max_h":lag_h.max(),
                "constant_columns":", ".join(constants),
            }
            return overview, pd.DataFrame(shifts)

        overviews, shift_frames = [], []
        for src, ngrid in [("ldaps",16),("gfs",9)]:
            o, s = audit_weather(src, ngrid); overviews.append(o); shift_frames.append(s)
        weather_overview = pd.DataFrame(overviews)
        weather_shift = pd.concat(shift_frames, ignore_index=True)
        display(weather_overview)
        display(weather_shift.reindex(weather_shift.shift_z.abs().sort_values(ascending=False).index)
                .groupby("source").head(10).round(4))
        weather_overview.to_csv(OUT_DIR / "weather_overview.csv", index=False)
        weather_shift.to_csv(OUT_DIR / "weather_train_test_shift.csv", index=False)
        """
    ),
    code(
        """
        #@title 2-2. 외부 데이터가 있으면 coverage만 감사 (모델 입력 생성 없음)
        external_dir = DATA_DIR / "external"
        external_rows = []
        if external_dir.exists():
            for p in sorted(external_dir.iterdir()):
                if p.is_file(): external_rows.append({"file":p.name, "size_mb":p.stat().st_size/2**20})
        display(pd.DataFrame(external_rows))

        jma_path = external_dir / "jma_prevruns.csv"
        if jma_path.exists():
            jma = pd.read_csv(jma_path, encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"])
            jma_cov = (jma.assign(year=jma.forecast_kst_dtm.dt.year)
                       .groupby(["model","year"])["wind_speed_10m_d2"]
                       .agg(rows="size", available="count").reset_index())
            jma_cov["coverage"] = jma_cov.available / jma_cov.rows
            display(jma_cov)
            assert not any(c.endswith("_d1") for c in [
                "wind_speed_10m_d2", "wind_direction_10m_d2", "temperature_2m_d2", "surface_pressure_d2"
            ])
            jma_cov.to_csv(OUT_DIR / "jma_coverage.csv", index=False)
        else:
            print("JMA 파일 없음: 공식 데이터 EDA는 계속 진행합니다. 외부 입력 비교는 파일을 받은 뒤 재실행하세요.")
        """
    ),
    md(
        """
        ### 3. 격자와 발전단지의 공간 관계

        LDAPS 최근접 격자는 그룹별로 다르지만, GFS는 세 그룹 모두 같은 격자에 가장 가깝습니다. 따라서 GFS에서 그룹 차이는 좌표보다 터빈/지형/가용성에서 생기고, LDAPS 다격자 구조는 국지 지형 신호를 담을 여지가 있습니다.
        """
    ),
    code(
        """
        #@title 3-1. 최근접 격자·거리·지도
        def haversine(lat, lon, grid_lat, grid_lon):
            p1, p2 = np.radians(lat), np.radians(np.asarray(grid_lat))
            dp = p2 - p1; dl = np.radians(np.asarray(grid_lon) - lon)
            return 6371 * 2 * np.arcsin(np.sqrt(np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2))

        grid_tables, nearest_rows = {}, []
        for source in ["ldaps", "gfs"]:
            grid = (pd.read_csv(DATA_DIR / f"train/{source}_train.csv", encoding="utf-8-sig",
                                usecols=["grid_id","latitude","longitude"])
                    .drop_duplicates().sort_values("grid_id"))
            grid_tables[source] = grid
            for group, row in group_meta.iterrows():
                dist = pd.Series(haversine(row.lat, row.lon, grid.latitude, grid.longitude), index=grid.grid_id)
                for rank, (gid, km) in enumerate(dist.sort_values().head(4).items(), 1):
                    nearest_rows.append({"source":source,"group":group,"rank":rank,"grid_id":gid,"distance_km":km})
        nearest = pd.DataFrame(nearest_rows)
        display(nearest)
        nearest.to_csv(OUT_DIR / "nearest_grids.csv", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, source in zip(axes, ["ldaps","gfs"]):
            g = grid_tables[source]
            ax.scatter(g.longitude, g.latitude, s=55, label=f"{source.upper()} grid")
            for _, r in g.iterrows(): ax.text(r.longitude, r.latitude, str(int(r.grid_id)), fontsize=8)
            ax.scatter(turbine.lon, turbine.lat, c=turbine.group, cmap="viridis", marker="x", s=45, label="turbines")
            ax.scatter(group_meta.lon, group_meta.lat, c="red", marker="*", s=130, label="group centroid")
            ax.set_title(source.upper()); ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend()
        fig.tight_layout(); fig.savefig(OUT_DIR / "grid_geometry.png", dpi=150); plt.show()
        """
    ),
    md(
        """
        ### 4. 물리 피처를 직접 만들어 타깃과 관계 확인

        상관계수는 채택 기준이 아닙니다. 포화형 power curve 때문에 `v³`의 선형 상관은 오히려 낮을 수 있습니다. 여기서는 **신호 존재와 중복 구조**를 확인합니다.

        현재 파이프라인에 이미 있는 것: 공기밀도, 풍속 세제곱, 밀도×풍속³, gust factor, 850 hPa 풍속/시어, 시간 rolling/ramp/day 통계, 공간 mean/std/range, LDAPS/GFS raw grid deviations. 아래 후보 중 이들과 같은 것은 새 실험이 아닙니다.
        """
    ),
    code(
        """
        #@title 4-1. 그룹별 최근접 풍속·hub-height·veer·density 후보
        gfs_cols = ["forecast_kst_dtm","grid_id","heightAboveGround_10_10u","heightAboveGround_10_10v",
                    "heightAboveGround_80_u","heightAboveGround_80_v","heightAboveGround_100_100u",
                    "heightAboveGround_100_100v","surface_0_gust","surface_0_sp","heightAboveGround_2_2t",
                    "isobaricInhPa_850_u","isobaricInhPa_850_v"]
        ld_cols = ["forecast_kst_dtm","grid_id","heightAboveGround_10_10u","heightAboveGround_10_10v",
                   "heightAboveGround_50_50MUmax","heightAboveGround_50_50MUmin",
                   "heightAboveGround_50_50MVmax","heightAboveGround_50_50MVmin"]
        gf = pd.read_csv(DATA_DIR / "train/gfs_train.csv", encoding="utf-8-sig", usecols=gfs_cols,
                         parse_dates=["forecast_kst_dtm"])
        ld = pd.read_csv(DATA_DIR / "train/ldaps_train.csv", encoding="utf-8-sig", usecols=ld_cols,
                         parse_dates=["forecast_kst_dtm"])

        feature_frames = []
        for group in [1,2,3]:
            ggid = int(nearest.query("source=='gfs' and group==@group and rank==1").grid_id.iloc[0])
            lgid = int(nearest.query("source=='ldaps' and group==@group and rank==1").grid_id.iloc[0])
            g = gf[gf.grid_id.eq(ggid)].copy(); l = ld[ld.grid_id.eq(lgid)].copy()
            for height, u, v in [
                (10,"heightAboveGround_10_10u","heightAboveGround_10_10v"),
                (80,"heightAboveGround_80_u","heightAboveGround_80_v"),
                (100,"heightAboveGround_100_100u","heightAboveGround_100_100v")]:
                g[f"gfs_ws{height}"] = np.hypot(g[u],g[v]); g[f"gfs_dir{height}"] = np.arctan2(g[v],g[u])
            g["gfs_ws850"] = np.hypot(g.isobaricInhPa_850_u,g.isobaricInhPa_850_v)
            g["density"] = g.surface_0_sp / (287.05 * g.heightAboveGround_2_2t)
            # 80→100m shear exponent를 안정화하고 117m까지만 짧게 외삽.
            g["alpha80_100"] = (np.log((g.gfs_ws100+.5)/(g.gfs_ws80+.5)) / np.log(100/80)).clip(-1,1)
            g["gfs_ws117"] = g.gfs_ws100 * (117/100) ** g.alpha80_100
            g["power_density117"] = .5 * g.density * g.gfs_ws117 ** 3
            g["veer100_10_cos"] = np.cos(g.gfs_dir100 - g.gfs_dir10)
            g["veer100_10_sin"] = np.sin(g.gfs_dir100 - g.gfs_dir10)
            g["gust_factor"] = g.surface_0_gust / (g.gfs_ws10 + .5)

            l["ldaps_ws10"] = np.hypot(l.heightAboveGround_10_10u,l.heightAboveGround_10_10v)
            l["ldaps_dir10"] = np.arctan2(l.heightAboveGround_10_10v,l.heightAboveGround_10_10u)
            l["ldaps_ws50max"] = np.hypot(l.heightAboveGround_50_50MUmax,l.heightAboveGround_50_50MVmax)
            l["ldaps_ws50min"] = np.hypot(l.heightAboveGround_50_50MUmin,l.heightAboveGround_50_50MVmin)
            l["ldaps_gust_range"] = l.ldaps_ws50max - l.ldaps_ws50min

            useg = ["forecast_kst_dtm","gfs_ws10","gfs_ws80","gfs_ws100","gfs_ws117","gfs_ws850",
                    "density","power_density117","alpha80_100","veer100_10_cos","veer100_10_sin","gust_factor"]
            usel = ["forecast_kst_dtm","ldaps_ws10","ldaps_dir10","ldaps_gust_range"]
            z = g[useg].merge(l[usel], on="forecast_kst_dtm")
            z["cross_model_ws_gap"] = z.gfs_ws100 - z.ldaps_ws10
            z["group"] = group
            z = z.merge(labels[["kst_dtm",f"kpx_group_{group}"]], left_on="forecast_kst_dtm", right_on="kst_dtm")
            z["cf"] = z[f"kpx_group_{group}"] / CAP[group]
            feature_frames.append(z.drop(columns=["kst_dtm",f"kpx_group_{group}"]))
        physics = pd.concat(feature_frames, ignore_index=True).dropna(subset=["cf"])

        fcols = [c for c in physics.columns if c not in {"forecast_kst_dtm","group","cf"}]
        corr_rows = []
        for group, d in physics.groupby("group"):
            for c in fcols:
                corr_rows.append({"group":group,"feature":c,
                                  "pearson":d[c].corr(d.cf), "spearman":d[c].rank().corr(d.cf.rank())})
        corr = pd.DataFrame(corr_rows)
        display(corr.sort_values(["group","spearman"], ascending=[True,False]).round(4))
        corr.to_csv(OUT_DIR / "physics_feature_correlations.csv", index=False)

        physics["ws850_bin"] = pd.cut(physics.gfs_ws850, bins=np.arange(0,31,2), include_lowest=True)
        power_curve = (physics.groupby(["group","ws850_bin"], observed=True).cf
                       .agg(["mean","median","size"]).reset_index())
        display(power_curve.head(20))
        power_curve.to_csv(OUT_DIR / "ws850_power_curve.csv", index=False)
        """
    ),
    md(
        """
        ### 5. SCADA 품질과 가용성

        SCADA는 평가 시점에 존재하지 않으므로 추론 피처로 사용하면 안 됩니다. 사용 목적은 학습 라벨의 터빈 정지/정비 시간을 찾아 sample weight를 조절하거나 정상 power curve를 진단하는 것입니다.

        특히 VESTAS power에는 음수 상태 코드와 수천만 단위 이상치가 존재하므로 raw 합계·평균을 만들면 안 됩니다. 현재 `cln03`처럼 “바람이 있는데 5/6 슬롯 이상 power≤1”인지를 쓰는 방식은 크기 이상치보다 안전합니다.
        """
    ),
    code(
        """
        #@title 5-1. SCADA 이상값·결측·시간별 가용률
        scada_quality, availability_parts = [], []
        specs = [("vestas", range(1,7),1),("vestas",range(7,13),2),("unison",range(1,6),3)]
        loaded = {}
        for maker in ["vestas","unison"]:
            d = pd.read_csv(DATA_DIR / f"train/scada_{maker}_train.csv", encoding="utf-8-sig",
                            parse_dates=["kst_dtm"])
            loaded[maker] = d
            pcols = [c for c in d if "power_kw10m" in c]
            wcols = [c for c in d if c.endswith("_ws")]
            values = d[pcols].to_numpy(float)
            scada_quality.append({
                "maker":maker,"rows":len(d),"start":d.kst_dtm.min(),"end":d.kst_dtm.max(),
                "power_missing":np.isnan(values).mean(),"ws_missing":d[wcols].isna().mean().mean(),
                "power_negative":np.nanmean(values<0),"power_zero":np.nanmean(values==0),
                "power_gt1000":np.nanmean(values>1000),"power_min":np.nanmin(values),"power_max":np.nanmax(values),
            })

        for maker, turbines, group in specs:
            d = loaded[maker].copy(); d["hour"] = d.kst_dtm.dt.floor("h")
            pcols = [f"{maker}_wtg{i:02d}_power_kw10m" for i in turbines]
            wcols = [f"{maker}_wtg{i:02d}_ws" for i in turbines]
            median_ws = d[wcols].median(axis=1)
            hourly_dead, hourly_slots = [], []
            for pc, wc in zip(pcols,wcols):
                ws_eff = d[wc].fillna(median_ws)
                dead = (~(d[pc] > 1).fillna(False)) & (ws_eff >= 5)
                hourly_dead.append(dead.groupby(d.hour).sum())
                hourly_slots.append(d.groupby("hour")[pc].size())
            unavailable = sum((dead >= np.maximum(slots-1,5)).astype(int)
                              for dead,slots in zip(hourly_dead,hourly_slots))
            av = pd.DataFrame({"forecast_kst_dtm":unavailable.index,"group":group,
                               "avail_strict":1-unavailable.to_numpy()/len(list(turbines))})
            availability_parts.append(av)
        availability = pd.concat(availability_parts, ignore_index=True)
        availability_summary = availability.groupby("group").avail_strict.agg(
            hours="size", mean="mean", full=lambda x:(x>=.999).mean(), below90=lambda x:(x<.9).mean(),
            below70=lambda x:(x<.7).mean()).reset_index()
        display(pd.DataFrame(scada_quality))
        display(availability_summary.round(4))
        pd.DataFrame(scada_quality).to_csv(OUT_DIR / "scada_quality.csv", index=False)
        availability_summary.to_csv(OUT_DIR / "scada_availability_summary.csv", index=False)
        availability.to_parquet(OUT_DIR / "scada_availability_hourly.parquet", index=False)
        """
    ),
    md(
        """
        ### 6. 현재 입력과 새 후보의 구분

        같은 공식을 이름만 바꿔 다시 넣지 않도록 feature registry를 먼저 고정합니다. 우선순위 `P0`은 제거/감사, `P1`은 새 정보가 명확한 후보, `P2`는 P1 결과 후 검토입니다.
        """
    ),
    code(
        """
        #@title 6-1. Feature registry와 첫 ablation 후보
        registry = pd.DataFrame([
            ["constant_exact_duplicate_drop","상수·exact duplicate 제거","일부 존재","P0","정보량 0; 재학습 없이 먼저 제거","safe"],
            ["density_ws_cube","rho, ws^3, rho*ws^3","이미 사용","closed","v10P에 포함","safe"],
            ["gust_shear_ws850","gust factor/range, ws850, shear","이미 사용","closed","v10P에 포함","safe"],
            ["temporal_context","rolling/ramp/day aggregate","이미 사용","closed","v10T에 포함; cutoff assert 유지","cutoff-check"],
            ["spatial_mean_std_rawdev","grid mean/std/range + raw deviations","이미 사용","closed","v10S+SPD에 포함","safe"],
            ["hub117_vector","80/100m vector에서 117m 풍속·방향 안정 외삽","미사용 가능","P1","실제 hub height에 정렬; alpha clip 필요","safe"],
            ["directional_veer","10↔100m 방향차 sin/cos","미사용","P1","속도 shear와 다른 대기 안정/지형 신호","safe"],
            ["group_direction_sector","group×풍향 sector 또는 circular interaction","부분 사용","P1","산 능선·wake 방향 의존성","safe"],
            ["along_wind_gradient","LDAPS 공간 gradient를 풍향 축으로 투영","미사용","P1","raw grid deviation을 물리적으로 축약","safe"],
            ["continuous_availability_weight","SCADA 기반 이산 cln03→연속 confidence","부분 사용","P2","라벨 오염 강도를 더 세밀하게 반영","train-only"],
            ["cross_model_uncertainty","모델 간 풍속/방향 disagreement 정규화","부분 사용","P2","기존 단순 gap과 다른 방향 일치도","publication-check"],
            ["raw_scada_inference","SCADA power/ws 직접 입력","사용 금지","reject","test에 없고 운영정보 leakage","forbidden"],
        ], columns=["family","formula_or_action","current_status","priority","reason","leakage"])
        display(registry)
        registry.to_csv(OUT_DIR / "feature_registry.csv", index=False)

        first_ablation = registry.query("priority in ['P0','P1']")[["priority","family","reason","leakage"]]
        print("\\n다음 실험 순서")
        display(first_ablation)
        """
    ),
    md(
        """
        ### 이 노트북을 본 뒤 결정할 것

        첫 실험은 대규모 “feature engineering 묶음”이 아니라 다음 순서로 진행합니다.

        1. **F0 제거 audit:** 상수·exact duplicate 제거. 점수가 같아야 정상입니다.
        2. **F1 hub/veer:** `hub117 vector + 10↔100m veer sin/cos`만 추가합니다.
        3. **F2 terrain-direction:** 그룹별 circular direction interaction만 추가합니다.
        4. **F3 along-wind spatial gradient:** LDAPS 다격자를 풍향축/수직축 gradient로 축약합니다.

        각 실험은 exp-132 `C0-ES`와 같은 split, q60 loss, patience, seed, GBM cache, no-snap 후처리를 사용합니다. 한 번에 한 family만 바꿔야 개선 원인을 설명할 수 있습니다.
        """
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"name": OUT.name, "provenance": []},
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT)
