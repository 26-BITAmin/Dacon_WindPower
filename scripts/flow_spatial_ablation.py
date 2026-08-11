"""Build the Colab notebook for the flow-aligned spatial feature ablation."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external_champion_train.ipynb"
OUT = ROOT / "flow_spatial_ablation.ipynb"


def lines(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def markdown(text: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": lines(text),
    }


def code(text: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": lines(text),
    }


def replace_required(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"source patch not found: {label}")
    return text.replace(old, new)


source_nb = json.loads(SOURCE.read_text(encoding="utf-8"))
source_cells = source_nb["cells"]
if len(source_cells) < 15:
    raise RuntimeError(f"unexpected source cell count: {len(source_cells)}")


setup = "".join(source_cells[1]["source"])
setup = replace_required(
    setup,
    '''    DATA_DIR = Path("/content/drive/MyDrive/BARAM2026/data")
    RUN_ROOT = Path("/content/drive/MyDrive/BARAM2026/hyejin_external_champion")
else:
    DATA_DIR = Path.cwd() / "data"
    RUN_ROOT = Path.cwd() / "runs" / "hyejin_external_champion"
''',
    '''    DATA_DIR = Path("/content/drive/MyDrive/BARAM2026/data")
    BASE_RUN_ROOT = Path("/content/drive/MyDrive/BARAM2026/hyejin_external_champion")
    RUN_ROOT = Path("/content/drive/MyDrive/BARAM2026/hyejin_flow_spatial")
else:
    DATA_DIR = Path.cwd() / "data"
    BASE_RUN_ROOT = Path.cwd() / "runs" / "hyejin_external_champion"
    RUN_ROOT = Path.cwd() / "runs" / "hyejin_flow_spatial"
''',
    "run roots",
)
setup = replace_required(
    setup,
    '''EXTERNAL_DIR = DATA_DIR / "external"
CACHE_DIR = RUN_ROOT / "cache"
for path in [RUN_ROOT, EXTERNAL_DIR, CACHE_DIR]:
''',
    '''EXTERNAL_DIR = DATA_DIR / "external"
BASE_CACHE_DIR = BASE_RUN_ROOT / "cache"
CACHE_DIR = RUN_ROOT / "cache"
for path in [RUN_ROOT, EXTERNAL_DIR, BASE_CACHE_DIR, CACHE_DIR]:
''',
    "cache roots",
)
setup = replace_required(
    setup,
    '''FINAL_GBM_SEEDS = 3
FINAL_DL_SEEDS = 3
''',
    '''FINAL_GBM_SEEDS = 3
FINAL_DL_SEEDS = 3
FLOW_VERSION = "v1"
RUN_FINAL_IF_PASS = True
PASS_CV_DELTA = 0.0005
PASS_MIN_POSITIVE_FOLDS = 3
PASS_MIN_SUBMETRIC_DELTA = -0.0005
''',
    "flow settings",
)
setup = setup.replace(
    'print("output  :", RUN_ROOT)',
    'print("base cache:", BASE_CACHE_DIR)\nprint("output    :", RUN_ROOT)',
)


cells = [
    markdown(
        r'''
## BARAM 2026 - Flow-aligned spatial feature ablation

이 노트북은 공식 LDAPS/GFS 격자 바람장을 평균 풍향 축으로 투영한 새 피처를 검증합니다.

- `C0`: 현재 269개 GBM 피처
- `F2_FLOW`: C0 + flow-aligned 공간 피처 12개
- DL, 외부 데이터, seed, GBM:DL 0.65:0.35, no-snap, alpha 0.40, clip 0.16은 고정
- 기존 외부 데이터와 DL OOF 캐시는 Drive에서 재사용
- primary gate를 통과할 때만 최종 submission 생성

핵심 판정은 `summary.csv`, `fold_scores.csv`, `group_scores.csv`, `gate_decision.json`에 저장됩니다.
''',
        "flow_intro",
    ),
    code(setup, "flow_setup"),
]


for source_index in range(2, 12):
    cell = deepcopy(source_cells[source_index])
    cell["id"] = f"flow_base_{source_index:02d}"
    cell["outputs"] = []
    cell["execution_count"] = None
    text = "".join(cell.get("source", []))
    if source_index == 5:
        text = replace_required(
            text,
            'SPD_PATH = CACHE_DIR / "spatial_spd.parquet"',
            'SPD_PATH = BASE_CACHE_DIR / "spatial_spd.parquet"',
            "shared SPD cache",
        )
    cell["source"] = lines(text)
    cells.append(cell)


flow_features = r'''
#@title 11. Flow-aligned 공간 피처 생성
FLOW_CACHE_PATH = CACHE_DIR / f"flow_spatial_features_{FLOW_VERSION}.parquet"
FLOW_K_NEAREST = 4

FLOW_SOURCES = {
    "L": {
        "train": TRAIN_DIR / "ldaps_train.csv",
        "test": TEST_DIR / "ldaps_test.csv",
        "u": "heightAboveGround_10_10u",
        "v": "heightAboveGround_10_10v",
    },
    "G": {
        "train": TRAIN_DIR / "gfs_train.csv",
        "test": TEST_DIR / "gfs_test.csv",
        "u": "heightAboveGround_100_100u",
        "v": "heightAboveGround_100_100v",
    },
}

def _fill_by_row(values):
    values = values.astype(np.float64, copy=True)
    row_mean = np.nanmean(values, axis=1)
    missing = np.isnan(values)
    values[missing] = np.take(row_mean, np.where(missing)[0])
    return values

def _flow_source_frame(info, source_tag, spec, k_nearest=4):
    usecols = ["forecast_kst_dtm", "grid_id", "latitude", "longitude", spec["u"], spec["v"]]
    raw = pd.concat(
        [pd.read_csv(spec["train"], encoding="utf-8-sig", usecols=usecols),
         pd.read_csv(spec["test"], encoding="utf-8-sig", usecols=usecols)],
        ignore_index=True,
    )
    raw["forecast_kst_dtm"] = pd.to_datetime(raw["forecast_kst_dtm"])
    grids = raw[["grid_id", "latitude", "longitude"]].drop_duplicates().set_index("grid_id")
    centroids = info.groupby("kpx_group")[["lat", "lon"]].mean()
    frames = []

    for group in [1, 2, 3]:
        lat0 = float(centroids.loc[group, "lat"])
        lon0 = float(centroids.loc[group, "lon"])
        distance = pd.Series(
            haversine(lat0, lon0, grids["latitude"], grids["longitude"]),
            index=grids.index,
        )
        selected = distance.nsmallest(min(k_nearest, len(distance))).index.tolist()
        grid_xy = grids.loc[selected]
        x_km = (grid_xy["longitude"].to_numpy() - lon0) * 111.0 * np.cos(np.deg2rad(lat0))
        y_km = (grid_xy["latitude"].to_numpy() - lat0) * 111.0
        design = np.column_stack([x_km, y_km, np.ones(len(selected))])
        design_pinv = np.linalg.pinv(design)

        sub = raw.loc[raw["grid_id"].isin(selected)]
        u = sub.pivot_table(index="forecast_kst_dtm", columns="grid_id", values=spec["u"], aggfunc="mean")
        v = sub.pivot_table(index="forecast_kst_dtm", columns="grid_id", values=spec["v"], aggfunc="mean")
        master_index = u.index.union(v.index).sort_values()
        u = u.reindex(index=master_index, columns=selected)
        v = v.reindex(index=master_index, columns=selected)
        U = _fill_by_row(u.to_numpy())
        V = _fill_by_row(v.to_numpy())

        coef_u = U @ design_pinv.T
        coef_v = V @ design_pinv.T
        du_dx, du_dy = coef_u[:, 0], coef_u[:, 1]
        dv_dx, dv_dy = coef_v[:, 0], coef_v[:, 1]

        mean_u = np.nanmean(U, axis=1)
        mean_v = np.nanmean(V, axis=1)
        mean_ws = np.hypot(mean_u, mean_v)
        ex = mean_u / np.maximum(mean_ws, 0.2)
        ey = mean_v / np.maximum(mean_ws, 0.2)
        nx, ny = -ey, ex

        dws_dx = (mean_u * du_dx + mean_v * dv_dx) / np.maximum(mean_ws, 0.2)
        dws_dy = (mean_u * du_dy + mean_v * dv_dy) / np.maximum(mean_ws, 0.2)
        along_grad = ex * dws_dx + ey * dws_dy
        cross_grad = nx * dws_dx + ny * dws_dy
        divergence = du_dx + dv_dy
        vorticity = dv_dx - du_dy

        node_ws = np.hypot(U, V)
        unit_u = U / np.maximum(node_ws, 0.2)
        unit_v = V / np.maximum(node_ws, 0.2)
        resultant = np.hypot(np.nanmean(unit_u, axis=1), np.nanmean(unit_v, axis=1))
        direction_spread = 1.0 - np.clip(resultant, 0.0, 1.0)
        along_node = U * ex[:, None] + V * ey[:, None]
        along_range = np.nanmax(along_node, axis=1) - np.nanmin(along_node, axis=1)

        prefix = f"flow_{source_tag}"
        frame = pd.DataFrame({
            "forecast_kst_dtm": master_index,
            "kpx_group": group,
            f"{prefix}_along_grad": along_grad,
            f"{prefix}_cross_grad": cross_grad,
            f"{prefix}_divergence": divergence,
            f"{prefix}_vorticity": vorticity,
            f"{prefix}_direction_spread": direction_spread,
            f"{prefix}_along_range": along_range,
        })
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)

def build_flow_features(info, force=False):
    if FLOW_CACHE_PATH.exists() and not force:
        flow = pd.read_parquet(FLOW_CACHE_PATH)
        print("flow 피처 캐시 로드:", flow.shape)
    else:
        started = time.time()
        ld_flow = _flow_source_frame(info, "L", FLOW_SOURCES["L"], FLOW_K_NEAREST)
        gf_flow = _flow_source_frame(info, "G", FLOW_SOURCES["G"], FLOW_K_NEAREST)
        flow = ld_flow.merge(gf_flow, on=["forecast_kst_dtm", "kpx_group"], how="outer")
        flow.to_parquet(FLOW_CACHE_PATH, index=False)
        print(f"flow 피처 생성 완료: {flow.shape} ({time.time()-started:.1f}s)")
    assert not flow.duplicated(["forecast_kst_dtm", "kpx_group"]).any()
    flow_cols = [c for c in flow.columns if c.startswith("flow_")]
    constant = [c for c in flow_cols if flow[c].nunique(dropna=True) <= 1]
    assert not constant, f"상수 flow 피처: {constant}"
    return flow, flow_cols
'''
cells.append(code(flow_features, "flow_feature_builder"))


source_data = "".join(source_cells[12]["source"])
weak_start = source_data.index("WEAK_FEATURES = [")
weak_end = source_data.index("cols_gspd = [", weak_start)
weak_block = source_data[weak_start:weak_end].strip()


data_prep = rf'''
#@title 12. C0/F2 공통 데이터와 기존 캐시 로드
_, _, _, sample_submission, info = build_datasets()
train_df_tps, test_df_tps, cols_tps, _ = build_v11(info)

spd, spd_cols = build_spd()
train_df_tps = train_df_tps.merge(spd, on="forecast_kst_dtm", how="left")
test_df_tps = test_df_tps.merge(spd, on="forecast_kst_dtm", how="left")

jma_raw = fetch_jma_prevruns(info)
jma = build_jma_frame(jma_raw)
train_df_tps = train_df_tps.merge(jma, on=["forecast_kst_dtm", "kpx_group"], how="left")
test_df_tps = test_df_tps.merge(jma, on=["forecast_kst_dtm", "kpx_group"], how="left")
for frame in [train_df_tps, test_df_tps]:
    frame["jgsm_dis"] = (frame["jgsm_ws"] - frame["ldaps_ws_10m"]).abs()
    frame["jmsm_dis"] = (frame["jmsm_ws"] - frame["ldaps_ws_10m"]).abs()

cols_gspd = cols_tps + JG_COLS + JM_COLS + spd_cols
ec_raw = fetch_ecmwf_prevruns(info)
ec = build_ec_frame(ec_raw)
train_df_tps = train_df_tps.merge(ec, on=["forecast_kst_dtm", "kpx_group"], how="left")
test_df_tps = test_df_tps.merge(ec, on=["forecast_kst_dtm", "kpx_group"], how="left")
for frame in [train_df_tps, test_df_tps]:
    frame["ec_dis"] = (frame["ec_ws"] - frame["ldaps_ws_10m"]).abs()

cols_c0 = cols_gspd + EC_COLS

{weak_block}

cols_gspd = [c for c in cols_gspd if c not in WEAK_FEATURES]
cols_c0 = [c for c in cols_c0 if c not in WEAK_FEATURES]

flow_frame, flow_cols = build_flow_features(info)
train_df_tps = train_df_tps.merge(flow_frame, on=["forecast_kst_dtm", "kpx_group"], how="left")
test_df_tps = test_df_tps.merge(flow_frame, on=["forecast_kst_dtm", "kpx_group"], how="left")
cols_f2 = cols_c0 + flow_cols

assert len(cols_c0) == 269, f"C0 피처 수가 예상과 다름: {{len(cols_c0)}}"
assert len(flow_cols) == 12, f"flow 피처 수가 예상과 다름: {{len(flow_cols)}}"
assert len(cols_f2) == 281
assert set(flow_cols).isdisjoint(cols_c0)

flow_summary = pd.DataFrame({{
    "feature": flow_cols,
    "train_coverage": [float(train_df_tps[c].notna().mean()) for c in flow_cols],
    "test_coverage": [float(test_df_tps[c].notna().mean()) for c in flow_cols],
    "train_mean": [float(train_df_tps[c].mean()) for c in flow_cols],
    "train_std": [float(train_df_tps[c].std()) for c in flow_cols],
}})
flow_summary.to_csv(RUN_ROOT / "flow_feature_summary.csv", index=False)
display(flow_summary.round(6))

sc = load_scada()
av = compute_availability(sc)
av["forecast_kst_dtm"] = pd.to_datetime(av["forecast_kst_dtm"])

base_gbm_path = BASE_CACHE_DIR / "external_champion_gbm_v1.parquet"
base_bias_path = BASE_CACHE_DIR / "external_champion_bias_v1.pkl"
base_dl_path = BASE_CACHE_DIR / "external_champion_dl_v1.parquet"
missing_base = [str(p) for p in [base_gbm_path, base_bias_path, base_dl_path] if not p.exists()]
assert not missing_base, (
    "먼저 external_champion_train.ipynb를 완료해야 합니다. 누락 캐시:\n" + "\n".join(missing_base)
)

gbm_c0 = pd.read_parquet(base_gbm_path)
with open(base_bias_path, "rb") as stream:
    bias_c0 = pickle.load(stream)
dl_cache = pd.read_parquet(base_dl_path)
print("C0 GBM cache:", gbm_c0.shape)
print("fixed DL cache:", dl_cache.shape)

run_config = {{
    "experiment": "F2_FLOW",
    "flow_version": FLOW_VERSION,
    "control_feature_count": len(cols_c0),
    "candidate_feature_count": len(cols_f2),
    "flow_features": flow_cols,
    "fixed_components": {{"dl_cache": str(base_dl_path), "w_gbm": 0.65, "snap": False,
                         "rolling_alpha": 0.40, "clip_lo": 0.16}},
    "gate": {{"cv_delta": PASS_CV_DELTA, "positive_folds": PASS_MIN_POSITIVE_FOLDS,
             "min_submetric_delta": PASS_MIN_SUBMETRIC_DELTA, "hold_nonnegative": True}},
}}
(RUN_ROOT / "run_config.json").write_text(
    json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"C0 {{len(cols_c0)}} features | F2 {{len(cols_f2)}} features (+{{len(flow_cols)}})")
'''
cells.append(code(data_prep, "flow_data_prep"))


gate = r'''
#@title 13. C0 대 F2_FLOW GBM ablation과 gate
def safe_lgb_predict(X_train, y_train, sample_weight, X_eval, n_estimators, jobs, seed,
                     objective="mae", label="lgb"):
    params = lgbm_params(n_estimators, jobs, seed, objective=objective)
    try:
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    except lgb.basic.LightGBMError as error:
        print(f"[{label}] GPU LightGBM 실패, CPU로 한 번 재시도: {str(error).splitlines()[0]}")
        params.update(device="cpu", force_col_wise=True)
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict(X_eval)

def estimate_bias_data_safe(train_df, weather_cols, train_end, n_estimators=500, jobs=2):
    result = {}
    for group in [1, 2, 3]:
        capacity = CAPACITY_KWH[f"kpx_group_{group}"]
        group_data = train_df.loc[
            (train_df["kpx_group"] == group)
            & (train_df["forecast_kst_dtm"] <= train_end)
            & train_df["target"].notna()
        ].copy()
        if group_data.empty:
            continue
        first_date = group_data["forecast_kst_dtm"].min()
        available_days = max((train_end - first_date).days, 1)
        holdout_days = min(int(available_days * 0.25), 180)
        if available_days - holdout_days < 90:
            holdout_days = max(0, available_days - 90)
        split_time = train_end - pd.Timedelta(days=holdout_days)
        fit_group = group_data.loc[group_data["forecast_kst_dtm"] <= split_time]
        check_group = group_data.loc[group_data["forecast_kst_dtm"] > split_time].copy()
        if fit_group.empty or check_group.empty:
            continue
        X_fit, X_check = prepare_matrices(fit_group, check_group, weather_cols)
        weights = make_sample_weight(fit_group["target"].to_numpy(), capacity, 0.05)
        prediction = safe_lgb_predict(
            X_fit, fit_group["target"], weights, X_check,
            n_estimators, jobs, SEED, label=f"bias-g{group}",
        )
        check_group["signed_error_rate"] = (prediction - check_group["target"]) / capacity
        check_group = check_group.loc[check_group["target"] >= capacity * 0.10]
        if not check_group.empty:
            result[group] = check_group[["ldaps_ws_10m", "signed_error_rate"]]
    return result

def cache_flow_gbm(train_df, cols, availability, jobs=2, seeds=3, checkpoint=None):
    rows, bias_maps, done = [], {}, set()
    if checkpoint is not None and checkpoint.exists():
        with open(checkpoint, "rb") as stream:
            state = pickle.load(stream)
        rows = state["rows"]
        bias_maps = state["bias_maps"]
        done = set(bias_maps)
        print(f"F2 체크포인트 로드: {len(done)}개 fold 완료")

    for fold_name, fold in tqdm(FOLD_LIST, desc="F2 GBM fold 학습"):
        if fold_name in done:
            continue
        started = time.time()
        train_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
        valid_all = train_df.loc[
            (train_df["forecast_kst_dtm"] >= fold["valid_start"])
            & (train_df["forecast_kst_dtm"] <= fold["valid_end"])
            & train_df["target"].notna()
        ]
        bias_data = estimate_bias_data_safe(train_df, cols, fold["train_end"], 500, jobs)
        bias_maps[fold_name] = bias_map_from_data(
            bias_data, {group: COARSE_BINS for group in bias_data}
        )

        for group in [1, 2, 3]:
            capacity = CAPACITY_KWH[f"kpx_group_{group}"]
            train_group = train_all.loc[
                (train_all["kpx_group"] == group) & train_all["target"].notna()
            ].copy()
            valid_group = valid_all.loc[valid_all["kpx_group"] == group].sort_values(
                "forecast_kst_dtm"
            ).copy()
            train_group = train_group.merge(
                availability.loc[availability["group"] == group,
                                 ["forecast_kst_dtm", "avail_strict"]],
                on="forecast_kst_dtm", how="left",
            )
            train_group["avail_strict"] = train_group["avail_strict"].fillna(1.0)
            X_train, X_valid = prepare_matrices(train_group, valid_group, cols)
            y_train = train_group["target"]
            weights = make_sample_weight_scada(
                y_train.to_numpy(), capacity, train_group["avail_strict"]
            )

            pred_lgb = np.mean([
                safe_lgb_predict(
                    X_train, y_train, weights, X_valid, 500, jobs, seed,
                    label=f"{fold_name}-g{group}-s{seed}",
                ) for seed in range(seeds)
            ], axis=0)
            pred_xgb = np.mean([
                xgbm.XGBRegressor(
                    n_estimators=500, max_depth=6, learning_rate=0.03,
                    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                    random_state=seed, verbosity=0, tree_method="hist", device="cuda",
                    objective="reg:absoluteerror",
                ).fit(X_train, y_train, sample_weight=weights).predict(X_valid)
                for seed in range(seeds)
            ], axis=0)
            rows.append(pd.DataFrame({
                "fold": fold_name,
                "group": group,
                "forecast_kst_dtm": valid_group["forecast_kst_dtm"].to_numpy(),
                "target": valid_group["target"].to_numpy(),
                "ws": valid_group["ldaps_ws_10m"].to_numpy(),
                "gbm_cf": (0.5 * pred_lgb + 0.5 * pred_xgb) / capacity,
            }))
        print(f"  [F2:{fold_name}] {time.time()-started:.0f}s", flush=True)
        if checkpoint is not None:
            with open(checkpoint, "wb") as stream:
                pickle.dump({"rows": rows, "bias_maps": bias_maps}, stream)
    return pd.concat(rows, ignore_index=True), bias_maps

f2_cache_path = CACHE_DIR / f"flow_f2_gbm_{FLOW_VERSION}.parquet"
f2_bias_path = CACHE_DIR / f"flow_f2_bias_{FLOW_VERSION}.pkl"
f2_checkpoint = CACHE_DIR / f"flow_f2_gbm_checkpoint_{FLOW_VERSION}.pkl"
if f2_cache_path.exists() and f2_bias_path.exists():
    gbm_f2 = pd.read_parquet(f2_cache_path)
    with open(f2_bias_path, "rb") as stream:
        bias_f2 = pickle.load(stream)
    print("F2 GBM cache 로드:", gbm_f2.shape)
else:
    gbm_f2, bias_f2 = cache_flow_gbm(
        train_df_tps, cols_f2, av, jobs=2, seeds=3, checkpoint=f2_checkpoint
    )
    gbm_f2.to_parquet(f2_cache_path, index=False)
    with open(f2_bias_path, "wb") as stream:
        pickle.dump(bias_f2, stream)

KEYS = ["fold", "group", "forecast_kst_dtm"]
cache_c0 = gbm_c0.merge(dl_cache, on=KEYS, how="left")
cache_f2 = gbm_f2.merge(dl_cache, on=KEYS, how="left")
assert cache_c0["dl_q60"].notna().all() and cache_f2["dl_q60"].notna().all()
assert set(map(tuple, cache_c0[KEYS].to_numpy())) == set(map(tuple, cache_f2[KEYS].to_numpy()))

def prediction_tidy(cache_frame, bias_maps, scales=None):
    parts = []
    for (fold_name, group), sub in cache_frame.groupby(["fold", "group"], sort=False):
        sub = sub.sort_values("forecast_kst_dtm")
        capacity = CAPACITY_KWH[f"kpx_group_{int(group)}"]
        cf = 0.65 * sub["gbm_cf"].to_numpy() + 0.35 * sub["dl_q60"].to_numpy()
        prediction = postprocess_simple(
            cf, capacity, sub["ws"].to_numpy(), int(group), bias_maps[fold_name]
        )
        if scales is not None:
            prediction = np.clip(prediction * scales[int(group)], 0, capacity)
        parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=prediction))
    return pd.concat(parts, ignore_index=True)

def score_one_group(frame, group):
    capacity = CAPACITY_KWH[f"kpx_group_{int(group)}"]
    valid = frame["target"].to_numpy() >= capacity * 0.10
    actual = frame.loc[valid, "target"].to_numpy()
    forecast = frame.loc[valid, "pred"].to_numpy()
    error_rate = np.abs(forecast - actual) / capacity
    one_nmae = 1.0 - float(error_rate.mean())
    unit_price = np.select([error_rate <= 0.06, error_rate <= 0.08], [4.0, 3.0], default=0.0)
    ficr = float(np.sum(actual * unit_price) / np.sum(actual * 4.0))
    return 0.5 * (one_nmae + ficr), one_nmae, ficr

tidy_by_key = {}
summary_rows, fold_rows, group_rows = [], [], []
for operation, scales in [("unscaled", None), ("g3s97", GROUP_SCALES)]:
    for experiment, frame, maps in [("C0", cache_c0, bias_c0), ("F2_FLOW", cache_f2, bias_f2)]:
        tidy = prediction_tidy(frame, maps, scales)
        tidy_by_key[(operation, experiment)] = tidy
        for partition, mask in [
            ("cv", tidy["fold"] != "hold_2024full"),
            ("hold", tidy["fold"] == "hold_2024full"),
        ]:
            score = score_from_tidy(tidy.loc[mask], "pred")
            summary_rows.append({
                "operation": operation, "experiment": experiment, "partition": partition,
                "Score": score[0], "1-NMAE": score[1], "FiCR": score[2],
            })
            for group, group_frame in tidy.loc[mask].groupby("group"):
                group_score = score_one_group(group_frame, group)
                group_rows.append({
                    "operation": operation, "experiment": experiment, "partition": partition,
                    "group": int(group), "Score": group_score[0],
                    "1-NMAE": group_score[1], "FiCR": group_score[2],
                })
        for fold_name, fold_frame in tidy.groupby("fold"):
            fold_score = score_from_tidy(fold_frame, "pred")
            fold_rows.append({
                "operation": operation, "experiment": experiment, "fold": fold_name,
                "Score": fold_score[0], "1-NMAE": fold_score[1], "FiCR": fold_score[2],
            })

summary = pd.DataFrame(summary_rows)
fold_scores = pd.DataFrame(fold_rows)
group_scores = pd.DataFrame(group_rows)

def add_control_deltas(table, key_cols):
    control = table.loc[table["experiment"] == "C0"].set_index(key_cols)
    result = table.copy()
    for metric_name in ["Score", "1-NMAE", "FiCR"]:
        result[f"delta_{metric_name}"] = [
            row[metric_name] - control.loc[tuple(row[col] for col in key_cols), metric_name]
            if row["experiment"] != "C0" else 0.0
            for _, row in result.iterrows()
        ]
    return result

summary = add_control_deltas(summary, ["operation", "partition"])
fold_scores = add_control_deltas(fold_scores, ["operation", "fold"])
group_scores = add_control_deltas(group_scores, ["operation", "partition", "group"])

primary = summary.loc[
    (summary["operation"] == "unscaled")
    & (summary["experiment"] == "F2_FLOW")
].set_index("partition")
primary_folds = fold_scores.loc[
    (fold_scores["operation"] == "unscaled")
    & (fold_scores["experiment"] == "F2_FLOW")
    & fold_scores["fold"].str.startswith("fold_")
]
positive_folds = int((primary_folds["delta_Score"] > 0).sum())
FLOW_GATE_PASS = bool(
    primary.loc["cv", "delta_Score"] >= PASS_CV_DELTA
    and positive_folds >= PASS_MIN_POSITIVE_FOLDS
    and primary.loc["hold", "delta_Score"] >= 0.0
    and primary.loc["cv", "delta_1-NMAE"] >= PASS_MIN_SUBMETRIC_DELTA
    and primary.loc["cv", "delta_FiCR"] > 0.0
)
gate_decision = {
    "passed": FLOW_GATE_PASS,
    "primary_operation": "unscaled",
    "cv_delta_score": float(primary.loc["cv", "delta_Score"]),
    "cv_delta_1_nmae": float(primary.loc["cv", "delta_1-NMAE"]),
    "cv_delta_ficr": float(primary.loc["cv", "delta_FiCR"]),
    "hold_delta_score": float(primary.loc["hold", "delta_Score"]),
    "positive_cv_folds": positive_folds,
    "required": {
        "cv_delta_score": PASS_CV_DELTA,
        "positive_cv_folds": PASS_MIN_POSITIVE_FOLDS,
        "hold_delta_score": 0.0,
        "min_cv_submetric_delta": PASS_MIN_SUBMETRIC_DELTA,
        "cv_delta_ficr_strictly_positive": True,
    },
}

summary.to_csv(RUN_ROOT / "summary.csv", index=False)
fold_scores.to_csv(RUN_ROOT / "fold_scores.csv", index=False)
group_scores.to_csv(RUN_ROOT / "group_scores.csv", index=False)
pd.concat(
    [tidy_by_key[("unscaled", "C0")].assign(experiment="C0"),
     tidy_by_key[("unscaled", "F2_FLOW")].assign(experiment="F2_FLOW")],
    ignore_index=True,
).to_parquet(RUN_ROOT / "oof_predictions.parquet", index=False)
(RUN_ROOT / "gate_decision.json").write_text(
    json.dumps(gate_decision, ensure_ascii=False, indent=2), encoding="utf-8"
)

print("\n=== pooled summary ===")
display(summary.round(6))
print("\n=== fold deltas ===")
display(fold_scores.loc[fold_scores["experiment"] == "F2_FLOW"].round(6))
print("\n=== group deltas ===")
display(group_scores.loc[group_scores["experiment"] == "F2_FLOW"].round(6))
print("\nGate:", "PASS" if FLOW_GATE_PASS else "REJECT", gate_decision)
'''
cells.append(code(gate, "flow_gate"))


final_training = r'''
#@title 14. Gate 통과 시 최종 C0/F2 학습과 submission 생성
SUBMISSION_PATHS = []
if not (RUN_FINAL_IF_PASS and FLOW_GATE_PASS):
    print("Gate 미통과 또는 RUN_FINAL_IF_PASS=False: 최종 학습을 실행하지 않습니다.")
else:
    component_path = CACHE_DIR / f"flow_final_components_{FLOW_VERSION}.parquet"
    final_bias_path = CACHE_DIR / f"flow_final_bias_{FLOW_VERSION}.pkl"

    if component_path.exists() and final_bias_path.exists():
        final_components = pd.read_parquet(component_path)
        with open(final_bias_path, "rb") as stream:
            final_bias_maps = pickle.load(stream)
        print("최종 component cache 로드:", final_components.shape)
    else:
        final_components = test_df_tps[
            ["forecast_id", "forecast_kst_dtm", "kpx_group", "ldaps_ws_10m"]
        ].copy()
        final_components["dl_cf"] = np.nan
        final_components["gbm_c0_cf"] = np.nan
        final_components["gbm_f2_cf"] = np.nan

        final_bias_maps = {}
        for name, cols in [("C0", cols_c0), ("F2_FLOW", cols_f2)]:
            bias_data = estimate_bias_data_safe(
                train_df_tps, cols, train_df_tps["forecast_kst_dtm"].max(), 500, 2
            )
            final_bias_maps[name] = bias_map_from_data(
                bias_data, {group: COARSE_BINS for group in bias_data}
            )

        for group in tqdm([1, 2, 3], desc="최종 공통 DL 및 C0/F2 GBM"):
            capacity = CAPACITY_KWH[f"kpx_group_{group}"]
            train_group = train_df_tps.loc[
                (train_df_tps["kpx_group"] == group) & train_df_tps["target"].notna()
            ].copy()
            test_group = test_df_tps.loc[test_df_tps["kpx_group"] == group].sort_values(
                "forecast_kst_dtm"
            ).copy()

            dl_prediction = dl_fold_predict(
                train_group, test_group, cols_gspd, seeds=FINAL_DL_SEEDS, cfg=DL_CFG
            )
            final_components.loc[test_group.index, "dl_cf"] = dl_prediction

            train_group = train_group.merge(
                av.loc[av["group"] == group, ["forecast_kst_dtm", "avail_strict"]],
                on="forecast_kst_dtm", how="left",
            )
            train_group["avail_strict"] = train_group["avail_strict"].fillna(1.0)
            y_train = train_group["target"]
            weights = make_sample_weight_scada(
                y_train.to_numpy(), capacity, train_group["avail_strict"]
            )

            for experiment, cols, output_col in [
                ("C0", cols_c0, "gbm_c0_cf"),
                ("F2_FLOW", cols_f2, "gbm_f2_cf"),
            ]:
                X_train, X_test = prepare_matrices(train_group, test_group, cols)
                pred_lgb = np.mean([
                    safe_lgb_predict(
                        X_train, y_train, weights, X_test, 500, 2, seed,
                        label=f"final-{experiment}-g{group}-s{seed}",
                    ) for seed in range(FINAL_GBM_SEEDS)
                ], axis=0)
                pred_xgb = np.mean([
                    xgbm.XGBRegressor(
                        n_estimators=500, max_depth=6, learning_rate=0.03,
                        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                        random_state=seed, verbosity=0, tree_method="hist", device="cuda",
                        objective="reg:absoluteerror",
                    ).fit(X_train, y_train, sample_weight=weights).predict(X_test)
                    for seed in range(FINAL_GBM_SEEDS)
                ], axis=0)
                final_components.loc[test_group.index, output_col] = (
                    0.5 * pred_lgb + 0.5 * pred_xgb
                ) / capacity

        required_components = ["dl_cf", "gbm_c0_cf", "gbm_f2_cf"]
        assert final_components[required_components].notna().all().all()
        final_components.to_parquet(component_path, index=False)
        with open(final_bias_path, "wb") as stream:
            pickle.dump(final_bias_maps, stream)

    def make_submission(gbm_column, bias_map, scales=None):
        parts = []
        for group in [1, 2, 3]:
            capacity = CAPACITY_KWH[f"kpx_group_{group}"]
            group_frame = final_components.loc[
                final_components["kpx_group"] == group
            ].sort_values("forecast_kst_dtm")
            cf = 0.65 * group_frame[gbm_column].to_numpy() + 0.35 * group_frame["dl_cf"].to_numpy()
            prediction = postprocess_simple(
                cf, capacity, group_frame["ldaps_ws_10m"].to_numpy(), group, bias_map
            )
            if scales is not None:
                prediction = np.clip(prediction * scales[group], 0, capacity)
            parts.append(pd.DataFrame({
                "forecast_id": group_frame["forecast_id"].to_numpy(),
                "forecast_kst_dtm": group_frame["forecast_kst_dtm"].to_numpy(),
                "kpx_group": group,
                "pred": prediction,
            }))
        long = pd.concat(parts, ignore_index=True)
        submission = long.pivot(
            index=["forecast_id", "forecast_kst_dtm"], columns="kpx_group", values="pred"
        )
        submission.columns = [f"kpx_group_{int(group)}" for group in submission.columns]
        submission = submission.reset_index()
        submission["forecast_kst_dtm"] = pd.to_datetime(
            submission["forecast_kst_dtm"]
        ).dt.strftime("%Y-%m-%d %H:%M:%S")
        return sample_submission[["forecast_id"]].merge(submission, on="forecast_id", how="left")

    submission_c0 = make_submission("gbm_c0_cf", final_bias_maps["C0"])
    submission_f2 = make_submission("gbm_f2_cf", final_bias_maps["F2_FLOW"])
    submission_f2_scaled = make_submission(
        "gbm_f2_cf", final_bias_maps["F2_FLOW"], GROUP_SCALES
    )

    control_output_path = RUN_ROOT / "submission_flow_c0_control.csv"
    flow_output_path = RUN_ROOT / "submission_flow_f2.csv"
    flow_scaled_output_path = RUN_ROOT / "submission_flow_f2_g3s97.csv"
    submission_c0.to_csv(control_output_path, index=False, encoding="utf-8-sig")
    submission_f2.to_csv(flow_output_path, index=False, encoding="utf-8-sig")
    submission_f2_scaled.to_csv(flow_scaled_output_path, index=False, encoding="utf-8-sig")
    SUBMISSION_PATHS = [control_output_path, flow_output_path, flow_scaled_output_path]

    delta_rows = []
    for group in [1, 2, 3]:
        col = f"kpx_group_{group}"
        delta = submission_f2[col] - submission_c0[col]
        delta_rows.append({
            "group": group,
            "mean_delta_kwh": float(delta.mean()),
            "mean_abs_delta_kwh": float(delta.abs().mean()),
            "max_abs_delta_kwh": float(delta.abs().max()),
            "mean_delta_pct_capacity": float(delta.mean() / CAPACITY_KWH[col] * 100),
        })
    pd.DataFrame(delta_rows).to_csv(RUN_ROOT / "submission_delta_summary.csv", index=False)
    print("Saved:")
    for path in SUBMISSION_PATHS:
        print(path)
'''
cells.append(code(final_training, "flow_final_training"))


verification = r'''
#@title 15. 결과 파일 검증
required_result_paths = [
    RUN_ROOT / "summary.csv",
    RUN_ROOT / "fold_scores.csv",
    RUN_ROOT / "group_scores.csv",
    RUN_ROOT / "gate_decision.json",
    RUN_ROOT / "flow_feature_summary.csv",
    RUN_ROOT / "oof_predictions.parquet",
]
assert all(path.exists() for path in required_result_paths)

for submission_path in SUBMISSION_PATHS:
    submission = pd.read_csv(submission_path, encoding="utf-8-sig")
    assert len(submission) == len(sample_submission)
    assert submission["forecast_id"].equals(sample_submission["forecast_id"])
    assert submission[TARGET_COLS].notna().all().all()
    for col in TARGET_COLS:
        assert submission[col].between(0, CAPACITY_KWH[col]).all()
    print(submission_path.name, submission.shape, "검증 통과")

print("결과 저장 경로:", RUN_ROOT)
print("Gate:", "PASS" if FLOW_GATE_PASS else "REJECT")
'''
cells.append(code(verification, "flow_verification"))


cells.append(
    markdown(
        r'''
### 생성 파일

`/content/drive/MyDrive/BARAM2026/hyejin_flow_spatial`에 다음 파일이 저장됩니다.

- `summary.csv`: C0/F2 pooled CV·HOLD 비교
- `fold_scores.csv`: fold별 변화
- `group_scores.csv`: 그룹별 1-NMAE·FiCR 변화
- `gate_decision.json`: 사전 정의 gate 판정
- `flow_feature_summary.csv`: 새 피처 커버리지와 분포
- `oof_predictions.parquet`: C0/F2 OOF 예측
- gate 통과 시 `submission_flow_c0_control.csv`, `submission_flow_f2.csv`, `submission_flow_f2_g3s97.csv`

제출 후보는 gate 통과 후 `submission_flow_f2.csv`이며, `g3s97`은 별도 1변경 후보입니다.
''',
        "flow_outputs",
    )
)


CELL_TITLES = {
    "flow_setup": "1. Colab·Drive 설정",
    "flow_base_02": "2. 기본 데이터셋 생성 함수",
    "flow_base_03": "3. 평가 지표·편향 보정·검증 fold",
    "flow_base_04": "4. 물리·시계열·공간 피처",
    "flow_base_05": "5. 다격자 편차 피처",
    "flow_base_06": "6. JMA 외부 예보",
    "flow_base_07": "7. ECMWF HRES 외부 예보",
    "flow_base_08": "8. q60 MTMLP·early stopping",
    "flow_base_09": "9. no-snap 후처리",
    "flow_base_10": "10. SCADA 가용성 가중치",
    "flow_base_11": "11. GBM·DL fold 캐시 함수",
    "flow_feature_builder": "12. Flow-aligned 공간 피처",
    "flow_data_prep": "13. C0·F2 공통 데이터 준비",
    "flow_gate": "14. C0 대 F2_FLOW ablation",
    "flow_final_training": "15. Gate 통과 시 최종 학습",
    "flow_verification": "16. 결과·submission 검증",
}

titled_cells = []
for cell in cells:
    if cell["cell_type"] == "code":
        cell_id = cell["id"]
        if cell_id not in CELL_TITLES:
            raise RuntimeError(f"missing cell title: {cell_id}")
        source_text = "".join(cell.get("source", []))
        source_lines = source_text.splitlines(keepends=True)
        if source_lines and source_lines[0].lstrip().startswith("#@title"):
            source_lines = source_lines[1:]
        cell["source"] = source_lines
        titled_cells.append(
            markdown(f'### {CELL_TITLES[cell_id]}', f'{cell_id}_title')
        )
    titled_cells.append(cell)
cells = titled_cells


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": OUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
        "experiment": {
            "name": "F2_FLOW",
            "control": "C0 external no-snap GBM",
            "changed_axis": "12 flow-aligned LDAPS/GFS spatial features in GBM only",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT)
