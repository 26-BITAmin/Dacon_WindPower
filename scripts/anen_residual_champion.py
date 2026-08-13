import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "notebooks" / "serial_residual_flow.ipynb"
OUT = ROOT / "notebooks" / "anen_residual_champion.ipynb"


def lines(text):
    text = text.strip("\n")
    parts = text.splitlines()
    return [line + "\n" for line in parts[:-1]] + [parts[-1]]


def markdown(text, cell_id):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": lines(text),
    }


def code(text, cell_id):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


source_notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
source_by_id = {cell.get("id"): cell for cell in source_notebook["cells"]}


def source_of(cell_id):
    return "".join(source_by_id[cell_id]["source"])


setup_source = source_of("serial_setup").replace(
    "hyejin_serial_residual_flow", "hyejin_anen_residual_champion"
)
setup_source = setup_source.replace(
    'FLOW_VERSION = "v1"\nRUN_FINAL_IF_PASS = True\nPASS_CV_DELTA = 0.0005\n'
    'PASS_MIN_POSITIVE_FOLDS = 3\nPASS_MIN_SUBMETRIC_DELTA = -0.0005\n',
    '',
)

data_prep_source = source_of("serial_data_prep").split(
    "flow_frame, flow_cols =", 1
)[0]
data_prep_source += r'''

assert len(cols_c0) == 269, f"C0 피처 수가 예상과 다름: {len(cols_c0)}"

sc = load_scada()
av = compute_availability(sc)
av["forecast_kst_dtm"] = pd.to_datetime(av["forecast_kst_dtm"])
availability = av.rename(columns={"group": "kpx_group"})[
    ["forecast_kst_dtm", "kpx_group", "avail_strict"]
]
train_df_tps = train_df_tps.merge(
    availability, on=["forecast_kst_dtm", "kpx_group"], how="left"
)
train_df_tps["avail_strict"] = train_df_tps["avail_strict"].fillna(1.0)
test_df_tps["avail_strict"] = 1.0

assert not train_df_tps.duplicated(["kpx_group", "forecast_kst_dtm"]).any()
assert not test_df_tps.duplicated(["kpx_group", "forecast_kst_dtm"]).any()
print("train:", train_df_tps.shape, "/ test:", test_df_tps.shape)
print("GBM features:", len(cols_c0))
'''


cells = [
    markdown(
        r'''
## BARAM 2026 - AnEn Residual Champion

현재 팀 챔피언에서 검증된 계열을 본인 파이프라인으로 독립 재현한 마지막 제출 후보입니다.

- AnEn K=30 풍속 멤버를 MTMLP 384x3의 입력으로 사용
- 학습행 AnEn과 DL base를 각각 시간 중앙 2분할 교차적합해 자기복사 누수 방지
- 대칭 Huber, 계절별 early stopping, DL 3시드
- DL 오차만 Optuna 설정 LightGBM 3시드로 보정
- 편향 보정, rolling 0.40, clip 0.16, 그룹 scale 0.98/0.98/0.97, snap 없음
- 기존 external champion OOF를 같은 fold에서 이겨야 최종 제출 파일 생성
''',
        "anen_intro",
    )
]


common_cells = [
    ("1. Colab·Drive 설정", "anen_setup", setup_source),
    ("2. 기본 데이터셋 생성 함수", "anen_base_dataset", source_of("serial_base_dataset")),
    ("3. 평가 지표·편향 보정·검증 fold", "anen_metric", source_of("serial_metric")),
    ("4. 물리·시계열·공간 피처", "anen_feature_blocks", source_of("serial_feature_blocks")),
    ("5. 다격자 편차 피처", "anen_spatial_deviation", source_of("serial_spatial_deviation")),
    ("6. JMA 외부 예보", "anen_jma", source_of("serial_jma")),
    ("7. ECMWF HRES 외부 예보", "anen_ecmwf", source_of("serial_ecmwf")),
    ("8. MTMLP 공통 구조", "anen_mtmlp", source_of("serial_mtmlp")),
    ("9. no-snap 후처리", "anen_postprocess", source_of("serial_postprocess")),
    ("10. SCADA 가용성 가중치", "anen_availability", source_of("serial_availability")),
    ("11. 269개 공통 피처 준비", "anen_data_prep", data_prep_source),
]

for title, cell_id, cell_source in common_cells:
    cells.append(markdown(f"### {title}", f"{cell_id}_title"))
    cells.append(code(cell_source, cell_id))


config_source = r'''
ANEN_VERSION = "pd16_v1"
ANEN_CACHE_DIR = CACHE_DIR / ANEN_VERSION
ANEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

K_ANEN = 30
ANEN_TWIN = 1
ANEN_HOUR_WINDOW = 4
ANEN_CHUNK = 1024
ANEN_COLS = [f"mws{k}" for k in range(K_ANEN)]
ANEN_DL_SEEDS = 3
ANEN_GBM_SEEDS = 3
ANEN_PATIENCE = 15
ANEN_ABSOLUTE_CV_FLOOR = 0.6490
ANEN_MIN_CV_DELTA = 0.0003
ANEN_MIN_POSITIVE_FOLDS = 3
ANEN_MIN_HOLD_DELTA = -0.0050
RUN_FINAL_IF_PASS = True

GROUP_CAPACITY = {1: 21600, 2: 21600, 3: 21000}
GROUP_SCALES = {1: 0.98, 2: 0.98, 3: 0.97}

ANEN_LGB_PARAMS = {
    "objective": "mae",
    "n_estimators": 633,
    "learning_rate": 0.032229933058504565,
    "num_leaves": 36,
    "min_child_samples": 141,
    "feature_fraction": 0.8262159071232439,
    "bagging_fraction": 0.9504907401583689,
    "bagging_freq": 9,
    "lambda_l1": 0.0006025002498529692,
    "lambda_l2": 0.010273205943481366,
}

W14 = {
    1: {"gfs_ws_100m": 2.0, "ldaps_ws_10m": 2.0, "ec_ws": 1.0,
        "jgsm_ws": 1.0, "wd_pair": 1.0, "gfs_wind_shear_100_10": 0.5,
        "ldaps_heightAboveGround_2_t": 0.5, "ldaps_surface_0_sp": 0.5,
        "phy_ws850": 2.0, "sdL_ws10_mean": 2.0,
        "phy_gfs_gust_factor": 2.0},
    2: {"gfs_ws_100m": 0.5, "ldaps_ws_10m": 2.0, "ec_ws": 1.0,
        "jmsm_ws": 1.0, "jgsm_ws": 0.5, "wd_pair": 0.5,
        "gfs_wind_shear_100_10": 0.5, "ldaps_heightAboveGround_2_t": 0.5,
        "ldaps_surface_0_sp": 0.5, "phy_ws850": 2.0,
        "phy_gfs_gust_factor": 1.0},
    3: {"gfs_ws_100m": 0.5, "ldaps_ws_10m": 2.0, "ec_ws": 0.5,
        "jmsm_ws": 1.0, "jgsm_ws": 0.5, "wd_pair": 0.5,
        "gfs_wind_shear_100_10": 0.5, "ldaps_heightAboveGround_2_t": 0.5,
        "phy_ws850": 2.0, "phy_ws50max": 0.5,
        "phy_gfs_gust_factor": 1.0},
}

REFERENCE_GBM_PATH = BASE_CACHE_DIR / "external_champion_gbm_v1.parquet"
REFERENCE_DL_PATH = BASE_CACHE_DIR / "external_champion_dl_v1.parquet"
REFERENCE_BIAS_PATH = BASE_CACHE_DIR / "external_champion_bias_v1.pkl"

required_anen = sorted({
    col
    for group_map in W14.values()
    for entry in group_map
    for col in (["gfs_wd_sin_100m", "gfs_wd_cos_100m"] if entry == "wd_pair" else [entry])
})
for name, frame in [("train", train_df_tps), ("test", test_df_tps)]:
    missing = [col for col in required_anen if col not in frame.columns]
    assert not missing, f"{name} AnEn 입력 누락: {missing}"

print("AnEn inputs:", len(required_anen), required_anen)
print("AnEn cache:", ANEN_CACHE_DIR)
print("reference cache:", all(path.exists() for path in [
    REFERENCE_GBM_PATH, REFERENCE_DL_PATH, REFERENCE_BIAS_PATH
]))
'''
cells.append(markdown("### 12. 챔피언 재현 설정", "anen_config_title"))
cells.append(code(config_source, "anen_config"))


anen_source = r'''
KEY = ["kpx_group", "forecast_kst_dtm"]


def entry_cols(entry):
    if entry == "wd_pair":
        return ["gfs_wd_sin_100m", "gfs_wd_cos_100m"]
    return [entry]


def lag_block(group_frame, columns, twin=ANEN_TWIN):
    values = group_frame[columns].ffill().bfill().fillna(0.0).to_numpy(dtype=np.float64)
    n_rows, n_features = values.shape
    output = np.empty((n_rows, n_features, 2 * twin + 1), dtype=np.float64)
    for out_index, shift in enumerate(range(-twin, twin + 1)):
        shifted = np.roll(values, -shift, axis=0)
        if shift < 0:
            shifted[:-shift] = values[0]
        elif shift > 0:
            shifted[-shift:] = values[-1]
        output[:, :, out_index] = shifted
    return output


def build_scada_vectors():
    raw = {
        "vestas": pd.read_csv(
            TRAIN_DIR / "scada_vestas_train.csv",
            encoding="utf-8-sig", parse_dates=["kst_dtm"],
        ),
        "unison": pd.read_csv(
            TRAIN_DIR / "scada_unison_train.csv",
            encoding="utf-8-sig", parse_dates=["kst_dtm"],
        ),
    }
    group_specs = {
        1: ("vestas", [f"vestas_wtg{i:02d}" for i in range(1, 7)]),
        2: ("vestas", [f"vestas_wtg{i:02d}" for i in range(7, 13)]),
        3: ("unison", [f"unison_wtg{i:02d}" for i in range(1, 6)]),
    }
    parts = []
    for group, (maker, turbines) in group_specs.items():
        frame = raw[maker]
        wind_cols = [f"{turbine}_ws" for turbine in turbines]
        direction_cols = [f"{turbine}_wd" for turbine in turbines]
        theta = np.deg2rad(frame[direction_cols].to_numpy(dtype=float))
        raw_u, raw_v = -np.sin(theta), -np.cos(theta)
        count_u = np.isfinite(raw_u).sum(axis=1)
        count_v = np.isfinite(raw_v).sum(axis=1)
        unit_u = np.divide(
            np.nansum(raw_u, axis=1), count_u,
            out=np.full(len(frame), np.nan), where=count_u > 0,
        )
        unit_v = np.divide(
            np.nansum(raw_v, axis=1), count_v,
            out=np.full(len(frame), np.nan), where=count_v > 0,
        )
        ten_minute = pd.DataFrame({
            "forecast_kst_dtm": frame["kst_dtm"].dt.ceil("h"),
            "park_ws": frame[wind_cols].mean(axis=1),
            "unit_u": unit_u,
            "unit_v": unit_v,
        })
        hourly = ten_minute.groupby("forecast_kst_dtm").agg(
            park_ws=("park_ws", "mean"),
            unit_u=("unit_u", "mean"),
            unit_v=("unit_v", "mean"),
            n=("park_ws", "size"),
        )
        hourly = hourly.loc[hourly["n"] == 6].drop(columns="n")
        norm = np.hypot(hourly["unit_u"], hourly["unit_v"]).replace(0, np.nan)
        hourly["u_obs"] = hourly["park_ws"] * hourly["unit_u"] / norm
        hourly["v_obs"] = hourly["park_ws"] * hourly["unit_v"] / norm
        parts.append(
            hourly.reset_index()[["forecast_kst_dtm", "park_ws", "u_obs", "v_obs"]]
            .assign(kpx_group=group)
        )
    vectors = pd.concat(parts, ignore_index=True)
    assert not vectors.duplicated(KEY).any()
    return vectors


def analog_speed_members(pool_group, query_group, weight_map):
    entries = [(entry, weight) for entry, weight in weight_map.items() if weight > 0]
    columns = sum((entry_cols(entry) for entry, _ in entries), [])
    pool_lags = lag_block(pool_group, columns)
    query_lags = lag_block(query_group, columns)
    observed = pool_group["u_obs"].notna().to_numpy()
    train_indices = np.flatnonzero(observed)
    assert len(train_indices) >= K_ANEN

    sigma = np.nanstd(pool_group[columns].to_numpy(dtype=np.float64), axis=0)
    sigma = np.nan_to_num(sigma, nan=1.0, posinf=1.0, neginf=1.0)
    column_weights = []
    column_index = 0
    for entry, weight in entries:
        n_columns = len(entry_cols(entry))
        for offset in range(n_columns):
            divisor = max(float(sigma[column_index + offset]), 1e-9)
            column_weights.append(
                weight * (0.5 if n_columns == 2 else 1.0) / divisor
            )
        column_index += n_columns
    column_weights = np.asarray(column_weights, dtype=np.float64)

    pool_values = np.ascontiguousarray(pool_lags[train_indices])
    pool_norm = (pool_values ** 2).sum(axis=2)
    pool_hours = pool_group["forecast_kst_dtm"].dt.hour.to_numpy()[train_indices]
    query_hours = query_group["forecast_kst_dtm"].dt.hour.to_numpy()
    observed_uv = pool_group.loc[observed, ["u_obs", "v_obs"]].to_numpy(dtype=np.float32)
    output = np.empty((len(query_group), K_ANEN), dtype=np.float32)

    for start in range(0, len(query_group), ANEN_CHUNK):
        query_values = query_lags[start:start + ANEN_CHUNK]
        query_norm = (query_values ** 2).sum(axis=2)
        distance = np.zeros((len(query_values), len(train_indices)), dtype=np.float64)
        for feature_index in range(pool_values.shape[1]):
            squared = (
                query_norm[:, feature_index][:, None]
                + pool_norm[:, feature_index][None, :]
                - 2.0 * (query_values[:, feature_index, :] @ pool_values[:, feature_index, :].T)
            )
            np.maximum(squared, 0.0, out=squared)
            distance += column_weights[feature_index] * np.sqrt(squared)
        hour_difference = np.abs(
            query_hours[start:start + len(query_values), None] - pool_hours[None, :]
        )
        distance[np.minimum(hour_difference, 24 - hour_difference) > ANEN_HOUR_WINDOW] = np.inf
        assert (np.isfinite(distance).sum(axis=1) >= K_ANEN).all()
        nearest = np.argpartition(distance, K_ANEN - 1, axis=1)[:, :K_ANEN]
        nearest_uv = observed_uv[nearest]
        output[start:start + len(query_values)] = np.hypot(
            nearest_uv[:, :, 0], nearest_uv[:, :, 1]
        )
    return output


def anen_speed_block(pool, query, tag):
    result = np.full((len(query), K_ANEN), np.nan, dtype=np.float32)
    query_keys = pd.MultiIndex.from_frame(query[KEY])
    for group in [1, 2, 3]:
        pool_group = pool.loc[pool["kpx_group"] == group].sort_values(
            "forecast_kst_dtm"
        ).reset_index(drop=True)
        query_group = query.loc[query["kpx_group"] == group].sort_values(
            "forecast_kst_dtm"
        ).reset_index(drop=True)
        if query_group.empty:
            continue
        members = analog_speed_members(pool_group, query_group, W14[group])
        positions = query_keys.get_indexer(pd.MultiIndex.from_frame(query_group[KEY]))
        assert (positions >= 0).all()
        result[positions] = members
        print(f"{tag}: g{group}, pool={len(pool_group):,}, query={len(query_group):,}")
    assert np.isfinite(result).all(), f"{tag}: AnEn 결측"
    return pd.concat(
        [query[KEY].reset_index(drop=True), pd.DataFrame(result, columns=ANEN_COLS)],
        axis=1,
    )


scada_vectors = build_scada_vectors()
train_anen_source = train_df_tps.merge(scada_vectors, on=KEY, how="left")
test_anen_source = test_df_tps.copy()
print("SCADA vector observations:", int(train_anen_source["u_obs"].notna().sum()))
'''
cells.append(markdown("### 13. SCADA 벡터와 AnEn K=30 멤버", "anen_builder_title"))
cells.append(code(anen_source, "anen_builder"))


dl_source = r'''
ANEN_EPOCH_ROWS = []


def frame_signature(frame):
    return pd.Timestamp(frame["forecast_kst_dtm"].max()).strftime("%Y%m%d%H%M")


def build_train_anen_members(train_window):
    signature = frame_signature(train_window)
    path = ANEN_CACHE_DIR / f"anen_train_{signature}.parquet"
    if path.exists():
        cached = pd.read_parquet(path)
        assert len(cached) == len(train_window)
        assert cached[ANEN_COLS].notna().all().all()
        print("AnEn train cache:", path.name, cached.shape)
        return cached

    observations = train_window.loc[train_window["u_obs"].notna()].copy()
    midpoint = observations["forecast_kst_dtm"].quantile(0.50)
    early_pool = observations.loc[observations["forecast_kst_dtm"] <= midpoint]
    late_pool = observations.loc[observations["forecast_kst_dtm"] > midpoint]
    early_query = train_window.loc[train_window["forecast_kst_dtm"] <= midpoint]
    late_query = train_window.loc[train_window["forecast_kst_dtm"] > midpoint]
    blocks = [
        anen_speed_block(early_pool, late_query, f"{signature}:early_to_late"),
        anen_speed_block(late_pool, early_query, f"{signature}:late_to_early"),
    ]
    cached = pd.concat(blocks, ignore_index=True)
    assert len(cached) == len(train_window)
    assert not cached.duplicated(KEY).any()
    cached.to_parquet(path, index=False)
    return cached


def build_query_anen_members(train_window, query_frame, tag):
    path = ANEN_CACHE_DIR / f"anen_query_{tag}.parquet"
    if path.exists():
        cached = pd.read_parquet(path)
        assert len(cached) == len(query_frame)
        assert cached[ANEN_COLS].notna().all().all()
        print("AnEn query cache:", path.name, cached.shape)
        return cached
    observations = train_window.loc[train_window["u_obs"].notna()].copy()
    cached = anen_speed_block(observations, query_frame, tag)
    cached.to_parquet(path, index=False)
    return cached


def merge_anen_members(frame, members):
    merged = frame.merge(members, on=KEY, how="inner", validate="one_to_one")
    assert len(merged) == len(frame), f"AnEn 병합 실패: {len(merged)} != {len(frame)}"
    assert merged[ANEN_COLS].notna().all().all()
    return merged


def anen_huber_loss(prediction, target, weight):
    return (_huber(prediction - target) * weight).sum() / weight.sum()


def anen_es_mask(frame):
    labeled = frame["target"].notna()
    dt = frame["forecast_kst_dtm"]
    last_year = int((dt.loc[labeled] - pd.Timedelta(hours=1)).dt.year.max())
    mask = (
        labeled
        & (dt.dt.year == last_year)
        & dt.dt.month.isin((3, 6, 9, 12))
        & dt.dt.day.isin(range(11, 21))
    )
    if int(mask.sum()) < 240:
        times = np.sort(dt.loc[labeled].unique())
        n_times = min(24 * 40, max(24 * 7, len(times) // 10))
        cutoff = times[-n_times]
        mask = labeled & (dt >= cutoff)
    assert mask.any() and (labeled & ~mask).any()
    return mask


def anen_xy(frame, imputer, scaler, fit=False):
    sub = frame.loc[frame["target"].notna()]
    raw = sub[ANEN_COLS].to_numpy(dtype=np.float32)
    values = (
        scaler.fit_transform(imputer.fit_transform(raw))
        if fit else scaler.transform(imputer.transform(raw))
    )
    groups = sub["kpx_group"].map(GIDX).to_numpy(dtype=np.int64)
    capacity = sub["kpx_group"].map(GROUP_CAPACITY).to_numpy(dtype=float)
    target_cf = (sub["target"].to_numpy(dtype=float) / capacity).astype(np.float32)
    low_weight = np.where(target_cf < 0.10, 0.05, 1.0)
    availability_weight = avail_factor(sub["avail_strict"].to_numpy())
    weight = (low_weight * availability_weight).astype(np.float32)
    return values.astype(np.float32), groups, target_cf, weight


def train_anen_dl(
    X_train, group_train, y_train, weight_train,
    X_es, group_es, y_es, weight_es, seed,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MTMLP(
        X_train.shape[1], width=384, depth=3, p_drop=0.15
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)
    Xt = torch.tensor(X_train, device=DEVICE)
    gt = torch.tensor(group_train, device=DEVICE)
    yt = torch.tensor(y_train, device=DEVICE)
    wt = torch.tensor(weight_train, device=DEVICE)
    Xe = torch.tensor(X_es, device=DEVICE)
    ge = torch.tensor(group_es, device=DEVICE)
    ye = torch.tensor(y_es, device=DEVICE)
    we = torch.tensor(weight_es, device=DEVICE)

    best_loss, best_epoch, bad_epochs, best_state = float("inf"), 0, 0, None
    for epoch in range(120):
        model.train()
        permutation = torch.randperm(len(Xt), device=DEVICE)
        for start in range(0, len(Xt), 1024):
            idx = permutation[start:start + 1024]
            optimizer.zero_grad()
            loss = anen_huber_loss(model(Xt[idx], gt[idx]), yt[idx], wt[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            validation_loss = anen_huber_loss(model(Xe, ge), ye, we).item()
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch + 1
            bad_epochs = 0
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= ANEN_PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_epoch, best_loss


def anen_dl_predict(train_frame, query_frame, tag):
    validation_mask = anen_es_mask(train_frame)
    fit_frame = train_frame.loc[~validation_mask]
    es_frame = train_frame.loc[validation_mask]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    X_train, group_train, y_train, weight_train = anen_xy(
        fit_frame, imputer, scaler, fit=True
    )
    X_es, group_es, y_es, weight_es = anen_xy(
        es_frame, imputer, scaler, fit=False
    )
    X_query = scaler.transform(
        imputer.transform(query_frame[ANEN_COLS].to_numpy(dtype=np.float32))
    ).astype(np.float32)
    group_query = query_frame["kpx_group"].map(GIDX).to_numpy(dtype=np.int64)

    predictions = []
    for seed in range(ANEN_DL_SEEDS):
        model, best_epoch, best_loss = train_anen_dl(
            X_train, group_train, y_train, weight_train,
            X_es, group_es, y_es, weight_es, seed,
        )
        predictions.append(predict(model, X_query, group_query))
        ANEN_EPOCH_ROWS.append({
            "tag": tag, "seed": seed, "best_epoch": best_epoch,
            "best_loss": best_loss, "n_fit": len(fit_frame),
            "n_es": len(es_frame), "n_query": len(query_frame),
        })
        del model
        torch.cuda.empty_cache()
    print(tag, "best epochs:", [row["best_epoch"] for row in ANEN_EPOCH_ROWS[-3:]])
    return np.mean(predictions, axis=0)


def build_dl_base(train_window, query_frame, tag):
    signature = frame_signature(train_window)
    train_path = ANEN_CACHE_DIR / f"dl_train_{signature}.parquet"
    query_path = ANEN_CACHE_DIR / f"dl_query_{tag}.parquet"

    train_members = build_train_anen_members(train_window)
    labeled = merge_anen_members(
        train_window.loc[train_window["target"].notna()].copy(), train_members
    )
    if train_path.exists():
        train_base = pd.read_parquet(train_path)
        assert len(train_base) == len(labeled)
        print("DL train cache:", train_path.name, train_base.shape)
    else:
        midpoint = labeled["forecast_kst_dtm"].quantile(0.50)
        early = labeled["forecast_kst_dtm"] <= midpoint
        crossfit_prediction = np.full(len(labeled), np.nan, dtype=float)
        for fit_mask, query_mask, suffix in [
            (early, ~early, "early_to_late"),
            (~early, early, "late_to_early"),
        ]:
            crossfit_prediction[np.flatnonzero(query_mask.to_numpy())] = anen_dl_predict(
                labeled.loc[fit_mask], labeled.loc[query_mask],
                f"{signature}:{suffix}",
            )
        assert np.isfinite(crossfit_prediction).all()
        train_base = labeled[KEY + ["target"]].copy()
        train_base["pd_dl_cf"] = crossfit_prediction
        train_base.to_parquet(train_path, index=False)

    if query_path.exists():
        query_base = pd.read_parquet(query_path)
        assert len(query_base) == len(query_frame)
        print("DL query cache:", query_path.name, query_base.shape)
    else:
        query_members = build_query_anen_members(train_window, query_frame, tag)
        query_with_members = merge_anen_members(query_frame, query_members)
        query_prediction = anen_dl_predict(labeled, query_with_members, tag)
        keep = KEY + (["forecast_id"] if "forecast_id" in query_frame.columns else [])
        query_base = query_frame[keep].copy()
        query_base["pd_dl_cf"] = query_prediction
        query_base.to_parquet(query_path, index=False)

    if ANEN_EPOCH_ROWS:
        pd.DataFrame(ANEN_EPOCH_ROWS).drop_duplicates(
            ["tag", "seed"], keep="last"
        ).to_csv(RUN_ROOT / "best_epochs.csv", index=False)
    return train_base, query_base
'''
cells.append(markdown("### 14. AnEn 입력 DL과 이중 교차적합", "anen_dl_title"))
cells.append(code(dl_source, "anen_dl"))


residual_source = r'''
def optimized_lgb_predict(X_train, y_train, sample_weight, X_eval, seed, label):
    params = dict(ANEN_LGB_PARAMS)
    params.update(random_state=seed, n_jobs=2, verbose=-1, device="gpu")
    try:
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    except lgb.basic.LightGBMError as error:
        print(f"[{label}] GPU LightGBM 실패, CPU 재시도: {str(error).splitlines()[0]}")
        params.update(device="cpu", force_col_wise=True)
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict(X_eval)


def residual_predict(train_group, eval_group, group, label):
    feature_cols = cols_c0 + ["pd_dl_cf"]
    X_train, X_eval = prepare_matrices(train_group, eval_group, feature_cols)
    capacity = GROUP_CAPACITY[group]
    residual_kwh = (
        train_group["target"].to_numpy(dtype=float)
        - train_group["pd_dl_cf"].to_numpy(dtype=float) * capacity
    )
    sample_weight = make_sample_weight_scada(
        train_group["target"].to_numpy(dtype=float),
        capacity,
        train_group["avail_strict"],
    )
    predictions = [
        optimized_lgb_predict(
            X_train, residual_kwh, sample_weight, X_eval, seed,
            f"{label}:g{group}:s{seed}",
        )
        for seed in range(ANEN_GBM_SEEDS)
    ]
    return np.mean(predictions, axis=0) / capacity


def estimate_final_bias_map(train_frame):
    bias_data = estimate_bias_data(
        train_frame, cols_c0, train_frame["forecast_kst_dtm"].max(),
        n_estimators=500, n_jobs=2,
    )
    return bias_map_from_data(
        bias_data, {group: COARSE_BINS for group in bias_data}
    )


def merge_dl_base(frame, base):
    merged = frame.merge(base[KEY + ["pd_dl_cf"]], on=KEY, how="inner", validate="one_to_one")
    assert len(merged) == len(frame)
    assert merged["pd_dl_cf"].notna().all()
    return merged


if REFERENCE_BIAS_PATH.exists():
    with open(REFERENCE_BIAS_PATH, "rb") as stream:
        fold_bias_maps = pickle.load(stream)
    print("기존 external bias map 재사용")
else:
    fold_bias_maps = {}
    print("기존 bias cache 없음: fold별로 재계산")
'''
cells.append(markdown("### 15. Optuna 설정 잔차 LightGBM", "anen_residual_title"))
cells.append(code(residual_source, "anen_residual"))


validation_source = r'''
FOLD_CACHE_PATH = ANEN_CACHE_DIR / "fold_predictions.parquet"
fold_map = {name: fold for name, fold in FOLD_LIST}


def bias_for_fold(fold_name, fold):
    if fold_name in fold_bias_maps:
        return fold_bias_maps[fold_name]
    bias_data = estimate_bias_data(
        train_df_tps, cols_c0, fold["train_end"],
        n_estimators=500, n_jobs=2,
    )
    fold_bias_maps[fold_name] = bias_map_from_data(
        bias_data, {group: COARSE_BINS for group in bias_data}
    )
    return fold_bias_maps[fold_name]


def predict_one_fold(fold_name, fold):
    fold_path = ANEN_CACHE_DIR / f"prediction_{fold_name}.parquet"
    if fold_path.exists():
        cached = pd.read_parquet(fold_path)
        assert cached[["target", "pred_unscaled", "pred_scaled"]].notna().all().all()
        print("fold prediction cache:", fold_path.name, cached.shape)
        return cached

    started = time.time()
    train_window = train_anen_source.loc[
        train_anen_source["forecast_kst_dtm"] <= fold["train_end"]
    ].copy()
    valid_window = train_anen_source.loc[
        (train_anen_source["forecast_kst_dtm"] >= fold["valid_start"])
        & (train_anen_source["forecast_kst_dtm"] <= fold["valid_end"])
        & train_anen_source["target"].notna()
    ].copy()
    train_base, valid_base = build_dl_base(train_window, valid_window, fold_name)
    train_model = merge_dl_base(
        train_window.loc[train_window["target"].notna()].copy(), train_base
    )
    valid_model = merge_dl_base(valid_window, valid_base)
    bias_map = bias_for_fold(fold_name, fold)

    parts = []
    for group in [1, 2, 3]:
        train_group = train_model.loc[train_model["kpx_group"] == group].copy()
        valid_group = valid_model.loc[valid_model["kpx_group"] == group].sort_values(
            "forecast_kst_dtm"
        ).copy()
        residual_cf = residual_predict(
            train_group, valid_group, group, fold_name
        )
        raw_cf = valid_group["pd_dl_cf"].to_numpy(dtype=float) + residual_cf
        capacity = GROUP_CAPACITY[group]
        unscaled = postprocess_simple(
            raw_cf, capacity,
            valid_group["ldaps_ws_10m"].to_numpy(dtype=float),
            group, bias_map, clip_lo=0.16, alpha=0.40,
        )
        scaled = np.clip(unscaled * GROUP_SCALES[group], 0, capacity)
        parts.append(pd.DataFrame({
            "fold": fold_name,
            "group": group,
            "forecast_kst_dtm": valid_group["forecast_kst_dtm"].to_numpy(),
            "target": valid_group["target"].to_numpy(dtype=float),
            "ws": valid_group["ldaps_ws_10m"].to_numpy(dtype=float),
            "dl_base_cf": valid_group["pd_dl_cf"].to_numpy(dtype=float),
            "residual_cf": residual_cf,
            "raw_cf": raw_cf,
            "pred_unscaled": unscaled,
            "pred_scaled": scaled,
        }))
    output = pd.concat(parts, ignore_index=True)
    output.to_parquet(fold_path, index=False)
    print(f"{fold_name} complete: {(time.time() - started) / 60:.1f}분")
    return output


# HOLD를 먼저 계산하면 같은 학습창인 Q1의 AnEn·DL·GBM을 재사용할 수 있습니다.
run_order = ["hold_2024full", "fold_2024Q2", "fold_2024Q3", "fold_2024Q4"]
fold_parts = [predict_one_fold(name, fold_map[name]) for name in run_order]
hold_predictions = fold_parts[0]
q1_fold = fold_map["fold_2024Q1"]
q1_raw = hold_predictions.loc[
    (hold_predictions["forecast_kst_dtm"] >= q1_fold["valid_start"])
    & (hold_predictions["forecast_kst_dtm"] <= q1_fold["valid_end"])
].copy()
q1_raw["fold"] = "fold_2024Q1"
q1_parts = []
q1_bias = bias_for_fold("fold_2024Q1", q1_fold)
for group in [1, 2, 3]:
    group_frame = q1_raw.loc[q1_raw["group"] == group].sort_values(
        "forecast_kst_dtm"
    ).copy()
    capacity = GROUP_CAPACITY[group]
    unscaled = postprocess_simple(
        group_frame["raw_cf"].to_numpy(dtype=float), capacity,
        group_frame["ws"].to_numpy(dtype=float), group, q1_bias,
        clip_lo=0.16, alpha=0.40,
    )
    group_frame["pred_unscaled"] = unscaled
    group_frame["pred_scaled"] = np.clip(
        unscaled * GROUP_SCALES[group], 0, capacity
    )
    q1_parts.append(group_frame)
fold_parts.append(pd.concat(q1_parts, ignore_index=True))

candidate_oof = pd.concat(fold_parts, ignore_index=True)
assert set(candidate_oof["fold"].unique()) == {name for name, _ in FOLD_LIST}
assert candidate_oof[["target", "pred_unscaled", "pred_scaled"]].notna().all().all()
candidate_oof.to_parquet(FOLD_CACHE_PATH, index=False)
with open(ANEN_CACHE_DIR / "fold_bias_maps.pkl", "wb") as stream:
    pickle.dump(fold_bias_maps, stream)
print("OOF:", candidate_oof.shape)
'''
cells.append(markdown("### 16. 4-fold·HOLD 챔피언 후보 검증", "anen_validation_title"))
cells.append(code(validation_source, "anen_validation"))


scoring_source = r'''
def one_group_score(frame, pred_col, group):
    capacity = GROUP_CAPACITY[group]
    actual = frame["target"].to_numpy(dtype=float)
    prediction = frame[pred_col].to_numpy(dtype=float)
    valid = actual >= capacity * 0.10
    actual, prediction = actual[valid], prediction[valid]
    error_rate = np.abs(prediction - actual) / capacity
    one_nmae = 1.0 - float(error_rate.mean())
    unit_price = np.select(
        [error_rate <= 0.06, error_rate <= 0.08], [4.0, 3.0], default=0.0
    )
    ficr = float(np.sum(actual * unit_price) / np.sum(actual * 4.0))
    return 0.5 * (one_nmae + ficr), one_nmae, ficr


def score_partitions(frame, pred_col, model_name):
    rows = []
    for partition, mask in [
        ("cv", frame["fold"] != "hold_2024full"),
        ("hold", frame["fold"] == "hold_2024full"),
    ]:
        result = score_from_tidy(frame.loc[mask], pred_col)
        rows.append({
            "model": model_name, "partition": partition,
            "Score": result[0], "1-NMAE": result[1], "FiCR": result[2],
        })
    return rows


reference_available = all(path.exists() for path in [
    REFERENCE_GBM_PATH, REFERENCE_DL_PATH, REFERENCE_BIAS_PATH
])
reference_oof = None
if reference_available:
    reference_gbm = pd.read_parquet(REFERENCE_GBM_PATH)
    reference_dl = pd.read_parquet(REFERENCE_DL_PATH)
    reference_cache = reference_gbm.merge(
        reference_dl, on=["fold", "group", "forecast_kst_dtm"], how="left"
    )
    with open(REFERENCE_BIAS_PATH, "rb") as stream:
        reference_bias = pickle.load(stream)
    reference_parts = []
    for (fold_name, group), sub in reference_cache.groupby(["fold", "group"], sort=False):
        sub = sub.sort_values("forecast_kst_dtm").copy()
        capacity = GROUP_CAPACITY[int(group)]
        combined_cf = 0.65 * sub["gbm_cf"].to_numpy() + 0.35 * sub["dl_q60"].to_numpy()
        prediction = postprocess_simple(
            combined_cf, capacity, sub["ws"].to_numpy(), int(group),
            reference_bias[fold_name], clip_lo=0.16, alpha=0.40,
        )
        reference_parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(
            pred_reference=prediction
        ))
    reference_oof = pd.concat(reference_parts, ignore_index=True)

summary_rows = []
summary_rows += score_partitions(candidate_oof, "pred_unscaled", "AnEn_residual_unscaled")
summary_rows += score_partitions(candidate_oof, "pred_scaled", "AnEn_residual_scaled")
if reference_available:
    summary_rows += score_partitions(reference_oof, "pred_reference", "external_reference")
summary = pd.DataFrame(summary_rows)

fold_rows = []
for fold_name, fold_frame in candidate_oof.groupby("fold"):
    for model_name, pred_col in [
        ("AnEn_residual_unscaled", "pred_unscaled"),
        ("AnEn_residual_scaled", "pred_scaled"),
    ]:
        result = score_from_tidy(fold_frame, pred_col)
        fold_rows.append({
            "model": model_name, "fold": fold_name,
            "Score": result[0], "1-NMAE": result[1], "FiCR": result[2],
        })
if reference_available:
    for fold_name, fold_frame in reference_oof.groupby("fold"):
        result = score_from_tidy(fold_frame, "pred_reference")
        fold_rows.append({
            "model": "external_reference", "fold": fold_name,
            "Score": result[0], "1-NMAE": result[1], "FiCR": result[2],
        })
fold_scores = pd.DataFrame(fold_rows)

group_rows = []
for partition, partition_mask in [
    ("cv", candidate_oof["fold"] != "hold_2024full"),
    ("hold", candidate_oof["fold"] == "hold_2024full"),
]:
    for group in [1, 2, 3]:
        group_frame = candidate_oof.loc[partition_mask & (candidate_oof["group"] == group)]
        for model_name, pred_col in [
            ("AnEn_residual_unscaled", "pred_unscaled"),
            ("AnEn_residual_scaled", "pred_scaled"),
        ]:
            result = one_group_score(group_frame, pred_col, group)
            group_rows.append({
                "model": model_name, "partition": partition, "group": group,
                "Score": result[0], "1-NMAE": result[1], "FiCR": result[2],
            })
group_scores = pd.DataFrame(group_rows)

candidate_cv = summary.loc[
    (summary["model"] == "AnEn_residual_scaled") & (summary["partition"] == "cv")
].iloc[0]
candidate_hold = summary.loc[
    (summary["model"] == "AnEn_residual_scaled") & (summary["partition"] == "hold")
].iloc[0]

if reference_available:
    reference_cv = summary.loc[
        (summary["model"] == "external_reference") & (summary["partition"] == "cv")
    ].iloc[0]
    reference_hold = summary.loc[
        (summary["model"] == "external_reference") & (summary["partition"] == "hold")
    ].iloc[0]
    cv_delta = float(candidate_cv["Score"] - reference_cv["Score"])
    cv_delta_nmae = float(candidate_cv["1-NMAE"] - reference_cv["1-NMAE"])
    cv_delta_ficr = float(candidate_cv["FiCR"] - reference_cv["FiCR"])
    hold_delta = float(candidate_hold["Score"] - reference_hold["Score"])
    candidate_fold_table = fold_scores.loc[
        fold_scores["model"] == "AnEn_residual_scaled", ["fold", "Score"]
    ].rename(columns={"Score": "candidate_score"})
    reference_fold_table = fold_scores.loc[
        fold_scores["model"] == "external_reference", ["fold", "Score"]
    ].rename(columns={"Score": "reference_score"})
    fold_delta_table = candidate_fold_table.merge(reference_fold_table, on="fold")
    fold_delta_table["delta_score"] = (
        fold_delta_table["candidate_score"] - fold_delta_table["reference_score"]
    )
    positive_folds = int(
        (fold_delta_table.loc[fold_delta_table["fold"].str.startswith("fold_"), "delta_score"] > 0).sum()
    )
else:
    cv_delta = cv_delta_nmae = cv_delta_ficr = hold_delta = float("nan")
    positive_folds = 0
    fold_delta_table = pd.DataFrame()

ANEN_GATE_PASS = bool(
    reference_available
    and candidate_cv["Score"] >= ANEN_ABSOLUTE_CV_FLOOR
    and cv_delta >= ANEN_MIN_CV_DELTA
    and positive_folds >= ANEN_MIN_POSITIVE_FOLDS
    and cv_delta_nmae >= -0.0010
    and cv_delta_ficr > 0.0
    and hold_delta >= ANEN_MIN_HOLD_DELTA
)

gate_decision = {
    "passed": ANEN_GATE_PASS,
    "recommendation": "SUBMIT" if ANEN_GATE_PASS else "DO_NOT_SUBMIT",
    "candidate": "AnEn_residual_scaled",
    "reference": "external_reference",
    "candidate_cv_score": float(candidate_cv["Score"]),
    "candidate_cv_1_nmae": float(candidate_cv["1-NMAE"]),
    "candidate_cv_ficr": float(candidate_cv["FiCR"]),
    "candidate_hold_score": float(candidate_hold["Score"]),
    "cv_delta_score": cv_delta,
    "cv_delta_1_nmae": cv_delta_nmae,
    "cv_delta_ficr": cv_delta_ficr,
    "hold_delta_score": hold_delta,
    "positive_cv_folds": positive_folds,
    "thresholds": {
        "absolute_cv_floor": ANEN_ABSOLUTE_CV_FLOOR,
        "min_cv_delta": ANEN_MIN_CV_DELTA,
        "min_positive_folds": ANEN_MIN_POSITIVE_FOLDS,
        "min_hold_delta": ANEN_MIN_HOLD_DELTA,
    },
}

run_config = {
    "experiment": "ANEN_RESIDUAL_PD16",
    "version": ANEN_VERSION,
    "architecture_source": "team champion family, independently rebuilt",
    "anen": {"K": K_ANEN, "twin": ANEN_TWIN, "hour_window": ANEN_HOUR_WINDOW,
             "weights": W14},
    "dl": {"inputs": ANEN_COLS, "width": 384, "depth": 3,
           "loss": "symmetric_huber", "seeds": ANEN_DL_SEEDS,
           "early_stopping": "last training year Mar/Jun/Sep/Dec days 11-20"},
    "residual_lgb": {**ANEN_LGB_PARAMS, "seeds": ANEN_GBM_SEEDS},
    "postprocess": {"rolling_alpha": 0.40, "clip_lo": 0.16,
                    "group_scales": GROUP_SCALES, "snap": False},
}

summary.to_csv(RUN_ROOT / "summary.csv", index=False)
fold_scores.to_csv(RUN_ROOT / "fold_scores.csv", index=False)
group_scores.to_csv(RUN_ROOT / "group_scores.csv", index=False)
fold_delta_table.to_csv(RUN_ROOT / "fold_delta_vs_external.csv", index=False)
candidate_oof.to_parquet(RUN_ROOT / "oof_anen_residual.parquet", index=False)
(RUN_ROOT / "gate_decision.json").write_text(
    json.dumps(gate_decision, ensure_ascii=False, indent=2), encoding="utf-8"
)
(RUN_ROOT / "run_config.json").write_text(
    json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
)

display(summary.round(6))
display(fold_delta_table.round(6))
display(group_scores.round(6))
print("Gate:", "PASS" if ANEN_GATE_PASS else "REJECT")
print(json.dumps(gate_decision, ensure_ascii=False, indent=2))
'''
cells.append(markdown("### 17. external 기준 비교와 제출 gate", "anen_scoring_title"))
cells.append(code(scoring_source, "anen_scoring"))


final_source = r'''
SUBMISSION_PATH = RUN_ROOT / "submission_anen_residual_pd16.csv"


def make_submission(final_components, bias_map):
    parts = []
    for group in [1, 2, 3]:
        capacity = GROUP_CAPACITY[group]
        group_frame = final_components.loc[
            final_components["kpx_group"] == group
        ].sort_values("forecast_kst_dtm")
        prediction = postprocess_simple(
            group_frame["raw_cf"].to_numpy(dtype=float), capacity,
            group_frame["ldaps_ws_10m"].to_numpy(dtype=float),
            group, bias_map, clip_lo=0.16, alpha=0.40,
        )
        prediction = np.clip(prediction * GROUP_SCALES[group], 0, capacity)
        parts.append(pd.DataFrame({
            "forecast_id": group_frame["forecast_id"].to_numpy(),
            "forecast_kst_dtm": group_frame["forecast_kst_dtm"].to_numpy(),
            "kpx_group": group,
            "prediction": prediction,
        }))
    long = pd.concat(parts, ignore_index=True)
    wide = long.pivot(
        index=["forecast_id", "forecast_kst_dtm"],
        columns="kpx_group", values="prediction",
    )
    wide.columns = [f"kpx_group_{int(group)}" for group in wide.columns]
    wide = wide.reset_index()
    wide["forecast_kst_dtm"] = pd.to_datetime(
        wide["forecast_kst_dtm"]
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    return sample_submission[["forecast_id"]].merge(
        wide, on="forecast_id", how="left", validate="one_to_one"
    )


if not ANEN_GATE_PASS and RUN_FINAL_IF_PASS:
    print("Gate REJECT: 마지막 제출권을 보호하기 위해 최종 학습을 실행하지 않습니다.")
    print("summary.csv와 gate_decision.json을 먼저 확인하세요.")
else:
    final_component_path = ANEN_CACHE_DIR / "final_components.parquet"
    final_bias_path = ANEN_CACHE_DIR / "final_bias.pkl"
    if final_component_path.exists() and final_bias_path.exists():
        final_components = pd.read_parquet(final_component_path)
        with open(final_bias_path, "rb") as stream:
            final_bias_map = pickle.load(stream)
        print("final cache:", final_components.shape)
    else:
        full_train = train_anen_source.copy()
        final_train_base, final_test_base = build_dl_base(
            full_train, test_anen_source, "final_test"
        )
        train_final = merge_dl_base(
            full_train.loc[full_train["target"].notna()].copy(), final_train_base
        )
        test_final = test_df_tps.merge(
            final_test_base[KEY + ["pd_dl_cf"]],
            on=KEY, how="inner", validate="one_to_one",
        )
        assert len(test_final) == len(test_df_tps)

        component_parts = []
        for group in [1, 2, 3]:
            train_group = train_final.loc[train_final["kpx_group"] == group].copy()
            test_group = test_final.loc[test_final["kpx_group"] == group].sort_values(
                "forecast_kst_dtm"
            ).copy()
            residual_cf = residual_predict(
                train_group, test_group, group, "final"
            )
            component_parts.append(pd.DataFrame({
                "forecast_id": test_group["forecast_id"].to_numpy(),
                "forecast_kst_dtm": test_group["forecast_kst_dtm"].to_numpy(),
                "kpx_group": group,
                "ldaps_ws_10m": test_group["ldaps_ws_10m"].to_numpy(dtype=float),
                "dl_base_cf": test_group["pd_dl_cf"].to_numpy(dtype=float),
                "residual_cf": residual_cf,
                "raw_cf": test_group["pd_dl_cf"].to_numpy(dtype=float) + residual_cf,
            }))
        final_components = pd.concat(component_parts, ignore_index=True)
        final_bias_map = estimate_final_bias_map(train_df_tps)
        final_components.to_parquet(final_component_path, index=False)
        with open(final_bias_path, "wb") as stream:
            pickle.dump(final_bias_map, stream)

    submission = make_submission(final_components, final_bias_map)
    submission.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8-sig")
    print("Saved:", SUBMISSION_PATH)
'''
cells.append(markdown("### 18. gate 기반 최종 학습과 제출 생성", "anen_final_title"))
cells.append(code(final_source, "anen_final"))


verify_source = r'''
required_result_paths = [
    RUN_ROOT / "summary.csv",
    RUN_ROOT / "fold_scores.csv",
    RUN_ROOT / "group_scores.csv",
    RUN_ROOT / "fold_delta_vs_external.csv",
    RUN_ROOT / "gate_decision.json",
    RUN_ROOT / "run_config.json",
    RUN_ROOT / "oof_anen_residual.parquet",
]
assert all(path.exists() for path in required_result_paths)

if ANEN_GATE_PASS:
    assert SUBMISSION_PATH.exists()
    submission = pd.read_csv(SUBMISSION_PATH, encoding="utf-8-sig")
    assert submission.shape == sample_submission.shape == (8760, 5)
    assert list(submission.columns) == list(sample_submission.columns)
    assert submission["forecast_id"].equals(sample_submission["forecast_id"])
    assert submission[TARGET_COLS].notna().all().all()
    for column in TARGET_COLS:
        assert submission[column].between(0, CAPACITY_KWH[column]).all()
    print(SUBMISSION_PATH.name, submission.shape, "검증 통과")
    print("제출 권장 파일:", SUBMISSION_PATH)
else:
    print("Gate REJECT: 이번 후보는 제출하지 않습니다.")

print("결과 저장 경로:", RUN_ROOT)
print("최종 판정:", gate_decision["recommendation"])
'''
cells.append(markdown("### 19. 결과·제출 파일 무결성 검증", "anen_verify_title"))
cells.append(code(verify_source, "anen_verify"))


cells.append(markdown(
    r'''
### 생성 파일

`/content/drive/MyDrive/BARAM2026/hyejin_anen_residual_champion`에 저장됩니다.

- `summary.csv`: 후보와 기존 external 기준의 CV·HOLD 점수
- `fold_scores.csv`: fold별 점수
- `fold_delta_vs_external.csv`: 동일 fold 기준 개선량
- `group_scores.csv`: 그룹별 후보 점수
- `gate_decision.json`: 마지막 제출 여부 판정
- `best_epochs.csv`: DL seed별 best epoch
- `oof_anen_residual.parquet`: OOF 예측과 DL·잔차 구성 요소
- `submission_anen_residual_pd16.csv`: gate를 통과했을 때만 생성되는 제출 권장 파일
- `cache/`: AnEn, DL, LightGBM 중간 결과

최종 셀에서 `SUBMIT`이 출력된 경우에만 `submission_anen_residual_pd16.csv`를 제출합니다.
`DO_NOT_SUBMIT`이면 마지막 제출권을 사용하지 말고 결과 파일을 공유합니다.
''',
    "anen_outputs",
))


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT)
