import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "notebooks" / "flow_spatial_ablation.ipynb"
OUT = ROOT / "notebooks" / "serial_residual_flow.ipynb"


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


setup_source = source_of("flow_setup").replace(
    "hyejin_flow_spatial", "hyejin_serial_residual_flow"
)

data_prep_source = source_of("flow_data_prep").split("base_gbm_path =", 1)[0]
data_prep_source += r'''

availability = av.rename(columns={"group": "kpx_group"})[
    ["forecast_kst_dtm", "kpx_group", "avail_strict"]
]
train_df_tps = train_df_tps.merge(
    availability, on=["forecast_kst_dtm", "kpx_group"], how="left"
)
train_df_tps["avail_strict"] = train_df_tps["avail_strict"].fillna(1.0)
test_df_tps["avail_strict"] = 1.0

print("train:", train_df_tps.shape, "/ test:", test_df_tps.shape)
print("C0 features:", len(cols_c0), "/ C0+Flow:", len(cols_f2))
'''


cells = [
    markdown(
        r'''
## BARAM 2026 - Serial Residual + Flow Ablation

원우의 직렬 잔차 구조에서 아이디어만 참고하고, 우리 데이터·외부 예보·Flow 피처로 독립 구현한 실험입니다.

- 1단계: 풍속 중심 MTMLP가 capacity factor base를 예측
- 2단계: LightGBM이 `target CF - DL base CF` 잔차를 학습
- 통제 실험: 잔차 LightGBM의 입력만 `C0` 대 `C0 + Flow 12개`로 변경
- DL 구조·시드·후처리·검증 fold는 두 실험에서 고정
''',
        "serial_intro",
    )
]


common_cells = [
    ("1. Colab·Drive 설정", "serial_setup", setup_source),
    ("2. 기본 데이터셋 생성 함수", "serial_base_dataset", source_of("flow_base_02")),
    ("3. 평가 지표·편향 보정·검증 fold", "serial_metric", source_of("flow_base_03")),
    ("4. 물리·시계열·공간 피처", "serial_feature_blocks", source_of("flow_base_04")),
    ("5. 다격자 편차 피처", "serial_spatial_deviation", source_of("flow_base_05")),
    ("6. JMA 외부 예보", "serial_jma", source_of("flow_base_06")),
    ("7. ECMWF HRES 외부 예보", "serial_ecmwf", source_of("flow_base_07")),
    ("8. MTMLP 공통 구조", "serial_mtmlp", source_of("flow_base_08")),
    ("9. no-snap 후처리", "serial_postprocess", source_of("flow_base_09")),
    ("10. SCADA 가용성 가중치", "serial_availability", source_of("flow_base_10")),
    ("11. Flow-aligned 공간 피처", "serial_flow_features", source_of("flow_feature_builder")),
    ("12. C0·Flow 공통 데이터 준비", "serial_data_prep", data_prep_source),
]

for title, cell_id, cell_source in common_cells:
    cells.append(markdown(f"### {title}", f"{cell_id}_title"))
    cells.append(code(cell_source, cell_id))


serial_config = r'''
SERIAL_VERSION = "v1"
SERIAL_CACHE_DIR = CACHE_DIR / f"serial_residual_{SERIAL_VERSION}"
SERIAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

SERIAL_DL_SEEDS = 3
SERIAL_GBM_SEEDS = 3
SERIAL_PATIENCE = 15
PRIMARY_OPERATION = "scaled"
PRIMARY_EXPERIMENT = "R1_FLOW"
RUN_FINAL_MODEL = True

SERIAL_PASS_CV_DELTA = 0.0005
SERIAL_PASS_MIN_POSITIVE_FOLDS = 3
SERIAL_PASS_MIN_SUBMETRIC_DELTA = -0.0005

GROUP_CAPACITY = {1: 21600, 2: 21600, 3: 21000}

SERIAL_LGB_PARAMS = {
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

WIND_BASE_CANDIDATES = [
    "ldaps_ws_10m",
    "gfs_ws_10m",
    "gfs_ws_80m",
    "gfs_ws_100m",
    "gfs_ws_pbl",
    "gfs_surface_0_gust",
    "gfs_wind_shear_100_10",
    "gfs_wind_shear_80_10",
    "phy_ws50max",
    "phy_ws850",
    "phy_pc100",
    "spa_l10_mean",
    "spa_g100_mean",
    "tmp_l10_r3_mean",
    "tmp_l10_r7_mean",
    "tmp_g100_r3_mean",
    "tmp_g100_r7_mean",
    "tmp_day_mean",
    "tmp_day_max",
    "jgsm_ws",
    "jmsm_ws",
    "ec_ws",
]
DL_WIND_COLS = [col for col in WIND_BASE_CANDIDATES if col in train_df_tps.columns]
assert len(DL_WIND_COLS) >= 15, f"풍속 DL 입력이 너무 적음: {DL_WIND_COLS}"

EXPERIMENT_COLS = {
    "R0_CONTROL": cols_c0 + ["serial_base_cf"],
    "R1_FLOW": cols_f2 + ["serial_base_cf"],
}
assert set(flow_cols).isdisjoint(cols_c0)
assert len(EXPERIMENT_COLS["R1_FLOW"]) - len(EXPERIMENT_COLS["R0_CONTROL"]) == 12

REFERENCE_OOF_PATH = Path(
    "/content/drive/MyDrive/BARAM2026/hyejin_flow_spatial/oof_predictions.parquet"
) if IN_COLAB else Path.cwd() / "runs" / "hyejin_flow_spatial" / "oof_predictions.parquet"

print("DL wind features:", len(DL_WIND_COLS))
print(DL_WIND_COLS)
print("serial cache:", SERIAL_CACHE_DIR)
'''
cells.append(markdown("### 13. Serial residual 실험 설정", "serial_config_title"))
cells.append(code(serial_config, "serial_config"))


serial_dl = r'''
SERIAL_EPOCH_ROWS = []


def serial_huber_loss(prediction, target, weight):
    return (_huber(prediction - target) * weight).sum() / weight.sum()


def serial_es_mask(frame):
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


def serial_xy(frame, cols, imputer, scaler, fit=False):
    sub = frame.loc[frame["target"].notna()]
    raw = sub[cols].to_numpy(dtype=np.float32)
    if fit:
        values = scaler.fit_transform(imputer.fit_transform(raw))
    else:
        values = scaler.transform(imputer.transform(raw))
    groups = sub["kpx_group"].map(GIDX).to_numpy(dtype=np.int64)
    capacity = sub["kpx_group"].map(GROUP_CAPACITY).to_numpy(dtype=float)
    target_cf = (sub["target"].to_numpy(dtype=float) / capacity).astype(np.float32)
    low_weight = np.where(target_cf < 0.10, 0.05, 1.0)
    availability_weight = avail_factor(sub["avail_strict"].to_numpy())
    weight = (low_weight * availability_weight).astype(np.float32)
    return values.astype(np.float32), groups, target_cf, weight


def train_serial_dl(
    X_train, group_train, y_train, weight_train,
    X_es, group_es, y_es, weight_es, seed,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MTMLP(
        X_train.shape[1],
        width=DL_CFG["width"],
        depth=DL_CFG["depth"],
        p_drop=DL_CFG["p_drop"],
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=DL_CFG["lr"], weight_decay=DL_CFG["wd"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=DL_CFG["epochs"]
    )

    Xt = torch.tensor(X_train, device=DEVICE)
    gt = torch.tensor(group_train, device=DEVICE)
    yt = torch.tensor(y_train, device=DEVICE)
    wt = torch.tensor(weight_train, device=DEVICE)
    Xe = torch.tensor(X_es, device=DEVICE)
    ge = torch.tensor(group_es, device=DEVICE)
    ye = torch.tensor(y_es, device=DEVICE)
    we = torch.tensor(weight_es, device=DEVICE)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    batch_size = 1024

    for epoch in range(DL_CFG["epochs"]):
        model.train()
        permutation = torch.randperm(len(Xt), device=DEVICE)
        for start in range(0, len(Xt), batch_size):
            idx = permutation[start:start + batch_size]
            optimizer.zero_grad()
            loss = serial_huber_loss(model(Xt[idx], gt[idx]), yt[idx], wt[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = serial_huber_loss(model(Xe, ge), ye, we).item()
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch + 1
            bad_epochs = 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= SERIAL_PATIENCE:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_epoch, best_loss


def serial_dl_predict(train_frame, query_frame, tag):
    validation_mask = serial_es_mask(train_frame)
    fit_frame = train_frame.loc[~validation_mask]
    es_frame = train_frame.loc[validation_mask]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    X_train, group_train, y_train, weight_train = serial_xy(
        fit_frame, DL_WIND_COLS, imputer, scaler, fit=True
    )
    X_es, group_es, y_es, weight_es = serial_xy(
        es_frame, DL_WIND_COLS, imputer, scaler, fit=False
    )
    X_query = scaler.transform(
        imputer.transform(query_frame[DL_WIND_COLS].to_numpy(dtype=np.float32))
    ).astype(np.float32)
    group_query = query_frame["kpx_group"].map(GIDX).to_numpy(dtype=np.int64)

    predictions = []
    for seed in range(SERIAL_DL_SEEDS):
        model, best_epoch, best_loss = train_serial_dl(
            X_train, group_train, y_train, weight_train,
            X_es, group_es, y_es, weight_es, seed,
        )
        predictions.append(predict(model, X_query, group_query))
        SERIAL_EPOCH_ROWS.append({
            "tag": tag,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "n_fit": len(fit_frame),
            "n_es": len(es_frame),
            "n_query": len(query_frame),
        })
        del model
        torch.cuda.empty_cache()
    print(tag, "best epochs:", SERIAL_EPOCH_ROWS[-SERIAL_DL_SEEDS:])
    return np.mean(predictions, axis=0)


def build_fold_base_predictions(fold_name, train_all, valid_all):
    path = SERIAL_CACHE_DIR / f"serial_base_{fold_name}.parquet"
    if path.exists():
        cached = pd.read_parquet(path)
        assert set(cached["role"].unique()) == {"train", "valid"}
        print("DL base cache:", path.name, cached.shape)
        return cached

    labeled = train_all.loc[train_all["target"].notna()].copy()
    train_signature = pd.Timestamp(labeled["forecast_kst_dtm"].max()).strftime("%Y%m%d%H%M")
    train_path = SERIAL_CACHE_DIR / f"serial_base_train_{train_signature}.parquet"

    if train_path.exists():
        train_base = pd.read_parquet(train_path)
        assert len(train_base) == len(labeled)
        assert np.isfinite(train_base["serial_base_cf"]).all()
        print("DL train crossfit cache:", train_path.name, train_base.shape)
    else:
        midpoint = labeled["forecast_kst_dtm"].quantile(0.50)
        first_half = labeled["forecast_kst_dtm"] <= midpoint
        train_prediction = np.full(len(labeled), np.nan, dtype=float)

        for fit_mask, query_mask, suffix in [
            (first_half, ~first_half, "early_to_late"),
            (~first_half, first_half, "late_to_early"),
        ]:
            train_prediction[np.flatnonzero(query_mask.to_numpy())] = serial_dl_predict(
                labeled.loc[fit_mask], labeled.loc[query_mask], f"{train_signature}:{suffix}"
            )
        assert np.isfinite(train_prediction).all()
        train_base = labeled[["kpx_group", "forecast_kst_dtm", "target"]].copy()
        train_base["serial_base_cf"] = train_prediction
        train_base["role"] = "train"
        train_base.to_parquet(train_path, index=False)

    valid_prediction = serial_dl_predict(
        labeled, valid_all, f"{fold_name}:train_to_valid"
    )
    valid_base = valid_all[["kpx_group", "forecast_kst_dtm", "target"]].copy()
    valid_base["serial_base_cf"] = valid_prediction
    valid_base["role"] = "valid"
    cached = pd.concat([train_base, valid_base], ignore_index=True)
    cached.insert(0, "fold", fold_name)
    cached.to_parquet(path, index=False)
    return cached
'''
cells.append(markdown("### 14. Huber DL base와 시간 2분할 교차적합", "serial_dl_title"))
cells.append(code(serial_dl, "serial_dl"))


serial_residual = r'''
def serial_lgb_predict(X_train, y_train, sample_weight, X_eval, seed, label):
    params = dict(SERIAL_LGB_PARAMS)
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


def residual_predict(train_group, eval_group, feature_cols, group, label):
    X_train, X_eval = prepare_matrices(train_group, eval_group, feature_cols)
    capacity = GROUP_CAPACITY[group]
    target_cf = train_group["target"].to_numpy(dtype=float) / capacity
    residual_target = target_cf - train_group["serial_base_cf"].to_numpy(dtype=float)
    sample_weight = make_sample_weight_scada(
        train_group["target"].to_numpy(dtype=float),
        capacity,
        train_group["avail_strict"],
    )
    predictions = [
        serial_lgb_predict(
            X_train, residual_target, sample_weight, X_eval, seed,
            f"{label}:g{group}:s{seed}",
        )
        for seed in range(SERIAL_GBM_SEEDS)
    ]
    return np.mean(predictions, axis=0)


def estimate_fixed_bias_map(train_frame, feature_cols, train_end):
    bias_data = {}
    for group in [1, 2, 3]:
        capacity = GROUP_CAPACITY[group]
        group_frame = train_frame.loc[
            (train_frame["kpx_group"] == group)
            & (train_frame["forecast_kst_dtm"] <= train_end)
            & train_frame["target"].notna()
        ].copy()
        first_date = group_frame["forecast_kst_dtm"].min()
        available_days = max((train_end - first_date).days, 1)
        holdout_days = min(int(available_days * 0.25), 180)
        if available_days - holdout_days < 90:
            holdout_days = max(0, available_days - 90)
        split_time = train_end - pd.Timedelta(days=holdout_days)
        fit_frame = group_frame.loc[group_frame["forecast_kst_dtm"] <= split_time]
        check_frame = group_frame.loc[group_frame["forecast_kst_dtm"] > split_time].copy()
        if fit_frame.empty or check_frame.empty:
            continue
        X_fit, X_check = prepare_matrices(fit_frame, check_frame, feature_cols)
        weight = make_sample_weight_scada(
            fit_frame["target"].to_numpy(dtype=float),
            capacity,
            fit_frame["avail_strict"],
        )
        prediction = serial_lgb_predict(
            X_fit, fit_frame["target"].to_numpy(dtype=float), weight,
            X_check, SEED, f"bias:g{group}",
        )
        check_frame["signed_error_rate"] = (
            prediction - check_frame["target"].to_numpy(dtype=float)
        ) / capacity
        check_frame = check_frame.loc[check_frame["target"] >= capacity * 0.10]
        if not check_frame.empty:
            bias_data[group] = check_frame[["ldaps_ws_10m", "signed_error_rate"]]
    return bias_map_from_data(
        bias_data, {group: COARSE_BINS for group in bias_data}
    )


def merge_base(frame, base_frame, role):
    keys = ["kpx_group", "forecast_kst_dtm"]
    base = base_frame.loc[base_frame["role"] == role, keys + ["serial_base_cf"]]
    merged = frame.merge(base, on=keys, how="inner", validate="one_to_one")
    assert len(merged) == len(base), f"base 병합 실패: {len(merged)} != {len(base)}"
    return merged
'''
cells.append(markdown("### 15. Optuna 설정 잔차 LightGBM", "serial_residual_title"))
cells.append(code(serial_residual, "serial_residual"))


serial_ablation = r'''
PREDICTION_CACHE_PATH = SERIAL_CACHE_DIR / "serial_fold_predictions.parquet"
BIAS_CACHE_PATH = SERIAL_CACHE_DIR / "serial_fold_bias.pkl"

prediction_parts = []
bias_maps = {}
if PREDICTION_CACHE_PATH.exists():
    prediction_parts = [pd.read_parquet(PREDICTION_CACHE_PATH)]
if BIAS_CACHE_PATH.exists():
    with open(BIAS_CACHE_PATH, "rb") as stream:
        bias_maps = pickle.load(stream)

completed_folds = set()
if prediction_parts:
    cached_predictions = prediction_parts[0]
    for fold_name in cached_predictions["fold"].unique():
        experiments = set(
            cached_predictions.loc[cached_predictions["fold"] == fold_name, "experiment"]
        )
        if experiments == set(EXPERIMENT_COLS):
            completed_folds.add(fold_name)

for fold_name, fold in FOLD_LIST:
    if fold_name in completed_folds:
        print("residual cache:", fold_name)
        continue
    started = time.time()
    train_all = train_df_tps.loc[
        train_df_tps["forecast_kst_dtm"] <= fold["train_end"]
    ].copy()
    valid_all = train_df_tps.loc[
        (train_df_tps["forecast_kst_dtm"] >= fold["valid_start"])
        & (train_df_tps["forecast_kst_dtm"] <= fold["valid_end"])
        & train_df_tps["target"].notna()
    ].copy()
    base_frame = build_fold_base_predictions(fold_name, train_all, valid_all)
    train_serial = merge_base(
        train_all.loc[train_all["target"].notna()], base_frame, "train"
    )
    valid_serial = merge_base(valid_all, base_frame, "valid")

    if fold_name not in bias_maps:
        if fold_name == "hold_2024full" and "fold_2024Q1" in bias_maps:
            bias_maps[fold_name] = bias_maps["fold_2024Q1"]
            print("bias map reuse: fold_2024Q1 -> hold_2024full")
        else:
            bias_maps[fold_name] = estimate_fixed_bias_map(
                train_df_tps, cols_c0, fold["train_end"]
            )

    fold_parts = []
    for experiment, feature_cols in EXPERIMENT_COLS.items():
        for group in [1, 2, 3]:
            train_group = train_serial.loc[train_serial["kpx_group"] == group].copy()
            valid_group = valid_serial.loc[
                valid_serial["kpx_group"] == group
            ].sort_values("forecast_kst_dtm").copy()
            residual_cf = residual_predict(
                train_group, valid_group, feature_cols, group,
                f"{fold_name}:{experiment}",
            )
            raw_cf = valid_group["serial_base_cf"].to_numpy(dtype=float) + residual_cf
            capacity = GROUP_CAPACITY[group]
            unscaled = postprocess_simple(
                raw_cf,
                capacity,
                valid_group["ldaps_ws_10m"].to_numpy(dtype=float),
                group,
                bias_maps[fold_name],
            )
            scaled = np.clip(unscaled * GROUP_SCALES[group], 0, capacity)
            fold_parts.append(pd.DataFrame({
                "fold": fold_name,
                "group": group,
                "forecast_kst_dtm": valid_group["forecast_kst_dtm"].to_numpy(),
                "target": valid_group["target"].to_numpy(dtype=float),
                "serial_base_cf": valid_group["serial_base_cf"].to_numpy(dtype=float),
                "residual_cf": residual_cf,
                "raw_cf": raw_cf,
                "pred_unscaled": unscaled,
                "pred_scaled": scaled,
                "experiment": experiment,
            }))
    fold_output = pd.concat(fold_parts, ignore_index=True)
    if prediction_parts and len(prediction_parts[0]):
        current = pd.concat([prediction_parts[0], fold_output], ignore_index=True)
    else:
        current = fold_output
    prediction_parts = [current]
    current.to_parquet(PREDICTION_CACHE_PATH, index=False)
    with open(BIAS_CACHE_PATH, "wb") as stream:
        pickle.dump(bias_maps, stream)
    if SERIAL_EPOCH_ROWS:
        pd.DataFrame(SERIAL_EPOCH_ROWS).to_csv(
            RUN_ROOT / "best_epochs.csv", index=False
        )
    print(f"{fold_name} complete: {(time.time()-started)/60:.1f}분")

serial_oof = prediction_parts[0]
expected = {name for name, _ in FOLD_LIST}
assert set(serial_oof["fold"].unique()) == expected
assert serial_oof[["target", "pred_unscaled", "pred_scaled"]].notna().all().all()
'''
cells.append(markdown("### 16. C0 대 C0+Flow serial ablation", "serial_ablation_title"))
cells.append(code(serial_ablation, "serial_ablation"))


serial_scoring = r'''
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


summary_rows, fold_rows, group_rows = [], [], []
for operation, pred_col in [
    ("unscaled", "pred_unscaled"),
    ("scaled", "pred_scaled"),
]:
    for experiment in EXPERIMENT_COLS:
        experiment_frame = serial_oof.loc[serial_oof["experiment"] == experiment]
        for partition, mask in [
            ("cv", experiment_frame["fold"] != "hold_2024full"),
            ("hold", experiment_frame["fold"] == "hold_2024full"),
        ]:
            partition_frame = experiment_frame.loc[mask]
            result = score_from_tidy(partition_frame, pred_col)
            summary_rows.append({
                "operation": operation,
                "experiment": experiment,
                "partition": partition,
                "Score": result[0],
                "1-NMAE": result[1],
                "FiCR": result[2],
            })
            for group in [1, 2, 3]:
                group_frame = partition_frame.loc[partition_frame["group"] == group]
                group_result = one_group_score(group_frame, pred_col, group)
                group_rows.append({
                    "operation": operation,
                    "experiment": experiment,
                    "partition": partition,
                    "group": group,
                    "Score": group_result[0],
                    "1-NMAE": group_result[1],
                    "FiCR": group_result[2],
                })
        for fold_name, fold_frame in experiment_frame.groupby("fold"):
            fold_result = score_from_tidy(fold_frame, pred_col)
            fold_rows.append({
                "operation": operation,
                "experiment": experiment,
                "fold": fold_name,
                "Score": fold_result[0],
                "1-NMAE": fold_result[1],
                "FiCR": fold_result[2],
            })

summary = pd.DataFrame(summary_rows)
fold_scores = pd.DataFrame(fold_rows)
group_scores = pd.DataFrame(group_rows)


def add_serial_deltas(table, keys):
    metrics = ["Score", "1-NMAE", "FiCR"]
    control = table.loc[table["experiment"] == "R0_CONTROL", keys + metrics].rename(
        columns={metric_name: f"control_{metric_name}" for metric_name in metrics}
    )
    result = table.merge(control, on=keys, how="left", validate="many_to_one")
    for metric_name in metrics:
        result[f"delta_{metric_name}"] = (
            result[metric_name] - result[f"control_{metric_name}"]
        )
    return result.drop(columns=[f"control_{metric_name}" for metric_name in metrics])


summary = add_serial_deltas(summary, ["operation", "partition"])
fold_scores = add_serial_deltas(fold_scores, ["operation", "fold"])
group_scores = add_serial_deltas(group_scores, ["operation", "partition", "group"])

primary = summary.loc[
    (summary["operation"] == PRIMARY_OPERATION)
    & (summary["experiment"] == PRIMARY_EXPERIMENT)
].set_index("partition")
primary_folds = fold_scores.loc[
    (fold_scores["operation"] == PRIMARY_OPERATION)
    & (fold_scores["experiment"] == PRIMARY_EXPERIMENT)
    & fold_scores["fold"].str.startswith("fold_")
]
unscaled_primary = summary.loc[
    (summary["operation"] == "unscaled")
    & (summary["experiment"] == PRIMARY_EXPERIMENT)
].set_index("partition")
positive_folds = int((primary_folds["delta_Score"] > 0).sum())

SERIAL_FLOW_GATE_PASS = bool(
    primary.loc["cv", "delta_Score"] >= SERIAL_PASS_CV_DELTA
    and positive_folds >= SERIAL_PASS_MIN_POSITIVE_FOLDS
    and primary.loc["hold", "delta_Score"] >= 0.0
    and primary.loc["cv", "delta_1-NMAE"] >= SERIAL_PASS_MIN_SUBMETRIC_DELTA
    and primary.loc["cv", "delta_FiCR"] > 0.0
    and unscaled_primary.loc["cv", "delta_Score"] >= 0.0
)

gate_decision = {
    "passed": SERIAL_FLOW_GATE_PASS,
    "primary_operation": PRIMARY_OPERATION,
    "primary_experiment": PRIMARY_EXPERIMENT,
    "cv_delta_score": float(primary.loc["cv", "delta_Score"]),
    "cv_delta_1_nmae": float(primary.loc["cv", "delta_1-NMAE"]),
    "cv_delta_ficr": float(primary.loc["cv", "delta_FiCR"]),
    "hold_delta_score": float(primary.loc["hold", "delta_Score"]),
    "unscaled_cv_delta_score": float(unscaled_primary.loc["cv", "delta_Score"]),
    "positive_cv_folds": positive_folds,
}

reference_result = {"available": REFERENCE_OOF_PATH.exists()}
if REFERENCE_OOF_PATH.exists():
    reference_oof = pd.read_parquet(REFERENCE_OOF_PATH)
    reference_f2 = reference_oof.loc[reference_oof["experiment"] == "F2_FLOW"]
    for partition, mask in [
        ("cv", reference_f2["fold"] != "hold_2024full"),
        ("hold", reference_f2["fold"] == "hold_2024full"),
    ]:
        result = score_from_tidy(reference_f2.loc[mask], "pred")
        reference_result[partition] = {
            "Score": result[0], "1-NMAE": result[1], "FiCR": result[2]
        }

run_config = {
    "experiment": "SERIAL_RESIDUAL_FLOW",
    "serial_version": SERIAL_VERSION,
    "independent_of_wonwoo_cache": True,
    "dl": {
        "inputs": DL_WIND_COLS,
        "loss": "symmetric_huber",
        "width": DL_CFG["width"],
        "depth": DL_CFG["depth"],
        "seeds": SERIAL_DL_SEEDS,
        "crossfit": "global_time_median_2way",
    },
    "residual_lgb": {**SERIAL_LGB_PARAMS, "seeds": SERIAL_GBM_SEEDS},
    "control_feature_count": len(EXPERIMENT_COLS["R0_CONTROL"]),
    "flow_feature_count": len(EXPERIMENT_COLS["R1_FLOW"]),
    "flow_features": flow_cols,
    "primary_operation": PRIMARY_OPERATION,
}

summary.to_csv(RUN_ROOT / "summary.csv", index=False)
fold_scores.to_csv(RUN_ROOT / "fold_scores.csv", index=False)
group_scores.to_csv(RUN_ROOT / "group_scores.csv", index=False)
serial_oof.to_parquet(RUN_ROOT / "oof_serial_predictions.parquet", index=False)
(RUN_ROOT / "gate_decision.json").write_text(
    json.dumps(gate_decision, ensure_ascii=False, indent=2), encoding="utf-8"
)
(RUN_ROOT / "reference_result.json").write_text(
    json.dumps(reference_result, ensure_ascii=False, indent=2), encoding="utf-8"
)
(RUN_ROOT / "run_config.json").write_text(
    json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
)

display(summary.round(6))
display(primary_folds.round(6))
display(
    group_scores.loc[
        (group_scores["operation"] == PRIMARY_OPERATION)
        & (group_scores["experiment"] == PRIMARY_EXPERIMENT)
    ].round(6)
)
print("Gate:", "PASS" if SERIAL_FLOW_GATE_PASS else "REJECT")
print(json.dumps(gate_decision, ensure_ascii=False, indent=2))
'''
cells.append(markdown("### 17. 지표 비교와 gate 판정", "serial_scoring_title"))
cells.append(code(serial_scoring, "serial_scoring"))


serial_final = r'''
SUBMISSION_PATHS = []
PRIMARY_SUBMISSION_NAME = (
    "submission_serial_flow_scaled.csv"
    if SERIAL_FLOW_GATE_PASS
    else "submission_serial_control_scaled.csv"
)


def build_final_base_predictions():
    path = SERIAL_CACHE_DIR / "serial_base_final.parquet"
    if path.exists():
        cached = pd.read_parquet(path)
        assert set(cached["role"].unique()) == {"train", "test"}
        print("final DL base cache:", cached.shape)
        return cached

    labeled = train_df_tps.loc[train_df_tps["target"].notna()].copy()
    midpoint = labeled["forecast_kst_dtm"].quantile(0.50)
    first_half = labeled["forecast_kst_dtm"] <= midpoint
    train_prediction = np.full(len(labeled), np.nan, dtype=float)
    for fit_mask, query_mask, suffix in [
        (first_half, ~first_half, "early_to_late"),
        (~first_half, first_half, "late_to_early"),
    ]:
        train_prediction[np.flatnonzero(query_mask.to_numpy())] = serial_dl_predict(
            labeled.loc[fit_mask], labeled.loc[query_mask], f"final:{suffix}"
        )
    assert np.isfinite(train_prediction).all()

    test_sorted = test_df_tps.sort_values(
        ["kpx_group", "forecast_kst_dtm"]
    ).copy()
    test_prediction = serial_dl_predict(labeled, test_sorted, "final:train_to_test")

    train_base = labeled[["kpx_group", "forecast_kst_dtm", "target"]].copy()
    train_base["serial_base_cf"] = train_prediction
    train_base["role"] = "train"
    train_base["forecast_id"] = ""
    test_base = test_sorted[
        ["forecast_id", "kpx_group", "forecast_kst_dtm"]
    ].copy()
    test_base["target"] = np.nan
    test_base["serial_base_cf"] = test_prediction
    test_base["role"] = "test"
    cached = pd.concat([train_base, test_base], ignore_index=True)
    cached.to_parquet(path, index=False)
    if SERIAL_EPOCH_ROWS:
        pd.DataFrame(SERIAL_EPOCH_ROWS).to_csv(
            RUN_ROOT / "best_epochs.csv", index=False
        )
    return cached


def make_serial_submission(components, cf_col, bias_map, scaled):
    parts = []
    for group in [1, 2, 3]:
        capacity = GROUP_CAPACITY[group]
        group_frame = components.loc[
            components["kpx_group"] == group
        ].sort_values("forecast_kst_dtm")
        prediction = postprocess_simple(
            group_frame[cf_col].to_numpy(dtype=float),
            capacity,
            group_frame["ldaps_ws_10m"].to_numpy(dtype=float),
            group,
            bias_map,
        )
        if scaled:
            prediction = np.clip(prediction * GROUP_SCALES[group], 0, capacity)
        parts.append(pd.DataFrame({
            "forecast_id": group_frame["forecast_id"].to_numpy(),
            "forecast_kst_dtm": group_frame["forecast_kst_dtm"].to_numpy(),
            "kpx_group": group,
            "pred": prediction,
        }))
    long = pd.concat(parts, ignore_index=True)
    submission = long.pivot(
        index=["forecast_id", "forecast_kst_dtm"],
        columns="kpx_group",
        values="pred",
    )
    submission.columns = [f"kpx_group_{int(group)}" for group in submission.columns]
    submission = submission.reset_index()
    submission["forecast_kst_dtm"] = pd.to_datetime(
        submission["forecast_kst_dtm"]
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    return sample_submission[["forecast_id"]].merge(
        submission, on="forecast_id", how="left"
    )


if not RUN_FINAL_MODEL:
    print("RUN_FINAL_MODEL=False: CV 결과만 저장하고 최종 학습은 생략합니다.")
else:
    if not SERIAL_FLOW_GATE_PASS:
        print("Flow gate 미통과: control만 최종 학습합니다.")
    final_base = build_final_base_predictions()
    train_final = merge_base(
        train_df_tps.loc[train_df_tps["target"].notna()], final_base, "train"
    )
    test_base = final_base.loc[final_base["role"] == "test", [
        "forecast_id", "kpx_group", "forecast_kst_dtm", "serial_base_cf"
    ]]
    test_final = test_df_tps.merge(
        test_base,
        on=["forecast_id", "kpx_group", "forecast_kst_dtm"],
        how="inner",
        validate="one_to_one",
    )
    assert len(test_final) == len(test_base)

    component_path = SERIAL_CACHE_DIR / "serial_final_components.parquet"
    bias_path = SERIAL_CACHE_DIR / "serial_final_bias.pkl"
    final_experiments = ["R0_CONTROL"]
    if SERIAL_FLOW_GATE_PASS:
        final_experiments.append("R1_FLOW")
    required_cf_cols = {
        "cf_control" if experiment == "R0_CONTROL" else "cf_flow"
        for experiment in final_experiments
    }
    cache_ready = component_path.exists() and bias_path.exists()
    if cache_ready:
        final_components = pd.read_parquet(component_path)
        cache_ready = required_cf_cols.issubset(final_components.columns)
    if cache_ready:
        with open(bias_path, "rb") as stream:
            final_bias_map = pickle.load(stream)
        print("final residual cache:", final_components.shape)
    else:
        final_bias_map = estimate_fixed_bias_map(
            train_df_tps, cols_c0, train_df_tps["forecast_kst_dtm"].max()
        )
        final_components = test_final[[
            "forecast_id", "forecast_kst_dtm", "kpx_group",
            "ldaps_ws_10m", "serial_base_cf",
        ]].copy()
        for experiment in final_experiments:
            feature_cols = EXPERIMENT_COLS[experiment]
            output_col = "cf_control" if experiment == "R0_CONTROL" else "cf_flow"
            final_components[output_col] = np.nan
            for group in [1, 2, 3]:
                train_group = train_final.loc[train_final["kpx_group"] == group].copy()
                test_group = test_final.loc[
                    test_final["kpx_group"] == group
                ].sort_values("forecast_kst_dtm").copy()
                residual_cf = residual_predict(
                    train_group, test_group, feature_cols, group,
                    f"final:{experiment}",
                )
                raw_cf = test_group["serial_base_cf"].to_numpy(dtype=float) + residual_cf
                final_components.loc[test_group.index, output_col] = raw_cf
        assert final_components[list(required_cf_cols)].notna().all().all()
        final_components.to_parquet(component_path, index=False)
        with open(bias_path, "wb") as stream:
            pickle.dump(final_bias_map, stream)

    output_specs = [
        ("cf_control", False, "submission_serial_control_unscaled.csv"),
        ("cf_control", True, "submission_serial_control_scaled.csv"),
    ]
    if SERIAL_FLOW_GATE_PASS:
        output_specs.extend([
            ("cf_flow", False, "submission_serial_flow_unscaled.csv"),
            ("cf_flow", True, "submission_serial_flow_scaled.csv"),
        ])
    for cf_col, scaled, file_name in output_specs:
        submission = make_serial_submission(
            final_components, cf_col, final_bias_map, scaled
        )
        output_path = RUN_ROOT / file_name
        submission.to_csv(output_path, index=False, encoding="utf-8-sig")
        SUBMISSION_PATHS.append(output_path)
        print("Saved:", output_path)
'''
cells.append(markdown("### 18. 최종 학습과 gate 기반 제출 생성", "serial_final_title"))
cells.append(code(serial_final, "serial_final"))


serial_verify = r'''
required_result_paths = [
    RUN_ROOT / "summary.csv",
    RUN_ROOT / "fold_scores.csv",
    RUN_ROOT / "group_scores.csv",
    RUN_ROOT / "gate_decision.json",
    RUN_ROOT / "reference_result.json",
    RUN_ROOT / "run_config.json",
    RUN_ROOT / "oof_serial_predictions.parquet",
    RUN_ROOT / "flow_feature_summary.csv",
]
assert all(path.exists() for path in required_result_paths)

for submission_path in SUBMISSION_PATHS:
    submission = pd.read_csv(submission_path, encoding="utf-8-sig")
    assert len(submission) == len(sample_submission) == 8760
    assert submission["forecast_id"].equals(sample_submission["forecast_id"])
    assert submission[TARGET_COLS].notna().all().all()
    for col in TARGET_COLS:
        assert submission[col].between(0, CAPACITY_KWH[col]).all()
    print(submission_path.name, submission.shape, "검증 통과")

print("결과 저장 경로:", RUN_ROOT)
print("Gate:", "PASS" if SERIAL_FLOW_GATE_PASS else "REJECT")
if RUN_FINAL_MODEL:
    print("primary submission candidate:", PRIMARY_SUBMISSION_NAME)
'''
cells.append(markdown("### 19. 결과 파일 검증", "serial_verify_title"))
cells.append(code(serial_verify, "serial_verify"))


cells.append(
    markdown(
        r'''
### 생성 파일

`/content/drive/MyDrive/BARAM2026/hyejin_serial_residual_flow`에 저장됩니다.

- `summary.csv`: serial control과 Flow 후보의 CV·HOLD 비교
- `fold_scores.csv`: fold별 변화
- `group_scores.csv`: 그룹별 변화
- `gate_decision.json`: 사전 정의 gate 판정
- `reference_result.json`: 이전 Flow OOF가 있으면 참고 점수 기록
- `best_epochs.csv`: Huber DL의 seed별 best epoch
- `oof_serial_predictions.parquet`: serial OOF 예측과 구성 요소
- control은 항상 unscaled/scaled submission 2개
- gate 통과 시 Flow unscaled/scaled submission 2개 추가

Flow gate 통과 시 주 후보는 `submission_serial_flow_scaled.csv`, 미통과 시 비교 후보는
`submission_serial_control_scaled.csv`입니다. 리더보드 제출 전 CV·HOLD 결과를 먼저 확인합니다.
''',
        "serial_outputs",
    )
)


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT)
