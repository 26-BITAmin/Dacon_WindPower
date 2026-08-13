import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "notebooks" / "anen_residual_champion.ipynb"
OUT = ROOT / "notebooks" / "deepset_distribution_diversity.ipynb"


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


setup_source = source_of("anen_setup").replace(
    "hyejin_anen_residual_champion", "hyejin_deepset_distribution_diversity"
)


cells = [
    markdown(
        r'''
## BARAM 2026 - DeepSets Distribution Diversity

챔피언형 anchor와 오차 상관이 다른 직접 확률회귀 파트너를 만드는 실험입니다.

- 순서가 임의인 AnEn 30개 멤버를 permutation-invariant DeepSets로 인코딩
- 269개 환경 피처를 별도 context encoder에 통과
- 발전량 CF의 5개 분위수를 직접 예측하며 잔차 LightGBM을 사용하지 않음
- anchor는 고정하고 새 모델 비중을 최대 20%로 제한
- full CV·leave-one-fold-out meta OOF에서 실제 external 챔피언까지 이길 때만 제출 파일 생성
- 직접 FiCR 위치 이동은 과거 검증에서 불안정했으므로 사용하지 않음
''',
        "div_intro",
    )
]


common_cells = [
    ("1. Colab·Drive 설정", "div_setup", setup_source),
    ("2. 기본 데이터셋 생성 함수", "div_base_dataset", source_of("anen_base_dataset")),
    ("3. 평가 지표·검증 fold", "div_metric", source_of("anen_metric")),
    ("4. 물리·시계열·공간 피처", "div_feature_blocks", source_of("anen_feature_blocks")),
    ("5. 다격자 편차 피처", "div_spatial_deviation", source_of("anen_spatial_deviation")),
    ("6. JMA 외부 예보", "div_jma", source_of("anen_jma")),
    ("7. ECMWF HRES 외부 예보", "div_ecmwf", source_of("anen_ecmwf")),
    ("8. no-snap 후처리", "div_postprocess", source_of("anen_postprocess")),
    ("9. SCADA 가용성 가중치", "div_availability", source_of("anen_availability")),
    ("10. 269개 환경 피처 준비", "div_data_prep", source_of("anen_data_prep")),
]

for title, cell_id, cell_source in common_cells:
    cells.append(markdown(f"### {title}", f"{cell_id}_title"))
    cells.append(code(cell_source, cell_id))


config_source = r'''
if IN_COLAB:
    DRIVE_ROOT = Path("/content/drive/MyDrive/BARAM2026")
else:
    DRIVE_ROOT = Path.cwd() / "runs"

ANCHOR_ROOT = DRIVE_ROOT / "hyejin_anen_residual_champion"
ANCHOR_CACHE_DIR = ANCHOR_ROOT / "cache" / "pd16_v1"
ANCHOR_OOF_PATH = ANCHOR_ROOT / "oof_anen_residual.parquet"
ANCHOR_SUBMISSION_PATH = ANCHOR_ROOT / "submission_anen_residual_pd16.csv"
REFERENCE_GBM_PATH = BASE_CACHE_DIR / "external_champion_gbm_v1.parquet"
REFERENCE_DL_PATH = BASE_CACHE_DIR / "external_champion_dl_v1.parquet"
REFERENCE_BIAS_PATH = BASE_CACHE_DIR / "external_champion_bias_v1.pkl"

DIVERSITY_VERSION = "deepset_q5_v1"
DIVERSITY_CACHE_DIR = CACHE_DIR / DIVERSITY_VERSION
DIVERSITY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

MEMBER_COLS = [f"mws{k}" for k in range(30)]
CONTEXT_COLS = list(cols_c0)
QUANTILE_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90]
GROUP_CAPACITY = {1: 21600, 2: 21600, 3: 21000}
GROUP_SCALES = {1: 0.98, 2: 0.98, 3: 0.97}

DIRECT_SEEDS = 3
DIRECT_EPOCHS = 100
DIRECT_PATIENCE = 15
DIRECT_BATCH_SIZE = 1024
ALPHA_GRID = [0.00, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
POINT_VARIANTS = ["q50", "q55"]
OPERATION_VARIANTS = ["unscaled", "scaled"]

PASS_MIN_CV_DELTA = 0.0007
PASS_MIN_META_DELTA = 0.0003
PASS_MIN_POSITIVE_FOLDS = 3
PASS_MIN_HOLD_DELTA = -0.0010
PASS_MIN_NMAE_DELTA = -0.0005
PASS_MIN_REFERENCE_CV_DELTA = 0.0003
PASS_MIN_REFERENCE_META_DELTA = 0.0003

assert len(CONTEXT_COLS) == 269
assert len(CONTEXT_COLS) == len(set(CONTEXT_COLS))
assert ANCHOR_OOF_PATH.exists(), (
    "먼저 anen_residual_champion.ipynb의 검증 셀까지 완료하세요: "
    + str(ANCHOR_OOF_PATH)
)
assert ANCHOR_SUBMISSION_PATH.exists(), (
    "먼저 수정된 anen_residual_champion.ipynb의 최종 anchor 생성 셀까지 "
    "완료하세요: " + str(ANCHOR_SUBMISSION_PATH)
)
assert all(path.exists() for path in [
    REFERENCE_GBM_PATH, REFERENCE_DL_PATH, REFERENCE_BIAS_PATH,
]), "기존 external 챔피언 OOF 캐시가 없습니다."

run_config = {
    "experiment": "DEEPSETS_DISTRIBUTION_DIVERSITY",
    "version": DIVERSITY_VERSION,
    "anchor": str(ANCHOR_OOF_PATH),
    "champion_reference": "external champion OOF cache",
    "architecture": "permutation-invariant DeepSets plus context encoder",
    "target": "direct capacity factor quantiles",
    "quantiles": QUANTILE_LEVELS,
    "uses_residual_gbm": False,
    "uses_ficr_shift": False,
    "max_partner_weight": max(ALPHA_GRID),
}
(RUN_ROOT / "run_config.json").write_text(
    json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
)

print("anchor:", ANCHOR_ROOT)
print("anchor cache:", ANCHOR_CACHE_DIR)
print("output:", RUN_ROOT)
print("context features:", len(CONTEXT_COLS))
'''
cells.append(markdown("### 11. 다양성 모델·anchor 설정", "div_config_title"))
cells.append(code(config_source, "div_config"))


member_source = r'''
KEY = ["kpx_group", "forecast_kst_dtm"]


def frame_signature(frame):
    return pd.Timestamp(frame["forecast_kst_dtm"].max()).strftime("%Y%m%d%H%M")


def load_train_members(train_window):
    path = ANCHOR_CACHE_DIR / f"anen_train_{frame_signature(train_window)}.parquet"
    assert path.exists(), f"anchor AnEn train cache 누락: {path}"
    members = pd.read_parquet(path)
    assert members[MEMBER_COLS].notna().all().all()
    return members


def load_query_members(fold_name, query_frame):
    cache_name = "hold_2024full" if fold_name == "fold_2024Q1" else fold_name
    path = ANCHOR_CACHE_DIR / f"anen_query_{cache_name}.parquet"
    assert path.exists(), f"anchor AnEn query cache 누락: {path}"
    members = pd.read_parquet(path)
    keys = query_frame[KEY]
    selected = keys.merge(members, on=KEY, how="left", validate="one_to_one")
    assert len(selected) == len(query_frame)
    assert selected[MEMBER_COLS].notna().all().all()
    return selected


def merge_members(frame, members):
    merged = frame.merge(members[KEY + MEMBER_COLS], on=KEY, how="inner", validate="one_to_one")
    assert len(merged) == len(frame)
    assert merged[MEMBER_COLS].notna().all().all()
    return merged


anchor_oof = pd.read_parquet(ANCHOR_OOF_PATH)
anchor_oof["forecast_kst_dtm"] = pd.to_datetime(anchor_oof["forecast_kst_dtm"])
required_anchor_cols = {
    "fold", "group", "forecast_kst_dtm", "target", "pred_scaled"
}
assert required_anchor_cols.issubset(anchor_oof.columns)
anchor_oof = anchor_oof[[
    "fold", "group", "forecast_kst_dtm", "target", "pred_scaled"
]].rename(columns={"pred_scaled": "anchor_pred"})
assert anchor_oof[["target", "anchor_pred"]].notna().all().all()
print("anchor OOF:", anchor_oof.shape)

# 실제 팀 챔피언 OOF를 같은 fold·후처리로 복원합니다.
reference_gbm = pd.read_parquet(REFERENCE_GBM_PATH)
reference_dl = pd.read_parquet(REFERENCE_DL_PATH)
reference_cache = reference_gbm.merge(
    reference_dl, on=["fold", "group", "forecast_kst_dtm"], how="left"
)
with open(REFERENCE_BIAS_PATH, "rb") as stream:
    reference_bias = pickle.load(stream)
reference_parts = []
for (fold_name, group), sub in reference_cache.groupby(
    ["fold", "group"], sort=False
):
    sub = sub.sort_values("forecast_kst_dtm").copy()
    capacity = GROUP_CAPACITY[int(group)]
    combined_cf = (
        0.65 * sub["gbm_cf"].to_numpy(dtype=float)
        + 0.35 * sub["dl_q60"].to_numpy(dtype=float)
    )
    prediction = postprocess_simple(
        combined_cf, capacity, sub["ws"].to_numpy(dtype=float), int(group),
        reference_bias[fold_name], clip_lo=0.16, alpha=0.40,
    )
    reference_parts.append(
        sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(
            external_pred=prediction
        )
    )
reference_oof = pd.concat(reference_parts, ignore_index=True)
reference_oof["forecast_kst_dtm"] = pd.to_datetime(
    reference_oof["forecast_kst_dtm"]
)
assert reference_oof[["target", "external_pred"]].notna().all().all()
print("external champion OOF:", reference_oof.shape)
'''
cells.append(markdown("### 12. Anchor OOF와 AnEn 멤버 캐시 정렬", "div_members_title"))
cells.append(code(member_source, "div_members"))


model_source = r'''
class DeepSetQuantileNet(nn.Module):
    def __init__(self, n_context, hidden=384, p_drop=0.15):
        super().__init__()
        self.member_encoder = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
        )
        self.member_attention = nn.Linear(64, 1)
        self.context_encoder = nn.Sequential(
            nn.Linear(n_context, 256), nn.BatchNorm1d(256), nn.SiLU(),
            nn.Dropout(p_drop), nn.Linear(256, 128), nn.SiLU(),
        )
        self.group_embedding = nn.Embedding(3, 8)
        self.trunk = nn.Sequential(
            nn.Linear(64 * 4 + 128 + 8, hidden), nn.BatchNorm1d(hidden),
            nn.SiLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden),
            nn.SiLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden),
            nn.SiLU(), nn.Dropout(p_drop),
        )
        self.quantile_head = nn.Linear(hidden, len(QUANTILE_LEVELS))

    def forward(self, members, context, group):
        encoded = self.member_encoder(members.unsqueeze(-1))
        pooled_mean = encoded.mean(dim=1)
        pooled_std = encoded.std(dim=1, unbiased=False)
        pooled_max = encoded.max(dim=1).values
        attention = torch.softmax(self.member_attention(encoded).squeeze(-1), dim=1)
        pooled_attention = (encoded * attention.unsqueeze(-1)).sum(dim=1)
        set_representation = torch.cat(
            [pooled_mean, pooled_std, pooled_max, pooled_attention], dim=1
        )
        context_representation = self.context_encoder(context)
        hidden = self.trunk(torch.cat([
            set_representation, context_representation,
            self.group_embedding(group),
        ], dim=1))
        raw = self.quantile_head(hidden)
        return torch.sort(raw, dim=1).values


QUANTILE_TENSOR = torch.tensor(QUANTILE_LEVELS, dtype=torch.float32, device=DEVICE)


def distribution_loss(prediction, target, weight):
    error = target[:, None] - prediction
    pinball = torch.maximum(
        QUANTILE_TENSOR[None, :] * error,
        (QUANTILE_TENSOR[None, :] - 1.0) * error,
    ).mean(dim=1)
    median_error = prediction[:, 2] - target
    abs_error = torch.abs(median_error)
    huber = torch.where(
        abs_error <= 0.05,
        0.5 * median_error ** 2,
        0.05 * (abs_error - 0.025),
    )
    row_loss = 0.45 * pinball + 0.55 * huber
    return (row_loss * weight).sum() / weight.sum()


def seasonal_es_mask(frame):
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
        mask = labeled & (dt >= times[-n_times])
    assert mask.any() and (labeled & ~mask).any()
    return mask


def fit_preprocessors(fit_frame):
    context_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    context_scaler = StandardScaler()
    context_scaler.fit(
        context_imputer.fit_transform(fit_frame[CONTEXT_COLS].to_numpy(dtype=np.float32))
    )
    members = fit_frame[MEMBER_COLS].to_numpy(dtype=np.float32)
    member_mean = float(np.nanmean(members))
    member_std = max(float(np.nanstd(members)), 1e-6)
    return context_imputer, context_scaler, member_mean, member_std


def make_arrays(frame, preprocessors, require_target):
    context_imputer, context_scaler, member_mean, member_std = preprocessors
    members = (
        frame[MEMBER_COLS].to_numpy(dtype=np.float32) - member_mean
    ) / member_std
    context = context_scaler.transform(
        context_imputer.transform(frame[CONTEXT_COLS].to_numpy(dtype=np.float32))
    ).astype(np.float32)
    groups = frame["kpx_group"].map({1: 0, 2: 1, 3: 2}).to_numpy(dtype=np.int64)
    if not require_target:
        return members, context, groups
    capacity = frame["kpx_group"].map(GROUP_CAPACITY).to_numpy(dtype=float)
    target = (frame["target"].to_numpy(dtype=float) / capacity).astype(np.float32)
    low_weight = np.where(target < 0.10, 0.05, 1.0)
    availability_weight = avail_factor(frame["avail_strict"].to_numpy())
    weight = (low_weight * availability_weight).astype(np.float32)
    return members, context, groups, target, weight


def tensor_tuple(arrays):
    members, context, groups, target, weight = arrays
    return (
        torch.tensor(members, device=DEVICE),
        torch.tensor(context, device=DEVICE),
        torch.tensor(groups, device=DEVICE),
        torch.tensor(target, device=DEVICE),
        torch.tensor(weight, device=DEVICE),
    )


def train_direct_model(fit_arrays, es_arrays, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DeepSetQuantileNet(len(CONTEXT_COLS)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=DIRECT_EPOCHS
    )
    members, context, groups, target, weight = tensor_tuple(fit_arrays)
    es_members, es_context, es_groups, es_target, es_weight = tensor_tuple(es_arrays)
    best_loss, best_epoch, bad_epochs, best_state = float("inf"), 0, 0, None

    for epoch in range(DIRECT_EPOCHS):
        model.train()
        permutation = torch.randperm(len(members), device=DEVICE)
        for start in range(0, len(members), DIRECT_BATCH_SIZE):
            idx = permutation[start:start + DIRECT_BATCH_SIZE]
            optimizer.zero_grad()
            prediction = model(members[idx], context[idx], groups[idx])
            loss = distribution_loss(prediction, target[idx], weight[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            es_prediction = model(es_members, es_context, es_groups)
            validation_loss = distribution_loss(
                es_prediction, es_target, es_weight
            ).item()
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch + 1
            bad_epochs = 0
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= DIRECT_PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_epoch, best_loss


@torch.no_grad()
def predict_direct_model(model, arrays, batch_size=8192):
    members, context, groups = arrays
    model.eval()
    output = []
    for start in range(0, len(members), batch_size):
        stop = start + batch_size
        output.append(model(
            torch.tensor(members[start:stop], device=DEVICE),
            torch.tensor(context[start:stop], device=DEVICE),
            torch.tensor(groups[start:stop], device=DEVICE),
        ).cpu().numpy())
    return np.concatenate(output, axis=0)


DIRECT_EPOCH_ROWS = []


def direct_predict(train_frame, query_frame, tag):
    validation_mask = seasonal_es_mask(train_frame)
    fit_frame = train_frame.loc[~validation_mask].copy()
    es_frame = train_frame.loc[validation_mask].copy()
    preprocessors = fit_preprocessors(fit_frame)
    fit_arrays = make_arrays(fit_frame, preprocessors, require_target=True)
    es_arrays = make_arrays(es_frame, preprocessors, require_target=True)
    query_arrays = make_arrays(query_frame, preprocessors, require_target=False)

    predictions = []
    for seed in range(DIRECT_SEEDS):
        model, best_epoch, best_loss = train_direct_model(
            fit_arrays, es_arrays, seed
        )
        predictions.append(predict_direct_model(model, query_arrays))
        DIRECT_EPOCH_ROWS.append({
            "tag": tag, "seed": seed, "best_epoch": best_epoch,
            "best_loss": best_loss, "n_fit": len(fit_frame),
            "n_es": len(es_frame), "n_query": len(query_frame),
        })
        del model
        torch.cuda.empty_cache()
    prediction = np.mean(predictions, axis=0)
    prediction.sort(axis=1)
    print(tag, "best epochs:", [row["best_epoch"] for row in DIRECT_EPOCH_ROWS[-3:]])
    return prediction
'''
cells.append(markdown("### 13. Permutation-invariant DeepSets 확률회귀", "div_model_title"))
cells.append(code(model_source, "div_model"))


validation_source = r'''
FOLD_MAP = {name: fold for name, fold in FOLD_LIST}


def predict_fold(fold_name):
    path = DIVERSITY_CACHE_DIR / f"direct_{fold_name}.parquet"
    if path.exists():
        cached = pd.read_parquet(path)
        assert cached[["q10", "q25", "q50", "q75", "q90"]].notna().all().all()
        print("direct cache:", path.name, cached.shape)
        return cached

    fold = FOLD_MAP[fold_name]
    train_window = train_df_tps.loc[
        train_df_tps["forecast_kst_dtm"] <= fold["train_end"]
    ].copy()
    valid_window = train_df_tps.loc[
        (train_df_tps["forecast_kst_dtm"] >= fold["valid_start"])
        & (train_df_tps["forecast_kst_dtm"] <= fold["valid_end"])
        & train_df_tps["target"].notna()
    ].copy()
    train_members = load_train_members(train_window)
    valid_members = load_query_members(fold_name, valid_window)
    train_model_frame = merge_members(
        train_window.loc[train_window["target"].notna()].copy(), train_members
    )
    valid_model_frame = merge_members(valid_window, valid_members)
    quantiles = direct_predict(train_model_frame, valid_model_frame, fold_name)
    output = valid_model_frame[KEY + ["target", "ldaps_ws_10m"]].copy()
    output["group"] = output.pop("kpx_group")
    for index, column in enumerate(["q10", "q25", "q50", "q75", "q90"]):
        output[column] = quantiles[:, index]
    output["fold"] = fold_name
    output.to_parquet(path, index=False)
    if DIRECT_EPOCH_ROWS:
        pd.DataFrame(DIRECT_EPOCH_ROWS).drop_duplicates(
            ["tag", "seed"], keep="last"
        ).to_csv(RUN_ROOT / "best_epochs.csv", index=False)
    return output


# Q1과 HOLD는 학습창이 같으므로 HOLD 모델의 2024 전체 예측을 Q1에 재사용합니다.
direct_parts = [
    predict_fold("hold_2024full"),
    predict_fold("fold_2024Q2"),
    predict_fold("fold_2024Q3"),
    predict_fold("fold_2024Q4"),
]
hold_direct = direct_parts[0]
q1_fold = FOLD_MAP["fold_2024Q1"]
q1_direct = hold_direct.loc[
    (hold_direct["forecast_kst_dtm"] >= q1_fold["valid_start"])
    & (hold_direct["forecast_kst_dtm"] <= q1_fold["valid_end"])
].copy()
q1_direct["fold"] = "fold_2024Q1"
direct_parts.append(q1_direct)
direct_oof = pd.concat(direct_parts, ignore_index=True)


def add_partner_predictions(frame):
    parts = []
    for (fold_name, group), group_frame in frame.groupby(["fold", "group"], sort=False):
        group_frame = group_frame.sort_values("forecast_kst_dtm").copy()
        capacity = GROUP_CAPACITY[int(group)]
        group_frame["q55"] = group_frame["q50"] + 0.20 * (
            group_frame["q75"] - group_frame["q50"]
        )
        for point in POINT_VARIANTS:
            raw = group_frame[point].to_numpy(dtype=float) * capacity
            smoothed = apply_rolling_smoothing(
                pd.Series(raw), window=3, alpha=0.40
            ).to_numpy()
            unscaled = np.clip(smoothed, capacity * 0.16, capacity)
            group_frame[f"partner_{point}_unscaled"] = unscaled
            group_frame[f"partner_{point}_scaled"] = np.clip(
                unscaled * GROUP_SCALES[int(group)], 0, capacity
            )
        parts.append(group_frame)
    return pd.concat(parts, ignore_index=True)


direct_oof = add_partner_predictions(direct_oof)
model_frame = anchor_oof.merge(
    direct_oof,
    on=["fold", "group", "forecast_kst_dtm"],
    how="inner", validate="one_to_one", suffixes=("_anchor", "_direct"),
)
assert len(model_frame) == len(anchor_oof) == len(direct_oof)
assert np.allclose(
    model_frame["target_anchor"], model_frame["target_direct"], equal_nan=True
)
model_frame = model_frame.rename(columns={"target_anchor": "target"}).drop(
    columns="target_direct"
)
model_frame = model_frame.merge(
    reference_oof,
    on=["fold", "group", "forecast_kst_dtm"],
    how="inner", validate="one_to_one", suffixes=("", "_reference"),
)
assert len(model_frame) == len(anchor_oof) == len(reference_oof)
assert np.allclose(
    model_frame["target"], model_frame["target_reference"], equal_nan=True
)
model_frame = model_frame.drop(columns="target_reference")
essential_oof_columns = [
    "target", "anchor_pred", "external_pred",
    "q10", "q25", "q50", "q75", "q90",
] + [
    f"partner_{point}_{operation}"
    for point in POINT_VARIANTS
    for operation in OPERATION_VARIANTS
]
assert model_frame[essential_oof_columns].notna().all().all()
model_frame.to_parquet(RUN_ROOT / "oof_deepset_diversity.parquet", index=False)
print("diversity OOF:", model_frame.shape)
'''
cells.append(markdown("### 14. 4-fold·HOLD 직접 확률예측", "div_validation_title"))
cells.append(code(validation_source, "div_validation"))


selection_source = r'''
CV_FOLDS = [fold["name"] for fold in FOLDS_Q]
HOLD_NAME = "hold_2024full"


def official_score(frame, prediction):
    scored = frame[["fold", "group", "forecast_kst_dtm", "target"]].copy()
    scored["prediction"] = np.asarray(prediction, dtype=float)
    return score_from_tidy(scored, "prediction")


def candidate_prediction(frame, point, operation, alpha):
    partner = frame[f"partner_{point}_{operation}"].to_numpy(dtype=float)
    anchor = frame["anchor_pred"].to_numpy(dtype=float)
    return anchor + alpha * (partner - anchor)


candidate_specs = []
for point in POINT_VARIANTS:
    for operation in OPERATION_VARIANTS:
        for alpha in ALPHA_GRID:
            if alpha == 0.0 and (point != "q50" or operation != "unscaled"):
                continue
            candidate_specs.append((point, operation, alpha))


def candidate_table(frame, folds):
    selected_frame = frame.loc[frame["fold"].isin(folds)]
    anchor_score = official_score(selected_frame, selected_frame["anchor_pred"])
    rows = []
    for point, operation, alpha in candidate_specs:
        prediction = candidate_prediction(selected_frame, point, operation, alpha)
        result = official_score(selected_frame, prediction)
        fold_deltas = []
        for fold_name in folds:
            fold_frame = selected_frame.loc[selected_frame["fold"] == fold_name]
            fold_result = official_score(
                fold_frame,
                candidate_prediction(fold_frame, point, operation, alpha),
            )
            fold_anchor = official_score(fold_frame, fold_frame["anchor_pred"])
            fold_deltas.append(fold_result[0] - fold_anchor[0])
        rows.append({
            "point": point, "operation": operation, "alpha": alpha,
            "Score": result[0], "1-NMAE": result[1], "FiCR": result[2],
            "delta_Score": result[0] - anchor_score[0],
            "delta_1-NMAE": result[1] - anchor_score[1],
            "delta_FiCR": result[2] - anchor_score[2],
            "positive_folds": int(np.sum(np.asarray(fold_deltas) > 0)),
            "min_fold_delta": float(np.min(fold_deltas)),
        })
    return pd.DataFrame(rows).sort_values(
        ["Score", "delta_1-NMAE"], ascending=False
    ).reset_index(drop=True)


candidate_scores = candidate_table(model_frame, CV_FOLDS)
eligible = candidate_scores.loc[
    (candidate_scores["alpha"] > 0)
    & (candidate_scores["positive_folds"] >= PASS_MIN_POSITIVE_FOLDS)
    & (candidate_scores["delta_1-NMAE"] >= PASS_MIN_NMAE_DELTA)
    & (candidate_scores["delta_FiCR"] > 0)
]
selected_row = (
    eligible.iloc[0] if not eligible.empty
    else candidate_scores.loc[candidate_scores["alpha"] == 0].iloc[0]
)
SELECTED_SPEC = {
    "point": selected_row["point"],
    "operation": selected_row["operation"],
    "alpha": float(selected_row["alpha"]),
}

# Nested leave-one-fold-out: 나머지 3개 fold로 사양을 선택해 남은 fold에 적용합니다.
meta_parts, meta_selection_rows = [], []
for heldout_fold in CV_FOLDS:
    train_folds = [fold for fold in CV_FOLDS if fold != heldout_fold]
    train_table = candidate_table(model_frame, train_folds)
    train_eligible = train_table.loc[
        (train_table["alpha"] > 0)
        & (train_table["positive_folds"] >= 2)
        & (train_table["delta_1-NMAE"] >= PASS_MIN_NMAE_DELTA)
        & (train_table["delta_FiCR"] > 0)
    ]
    chosen = (
        train_eligible.iloc[0] if not train_eligible.empty
        else train_table.loc[train_table["alpha"] == 0].iloc[0]
    )
    heldout = model_frame.loc[model_frame["fold"] == heldout_fold].copy()
    heldout["meta_pred"] = candidate_prediction(
        heldout, chosen["point"], chosen["operation"], float(chosen["alpha"])
    )
    meta_parts.append(heldout)
    meta_selection_rows.append({
        "heldout_fold": heldout_fold,
        "point": chosen["point"], "operation": chosen["operation"],
        "alpha": float(chosen["alpha"]),
        "train_delta_score": float(chosen["delta_Score"]),
    })

meta_oof = pd.concat(meta_parts, ignore_index=True)
meta_anchor_score = official_score(meta_oof, meta_oof["anchor_pred"])
meta_candidate_score = official_score(meta_oof, meta_oof["meta_pred"])
meta_reference_score = official_score(meta_oof, meta_oof["external_pred"])

cv_frame = model_frame.loc[model_frame["fold"].isin(CV_FOLDS)].copy()
cv_frame["selected_pred"] = candidate_prediction(
    cv_frame, **SELECTED_SPEC
)
hold_frame = model_frame.loc[model_frame["fold"] == HOLD_NAME].copy()
hold_frame["selected_pred"] = candidate_prediction(
    hold_frame, **SELECTED_SPEC
)
anchor_cv = official_score(cv_frame, cv_frame["anchor_pred"])
candidate_cv = official_score(cv_frame, cv_frame["selected_pred"])
reference_cv = official_score(cv_frame, cv_frame["external_pred"])
anchor_hold = official_score(hold_frame, hold_frame["anchor_pred"])
candidate_hold = official_score(hold_frame, hold_frame["selected_pred"])
reference_hold = official_score(hold_frame, hold_frame["external_pred"])

fold_rows = []
for fold_name in CV_FOLDS:
    fold_frame = cv_frame.loc[cv_frame["fold"] == fold_name]
    anchor_result = official_score(fold_frame, fold_frame["anchor_pred"])
    candidate_result = official_score(fold_frame, fold_frame["selected_pred"])
    reference_result = official_score(fold_frame, fold_frame["external_pred"])
    fold_rows.append({
        "fold": fold_name,
        "anchor_score": anchor_result[0],
        "reference_score": reference_result[0],
        "candidate_score": candidate_result[0],
        "delta_score": candidate_result[0] - anchor_result[0],
        "delta_vs_reference": candidate_result[0] - reference_result[0],
    })
fold_scores = pd.DataFrame(fold_rows)
positive_folds = int((fold_scores["delta_score"] > 0).sum())
positive_reference_folds = int(
    (fold_scores["delta_vs_reference"] > 0).sum()
)

partner_column = f"partner_{SELECTED_SPEC['point']}_{SELECTED_SPEC['operation']}"
partner_correlation = float(
    cv_frame["anchor_pred"].corr(cv_frame[partner_column])
)
cv_delta = float(candidate_cv[0] - anchor_cv[0])
meta_delta = float(meta_candidate_score[0] - meta_anchor_score[0])
hold_delta = float(candidate_hold[0] - anchor_hold[0])
reference_cv_delta = float(candidate_cv[0] - reference_cv[0])
reference_meta_delta = float(meta_candidate_score[0] - meta_reference_score[0])
reference_hold_delta = float(candidate_hold[0] - reference_hold[0])

DIVERSITY_GATE_PASS = bool(
    SELECTED_SPEC["alpha"] > 0
    and cv_delta >= PASS_MIN_CV_DELTA
    and meta_delta >= PASS_MIN_META_DELTA
    and positive_folds >= PASS_MIN_POSITIVE_FOLDS
    and hold_delta >= PASS_MIN_HOLD_DELTA
    and candidate_cv[1] - anchor_cv[1] >= PASS_MIN_NMAE_DELTA
    and candidate_cv[2] - anchor_cv[2] > 0
    and reference_cv_delta >= PASS_MIN_REFERENCE_CV_DELTA
    and reference_meta_delta >= PASS_MIN_REFERENCE_META_DELTA
    and positive_reference_folds >= PASS_MIN_POSITIVE_FOLDS
    and reference_hold_delta >= PASS_MIN_HOLD_DELTA
    and candidate_cv[1] - reference_cv[1] >= PASS_MIN_NMAE_DELTA
    and candidate_cv[2] - reference_cv[2] > 0
)

summary = pd.DataFrame([
    {"partition": "cv", "model": "external_champion", "Score": reference_cv[0],
     "1-NMAE": reference_cv[1], "FiCR": reference_cv[2]},
    {"partition": "cv", "model": "anchor", "Score": anchor_cv[0],
     "1-NMAE": anchor_cv[1], "FiCR": anchor_cv[2]},
    {"partition": "cv", "model": "diversity_blend", "Score": candidate_cv[0],
     "1-NMAE": candidate_cv[1], "FiCR": candidate_cv[2]},
    {"partition": "meta_cv", "model": "external_champion", "Score": meta_reference_score[0],
     "1-NMAE": meta_reference_score[1], "FiCR": meta_reference_score[2]},
    {"partition": "meta_cv", "model": "anchor", "Score": meta_anchor_score[0],
     "1-NMAE": meta_anchor_score[1], "FiCR": meta_anchor_score[2]},
    {"partition": "meta_cv", "model": "diversity_blend", "Score": meta_candidate_score[0],
     "1-NMAE": meta_candidate_score[1], "FiCR": meta_candidate_score[2]},
    {"partition": "hold", "model": "external_champion", "Score": reference_hold[0],
     "1-NMAE": reference_hold[1], "FiCR": reference_hold[2]},
    {"partition": "hold", "model": "anchor", "Score": anchor_hold[0],
     "1-NMAE": anchor_hold[1], "FiCR": anchor_hold[2]},
    {"partition": "hold", "model": "diversity_blend", "Score": candidate_hold[0],
     "1-NMAE": candidate_hold[1], "FiCR": candidate_hold[2]},
])

gate_decision = {
    "passed": DIVERSITY_GATE_PASS,
    "recommendation": "SUBMIT" if DIVERSITY_GATE_PASS else "DO_NOT_SUBMIT",
    "selected_spec": SELECTED_SPEC,
    "cv_delta_score": cv_delta,
    "meta_delta_score": meta_delta,
    "hold_delta_score": hold_delta,
    "cv_delta_1_nmae": float(candidate_cv[1] - anchor_cv[1]),
    "cv_delta_ficr": float(candidate_cv[2] - anchor_cv[2]),
    "positive_cv_folds": positive_folds,
    "reference_cv_delta_score": reference_cv_delta,
    "reference_meta_delta_score": reference_meta_delta,
    "reference_hold_delta_score": reference_hold_delta,
    "reference_cv_delta_1_nmae": float(candidate_cv[1] - reference_cv[1]),
    "reference_cv_delta_ficr": float(candidate_cv[2] - reference_cv[2]),
    "positive_reference_folds": positive_reference_folds,
    "partner_anchor_correlation": partner_correlation,
    "thresholds": {
        "min_cv_delta": PASS_MIN_CV_DELTA,
        "min_meta_delta": PASS_MIN_META_DELTA,
        "min_positive_folds": PASS_MIN_POSITIVE_FOLDS,
        "min_hold_delta": PASS_MIN_HOLD_DELTA,
        "min_nmae_delta": PASS_MIN_NMAE_DELTA,
        "min_reference_cv_delta": PASS_MIN_REFERENCE_CV_DELTA,
        "min_reference_meta_delta": PASS_MIN_REFERENCE_META_DELTA,
    },
}

candidate_scores.to_csv(RUN_ROOT / "candidate_scores.csv", index=False)
pd.DataFrame(meta_selection_rows).to_csv(RUN_ROOT / "meta_selection.csv", index=False)
summary.to_csv(RUN_ROOT / "summary.csv", index=False)
fold_scores.to_csv(RUN_ROOT / "fold_scores.csv", index=False)
(RUN_ROOT / "gate_decision.json").write_text(
    json.dumps(gate_decision, ensure_ascii=False, indent=2), encoding="utf-8"
)

display(candidate_scores.head(15).round(6))
display(pd.DataFrame(meta_selection_rows).round(6))
display(summary.round(6))
display(fold_scores.round(6))
print("Gate:", "PASS" if DIVERSITY_GATE_PASS else "REJECT")
print(json.dumps(gate_decision, ensure_ascii=False, indent=2))
'''
cells.append(markdown("### 15. Nested OOF 다양성 blend 선택", "div_selection_title"))
cells.append(code(selection_source, "div_selection"))


final_source = r'''
FINAL_QUANTILE_PATH = DIVERSITY_CACHE_DIR / "final_test_quantiles.parquet"
PARTNER_SUBMISSION_PATH = RUN_ROOT / "submission_deepset_partner_diagnostic.csv"
BLEND_SUBMISSION_PATH = RUN_ROOT / "submission_deepset_diversity_blend.csv"


def build_final_quantiles():
    if FINAL_QUANTILE_PATH.exists():
        cached = pd.read_parquet(FINAL_QUANTILE_PATH)
        assert len(cached) == len(test_df_tps)
        print("final direct cache:", cached.shape)
        return cached

    train_window = train_df_tps.copy()
    train_members = load_train_members(train_window)
    test_member_path = ANCHOR_CACHE_DIR / "anen_query_final_test.parquet"
    assert test_member_path.exists(), (
        "anchor의 최종 학습 셀을 먼저 완료해야 합니다: " + str(test_member_path)
    )
    test_members = pd.read_parquet(test_member_path)
    train_model_frame = merge_members(
        train_window.loc[train_window["target"].notna()].copy(), train_members
    )
    test_model_frame = merge_members(test_df_tps.copy(), test_members)
    quantiles = direct_predict(train_model_frame, test_model_frame, "final_test")
    output = test_model_frame[[
        "forecast_id", "forecast_kst_dtm", "kpx_group", "ldaps_ws_10m"
    ]].copy()
    for index, column in enumerate(["q10", "q25", "q50", "q75", "q90"]):
        output[column] = quantiles[:, index]
    output.to_parquet(FINAL_QUANTILE_PATH, index=False)
    if DIRECT_EPOCH_ROWS:
        pd.DataFrame(DIRECT_EPOCH_ROWS).drop_duplicates(
            ["tag", "seed"], keep="last"
        ).to_csv(RUN_ROOT / "best_epochs.csv", index=False)
    return output


def make_partner_submission(quantile_frame, point, operation):
    frame = quantile_frame.copy()
    frame["q55"] = frame["q50"] + 0.20 * (frame["q75"] - frame["q50"])
    parts = []
    for group in [1, 2, 3]:
        group_frame = frame.loc[frame["kpx_group"] == group].sort_values(
            "forecast_kst_dtm"
        )
        capacity = GROUP_CAPACITY[group]
        raw = group_frame[point].to_numpy(dtype=float) * capacity
        prediction = apply_rolling_smoothing(
            pd.Series(raw), window=3, alpha=0.40
        ).to_numpy()
        prediction = np.clip(prediction, capacity * 0.16, capacity)
        if operation == "scaled":
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


if not DIVERSITY_GATE_PASS:
    print("Gate REJECT: 최종 테스트 모델과 blend를 생성하지 않습니다.")
else:
    assert ANCHOR_SUBMISSION_PATH.exists(), (
        "anchor 제출 파일이 없습니다. 수정된 anen_residual_champion.ipynb의 "
        "최종 셀까지 실행하세요: " + str(ANCHOR_SUBMISSION_PATH)
    )
    anchor_submission = pd.read_csv(
        ANCHOR_SUBMISSION_PATH, encoding="utf-8-sig"
    )
    assert anchor_submission.shape == sample_submission.shape == (8760, 5)
    assert list(anchor_submission.columns) == list(sample_submission.columns)
    assert anchor_submission["forecast_id"].equals(sample_submission["forecast_id"])

    final_quantiles = build_final_quantiles()
    partner_submission = make_partner_submission(
        final_quantiles, SELECTED_SPEC["point"], SELECTED_SPEC["operation"]
    )
    partner_submission.to_csv(
        PARTNER_SUBMISSION_PATH, index=False, encoding="utf-8-sig"
    )

    blend_submission = anchor_submission.copy()
    alpha = SELECTED_SPEC["alpha"]
    delta_rows = []
    for group in [1, 2, 3]:
        column = f"kpx_group_{group}"
        anchor_values = anchor_submission[column].to_numpy(dtype=float)
        partner_values = partner_submission[column].to_numpy(dtype=float)
        blend_values = anchor_values + alpha * (partner_values - anchor_values)
        blend_submission[column] = np.clip(
            blend_values, 0, GROUP_CAPACITY[group]
        )
        delta_rows.append({
            "group": group, "alpha": alpha,
            "anchor_partner_correlation": float(
                np.corrcoef(anchor_values, partner_values)[0, 1]
            ),
            "mean_delta_kwh": float((blend_values - anchor_values).mean()),
            "mean_abs_delta_kwh": float(np.abs(blend_values - anchor_values).mean()),
            "max_abs_delta_kwh": float(np.abs(blend_values - anchor_values).max()),
        })
    blend_submission.to_csv(
        BLEND_SUBMISSION_PATH, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(delta_rows).to_csv(
        RUN_ROOT / "submission_delta_summary.csv", index=False
    )
    print("Saved:", BLEND_SUBMISSION_PATH)
'''
cells.append(markdown("### 16. Gate 기반 최종 DeepSets·blend 생성", "div_final_title"))
cells.append(code(final_source, "div_final"))


verify_source = r'''
required_paths = [
    RUN_ROOT / "run_config.json",
    RUN_ROOT / "candidate_scores.csv",
    RUN_ROOT / "meta_selection.csv",
    RUN_ROOT / "summary.csv",
    RUN_ROOT / "fold_scores.csv",
    RUN_ROOT / "gate_decision.json",
    RUN_ROOT / "oof_deepset_diversity.parquet",
]
assert all(path.exists() for path in required_paths)

if DIVERSITY_GATE_PASS:
    for path in [PARTNER_SUBMISSION_PATH, BLEND_SUBMISSION_PATH]:
        submission = pd.read_csv(path, encoding="utf-8-sig")
        assert submission.shape == sample_submission.shape == (8760, 5)
        assert list(submission.columns) == list(sample_submission.columns)
        assert submission["forecast_id"].equals(sample_submission["forecast_id"])
        assert submission[TARGET_COLS].notna().all().all()
        for column in TARGET_COLS:
            assert submission[column].between(0, CAPACITY_KWH[column]).all()
        print(path.name, submission.shape, "검증 통과")
    print("제출 권장 파일:", BLEND_SUBMISSION_PATH)
else:
    print("Gate REJECT: 마지막 제출권을 사용하지 않습니다.")

print("결과 저장 경로:", RUN_ROOT)
print("최종 판정:", gate_decision["recommendation"])
'''
cells.append(markdown("### 17. 결과·제출 파일 무결성 검증", "div_verify_title"))
cells.append(code(verify_source, "div_verify"))


cells.append(markdown(
    r'''
### 생성 파일

`/content/drive/MyDrive/BARAM2026/hyejin_deepset_distribution_diversity`에 저장됩니다.

- `candidate_scores.csv`: anchor 고정 상태의 point·scale·alpha 후보 비교
- `meta_selection.csv`: leave-one-fold-out에서 실제 선택된 사양
- `summary.csv`: anchor와 diversity blend의 CV·meta CV·HOLD 비교
- `fold_scores.csv`: 선택 후보의 fold별 개선량
- `gate_decision.json`: 마지막 제출 여부
- `best_epochs.csv`: DeepSets seed별 best epoch
- `oof_deepset_diversity.parquet`: anchor·직접 분위수·blend OOF
- `submission_deepset_partner_diagnostic.csv`: 직접 모델 진단용이며 단독 제출 금지
- `submission_deepset_diversity_blend.csv`: gate 통과 시 최종 제출 후보

최종 셀에서 `SUBMIT`일 때만 `submission_deepset_diversity_blend.csv`를 제출합니다.
''',
    "div_outputs",
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
