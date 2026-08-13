"""Build the lightweight external-champion plus serial-correction blend notebook."""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "external_serial_blend.ipynb"


def lines(text: str) -> list[str]:
    text = text.strip("\n")
    parts = text.splitlines()
    return [line + "\n" for line in parts[:-1]] + [parts[-1]]


ids = count(1)


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": f"blend_md_{next(ids):02d}",
        "metadata": {},
        "source": lines(text),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"blend_code_{next(ids):02d}",
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


cells = [
    markdown(
        r'''
## BARAM 2026 - External Champion + Serial Correction Blend

기존 external champion을 기준 모델로 고정하고, 독립 serial 모델은 작은 보정 신호로만 사용합니다.

- external 예측 비중: 80~100%
- serial 예측 비중: 그룹별 0~20%
- 고정 scaling과 snap 사용 안 함
- 비율 선택에는 2024 분기 CV만 사용
- 2024 전체 HOLD와 leave-one-fold-out meta OOF는 최종 gate에만 사용
- 기존 external·serial 캐시를 재사용하므로 모델 재학습 없음
'''
    ),
]


setup = r'''
import json, pickle, sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_ROOT = Path("/content/drive/MyDrive/BARAM2026")
else:
    DRIVE_ROOT = Path.cwd() / "runs"

EXTERNAL_ROOT = DRIVE_ROOT / "hyejin_external_champion"
SERIAL_ROOT = DRIVE_ROOT / "hyejin_serial_residual_flow"
RUN_ROOT = DRIVE_ROOT / "hyejin_external_serial_blend"
RUN_ROOT.mkdir(parents=True, exist_ok=True)

EXTERNAL_GBM_PATH = EXTERNAL_ROOT / "cache" / "external_champion_gbm_v1.parquet"
EXTERNAL_DL_PATH = EXTERNAL_ROOT / "cache" / "external_champion_dl_v1.parquet"
EXTERNAL_BIAS_PATH = EXTERNAL_ROOT / "cache" / "external_champion_bias_v1.pkl"
EXTERNAL_VALIDATION_PATH = EXTERNAL_ROOT / "validation_result.json"
EXTERNAL_SUBMISSION_PATH = EXTERNAL_ROOT / "submission_external_nosnap_ec.csv"

SERIAL_OOF_PATH = SERIAL_ROOT / "oof_serial_predictions.parquet"
SERIAL_SUBMISSION_PATH = SERIAL_ROOT / "submission_serial_control_unscaled.csv"

REQUIRED_PATHS = [
    EXTERNAL_GBM_PATH,
    EXTERNAL_DL_PATH,
    EXTERNAL_BIAS_PATH,
    EXTERNAL_VALIDATION_PATH,
    EXTERNAL_SUBMISSION_PATH,
    SERIAL_OOF_PATH,
    SERIAL_SUBMISSION_PATH,
]
missing = [str(path) for path in REQUIRED_PATHS if not path.exists()]
assert not missing, "필요한 기존 결과가 없습니다:\n" + "\n".join(missing)

TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
GROUP_CAPACITY = {1: 21600.0, 2: 21600.0, 3: 21000.0}
FOLD_NAMES = ["fold_2024Q1", "fold_2024Q2", "fold_2024Q3", "fold_2024Q4"]
HOLD_NAME = "hold_2024full"

ALPHA_GRID = [0.00, 0.05, 0.10, 0.15, 0.20]
EXTERNAL_GBM_WEIGHT = 0.65
RUN_SUBMISSION_IF_PASS = True

PASS_CV_DELTA = 0.0003
PASS_META_DELTA = 0.0002
PASS_MIN_POSITIVE_FOLDS = 3
PASS_MIN_GROUP_DELTA = -0.0003
PASS_MIN_NMAE_DELTA = -0.0003

print("external:", EXTERNAL_ROOT)
print("serial  :", SERIAL_ROOT)
print("output  :", RUN_ROOT)
print("candidate count:", len(ALPHA_GRID) ** 3)
'''
cells.extend([markdown("### 1. Drive 경로와 실험 설정"), code(setup)])


metrics = r'''
def apply_rolling_smoothing(prediction, window=3, alpha=0.40):
    series = pd.Series(prediction)
    rolling = series.rolling(window, center=True, min_periods=1).mean()
    return (alpha * series + (1 - alpha) * rolling).to_numpy()


def apply_bias_correction(prediction, wind_speed, group, bias_map):
    output = np.asarray(prediction, dtype=float).copy()
    if group not in bias_map:
        return output
    wind_speed = np.asarray(wind_speed, dtype=float)
    for interval, bias in bias_map[group].items():
        mask = (wind_speed > interval.left) & (wind_speed <= interval.right)
        output[mask] *= 1 - bias
    return output


def postprocess_external(cf, capacity, wind_speed, group, bias_map):
    prediction = apply_bias_correction(
        np.asarray(cf, dtype=float) * capacity,
        wind_speed,
        group,
        bias_map,
    )
    prediction = apply_rolling_smoothing(prediction, window=3, alpha=0.40)
    return np.clip(prediction, capacity * 0.16, capacity)


def score_values(frame, prediction):
    prediction = np.asarray(prediction, dtype=float)
    group_nmae, group_ficr = [], []
    present_groups = sorted(int(group) for group in frame["group"].unique())
    assert present_groups
    for group in present_groups:
        capacity = GROUP_CAPACITY[group]
        mask = frame["group"].to_numpy() == group
        actual = frame.loc[mask, "target"].to_numpy(dtype=float)
        forecast = prediction[mask]
        valid = actual >= capacity * 0.10
        actual, forecast = actual[valid], forecast[valid]
        error_rate = np.abs(forecast - actual) / capacity
        group_nmae.append(float(error_rate.mean()))
        unit_price = np.select(
            [error_rate <= 0.06, error_rate <= 0.08],
            [4.0, 3.0],
            default=0.0,
        )
        group_ficr.append(float(np.sum(actual * unit_price) / np.sum(actual * 4.0)))
    one_nmae = 1 - float(np.mean(group_nmae))
    ficr = float(np.mean(group_ficr))
    return {"Score": 0.5 * (one_nmae + ficr), "1-NMAE": one_nmae, "FiCR": ficr}


def alpha_prediction(frame, alphas):
    alpha = frame["group"].map({1: alphas[0], 2: alphas[1], 3: alphas[2]}).to_numpy()
    external = frame["external_pred"].to_numpy(dtype=float)
    serial = frame["serial_pred"].to_numpy(dtype=float)
    return external + alpha * (serial - external)


def candidate_name(alphas):
    return "_".join(f"g{group}a{int(round(alpha * 100)):02d}" for group, alpha in enumerate(alphas, 1))
'''
cells.extend([markdown("### 2. 평가 지표와 external 후처리"), code(metrics)])


load_external = r'''
external_gbm = pd.read_parquet(EXTERNAL_GBM_PATH)
external_dl = pd.read_parquet(EXTERNAL_DL_PATH)
with open(EXTERNAL_BIAS_PATH, "rb") as stream:
    external_bias = pickle.load(stream)

external_cache = external_gbm.merge(
    external_dl,
    on=["fold", "group", "forecast_kst_dtm"],
    how="left",
    validate="one_to_one",
)
assert external_cache["dl_q60"].notna().all()

external_parts = []
for (fold, group), sub in external_cache.groupby(["fold", "group"], sort=False):
    sub = sub.sort_values("forecast_kst_dtm").copy()
    capacity = GROUP_CAPACITY[int(group)]
    cf = (
        EXTERNAL_GBM_WEIGHT * sub["gbm_cf"].to_numpy(dtype=float)
        + (1 - EXTERNAL_GBM_WEIGHT) * sub["dl_q60"].to_numpy(dtype=float)
    )
    prediction = postprocess_external(
        cf,
        capacity,
        sub["ws"].to_numpy(dtype=float),
        int(group),
        external_bias[fold],
    )
    external_parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(
        external_pred=prediction
    ))

external_oof = pd.concat(external_parts, ignore_index=True)
assert set(external_oof["fold"].unique()) == set(FOLD_NAMES + [HOLD_NAME])

reference = json.loads(EXTERNAL_VALIDATION_PATH.read_text(encoding="utf-8"))
external_cv = external_oof.loc[external_oof["fold"].isin(FOLD_NAMES)]
external_hold = external_oof.loc[external_oof["fold"] == HOLD_NAME]
reconstructed_cv = score_values(external_cv, external_cv["external_pred"])
reconstructed_hold = score_values(external_hold, external_hold["external_pred"])
for partition, rebuilt in [("cv", reconstructed_cv), ("hold", reconstructed_hold)]:
    for metric_name in ["Score", "1-NMAE", "FiCR"]:
        assert abs(rebuilt[metric_name] - reference[partition][metric_name]) < 1e-8, (
            partition, metric_name, rebuilt[metric_name], reference[partition][metric_name]
        )

print("external cache:", external_cache.shape)
print("external CV  :", reconstructed_cv)
print("external HOLD:", reconstructed_hold)
'''
cells.extend([markdown("### 3. External champion OOF 복원"), code(load_external)])


align_serial = r'''
serial_oof = pd.read_parquet(SERIAL_OOF_PATH)
serial_control = serial_oof.loc[
    serial_oof["experiment"] == "R0_CONTROL",
    ["fold", "group", "forecast_kst_dtm", "target", "pred_unscaled"],
].rename(columns={"target": "serial_target", "pred_unscaled": "serial_pred"})

blend_oof = external_oof.merge(
    serial_control,
    on=["fold", "group", "forecast_kst_dtm"],
    how="inner",
    validate="one_to_one",
)
assert len(blend_oof) == len(external_oof) == len(serial_control)
assert np.allclose(blend_oof["target"], blend_oof["serial_target"], equal_nan=True)
blend_oof = blend_oof.drop(columns="serial_target")
assert blend_oof[["external_pred", "serial_pred", "target"]].notna().all().all()

for group in [1, 2, 3]:
    group_frame = blend_oof.loc[
        (blend_oof["fold"].isin(FOLD_NAMES)) & (blend_oof["group"] == group)
    ]
    correlation = group_frame["external_pred"].corr(group_frame["serial_pred"])
    mean_delta_cf = (
        (group_frame["serial_pred"] - group_frame["external_pred"]).abs().mean()
        / GROUP_CAPACITY[group]
    )
    print(f"group {group}: corr={correlation:.6f}, mean |delta| CF={mean_delta_cf:.4%}")
'''
cells.extend([markdown("### 4. Serial control OOF 정렬"), code(align_serial)])


candidate_search = r'''
CANDIDATES = list(product(ALPHA_GRID, repeat=3))
ZERO_ALPHA = (0.0, 0.0, 0.0)


def evaluate_candidate(frame, alphas):
    return score_values(frame, alpha_prediction(frame, alphas))


def candidate_diagnostics(frame, alphas, fold_names):
    selected = frame.loc[frame["fold"].isin(fold_names)]
    baseline = score_values(selected, selected["external_pred"])
    candidate = evaluate_candidate(selected, alphas)
    fold_deltas = []
    for fold_name in fold_names:
        fold_frame = frame.loc[frame["fold"] == fold_name]
        base_fold = score_values(fold_frame, fold_frame["external_pred"])
        cand_fold = evaluate_candidate(fold_frame, alphas)
        fold_deltas.append(cand_fold["Score"] - base_fold["Score"])
    group_deltas = []
    for group in [1, 2, 3]:
        group_frame = selected.loc[selected["group"] == group]
        base_group = score_values(group_frame, group_frame["external_pred"])
        cand_group = evaluate_candidate(group_frame, alphas)
        group_deltas.append(cand_group["Score"] - base_group["Score"])
    return {
        "candidate": candidate_name(alphas),
        "alpha_g1": alphas[0],
        "alpha_g2": alphas[1],
        "alpha_g3": alphas[2],
        "Score": candidate["Score"],
        "1-NMAE": candidate["1-NMAE"],
        "FiCR": candidate["FiCR"],
        "delta_Score": candidate["Score"] - baseline["Score"],
        "delta_1-NMAE": candidate["1-NMAE"] - baseline["1-NMAE"],
        "delta_FiCR": candidate["FiCR"] - baseline["FiCR"],
        "positive_folds": int(np.sum(np.asarray(fold_deltas) > 0)),
        "min_fold_delta": float(np.min(fold_deltas)),
        "min_group_delta": float(np.min(group_deltas)),
    }


def select_candidate(frame, fold_names, min_positive_folds):
    rows = [candidate_diagnostics(frame, alphas, fold_names) for alphas in CANDIDATES]
    table = pd.DataFrame(rows)
    safe = table.loc[
        (table[["alpha_g1", "alpha_g2", "alpha_g3"]].sum(axis=1) > 0)
        & (table["delta_Score"] > 0)
        & (table["delta_1-NMAE"] >= PASS_MIN_NMAE_DELTA)
        & (table["delta_FiCR"] > 0)
        & (table["positive_folds"] >= min_positive_folds)
        & (table["min_group_delta"] >= -0.00075)
    ].sort_values(["Score", "min_fold_delta"], ascending=False)
    if safe.empty:
        selected = table.loc[table["candidate"] == candidate_name(ZERO_ALPHA)].iloc[0]
    else:
        selected = safe.iloc[0]
    alphas = tuple(float(selected[f"alpha_g{group}"]) for group in [1, 2, 3])
    return alphas, selected.to_dict(), table


SELECTED_ALPHA, selected_cv_row, candidate_scores = select_candidate(
    blend_oof, FOLD_NAMES, PASS_MIN_POSITIVE_FOLDS
)
candidate_scores.to_csv(RUN_ROOT / "candidate_scores.csv", index=False)
print("CV selected:", candidate_name(SELECTED_ALPHA), SELECTED_ALPHA)
print(json.dumps(selected_cv_row, ensure_ascii=False, indent=2))
'''
cells.extend([markdown("### 5. 그룹별 소규모 blend 탐색"), code(candidate_search)])


meta_validation = r'''
meta_parts, meta_rows = [], []
for heldout_fold in FOLD_NAMES:
    train_folds = [fold for fold in FOLD_NAMES if fold != heldout_fold]
    selected_alpha, train_row, _ = select_candidate(
        blend_oof, train_folds, min_positive_folds=2
    )
    heldout = blend_oof.loc[blend_oof["fold"] == heldout_fold].copy()
    heldout["meta_pred"] = alpha_prediction(heldout, selected_alpha)
    meta_parts.append(heldout)
    baseline = score_values(heldout, heldout["external_pred"])
    candidate = score_values(heldout, heldout["meta_pred"])
    meta_rows.append({
        "heldout_fold": heldout_fold,
        "candidate": candidate_name(selected_alpha),
        "alpha_g1": selected_alpha[0],
        "alpha_g2": selected_alpha[1],
        "alpha_g3": selected_alpha[2],
        "train_delta_score": train_row["delta_Score"],
        "heldout_delta_score": candidate["Score"] - baseline["Score"],
        "heldout_delta_1_nmae": candidate["1-NMAE"] - baseline["1-NMAE"],
        "heldout_delta_ficr": candidate["FiCR"] - baseline["FiCR"],
    })

meta_oof = pd.concat(meta_parts, ignore_index=True)
meta_baseline = score_values(meta_oof, meta_oof["external_pred"])
meta_candidate = score_values(meta_oof, meta_oof["meta_pred"])
META_DELTA_SCORE = meta_candidate["Score"] - meta_baseline["Score"]
meta_selection = pd.DataFrame(meta_rows)
meta_selection.to_csv(RUN_ROOT / "meta_selection.csv", index=False)

display(meta_selection.round(6))
print("meta baseline :", meta_baseline)
print("meta candidate:", meta_candidate)
print("meta delta    :", META_DELTA_SCORE)
'''
cells.extend([markdown("### 6. Leave-one-fold-out meta 검증"), code(meta_validation)])


score_gate = r'''
selected_prediction = alpha_prediction(blend_oof, SELECTED_ALPHA)
blend_oof["blend_pred"] = selected_prediction

summary_rows = []
for partition, mask in [
    ("cv", blend_oof["fold"].isin(FOLD_NAMES)),
    ("hold", blend_oof["fold"] == HOLD_NAME),
]:
    part = blend_oof.loc[mask]
    for experiment, pred_col in [
        ("EXTERNAL_CONTROL", "external_pred"),
        ("SERIAL_BLEND", "blend_pred"),
    ]:
        result = score_values(part, part[pred_col])
        summary_rows.append({"partition": partition, "experiment": experiment, **result})
summary = pd.DataFrame(summary_rows)

fold_rows = []
for fold_name in FOLD_NAMES + [HOLD_NAME]:
    part = blend_oof.loc[blend_oof["fold"] == fold_name]
    base = score_values(part, part["external_pred"])
    candidate = score_values(part, part["blend_pred"])
    fold_rows.append({
        "fold": fold_name,
        **{f"control_{key}": value for key, value in base.items()},
        **{key: value for key, value in candidate.items()},
        **{f"delta_{key}": candidate[key] - base[key] for key in base},
    })
fold_scores = pd.DataFrame(fold_rows)

group_rows = []
for partition, mask in [
    ("cv", blend_oof["fold"].isin(FOLD_NAMES)),
    ("hold", blend_oof["fold"] == HOLD_NAME),
]:
    for group in [1, 2, 3]:
        part = blend_oof.loc[mask & (blend_oof["group"] == group)]
        base = score_values(part, part["external_pred"])
        candidate = score_values(part, part["blend_pred"])
        group_rows.append({
            "partition": partition,
            "group": group,
            **{f"control_{key}": value for key, value in base.items()},
            **{key: value for key, value in candidate.items()},
            **{f"delta_{key}": candidate[key] - base[key] for key in base},
        })
group_scores = pd.DataFrame(group_rows)

cv_base = summary.loc[
    (summary["partition"] == "cv") & (summary["experiment"] == "EXTERNAL_CONTROL")
].iloc[0]
cv_candidate = summary.loc[
    (summary["partition"] == "cv") & (summary["experiment"] == "SERIAL_BLEND")
].iloc[0]
hold_base = summary.loc[
    (summary["partition"] == "hold") & (summary["experiment"] == "EXTERNAL_CONTROL")
].iloc[0]
hold_candidate = summary.loc[
    (summary["partition"] == "hold") & (summary["experiment"] == "SERIAL_BLEND")
].iloc[0]

cv_delta = float(cv_candidate["Score"] - cv_base["Score"])
hold_delta = float(hold_candidate["Score"] - hold_base["Score"])
cv_delta_nmae = float(cv_candidate["1-NMAE"] - cv_base["1-NMAE"])
cv_delta_ficr = float(cv_candidate["FiCR"] - cv_base["FiCR"])
positive_folds = int((fold_scores.loc[fold_scores["fold"].isin(FOLD_NAMES), "delta_Score"] > 0).sum())
min_group_delta = float(group_scores.loc[group_scores["partition"] == "cv", "delta_Score"].min())

BLEND_GATE_PASS = bool(
    sum(SELECTED_ALPHA) > 0
    and cv_delta >= PASS_CV_DELTA
    and META_DELTA_SCORE >= PASS_META_DELTA
    and positive_folds >= PASS_MIN_POSITIVE_FOLDS
    and hold_delta >= 0
    and min_group_delta >= PASS_MIN_GROUP_DELTA
    and cv_delta_nmae >= PASS_MIN_NMAE_DELTA
    and cv_delta_ficr > 0
)

gate_decision = {
    "passed": BLEND_GATE_PASS,
    "selected_alpha": {f"group_{group}": SELECTED_ALPHA[group - 1] for group in [1, 2, 3]},
    "cv_delta_score": cv_delta,
    "meta_delta_score": float(META_DELTA_SCORE),
    "hold_delta_score": hold_delta,
    "cv_delta_1_nmae": cv_delta_nmae,
    "cv_delta_ficr": cv_delta_ficr,
    "positive_cv_folds": positive_folds,
    "min_cv_group_delta": min_group_delta,
}

summary.to_csv(RUN_ROOT / "summary.csv", index=False)
fold_scores.to_csv(RUN_ROOT / "fold_scores.csv", index=False)
group_scores.to_csv(RUN_ROOT / "group_scores.csv", index=False)
blend_oof.to_parquet(RUN_ROOT / "blend_oof_predictions.parquet", index=False)
(RUN_ROOT / "gate_decision.json").write_text(
    json.dumps(gate_decision, ensure_ascii=False, indent=2), encoding="utf-8"
)

display(summary.round(6))
display(fold_scores.round(6))
display(group_scores.round(6))
print("Gate:", "PASS" if BLEND_GATE_PASS else "REJECT")
print(json.dumps(gate_decision, ensure_ascii=False, indent=2))
'''
cells.extend([markdown("### 7. External 기준 비교와 최종 gate"), code(score_gate)])


submission = r'''
external_submission = pd.read_csv(EXTERNAL_SUBMISSION_PATH, encoding="utf-8-sig")
serial_submission = pd.read_csv(SERIAL_SUBMISSION_PATH, encoding="utf-8-sig")
assert len(external_submission) == len(serial_submission) == 8760
assert external_submission["forecast_id"].equals(serial_submission["forecast_id"])
assert external_submission["forecast_kst_dtm"].equals(serial_submission["forecast_kst_dtm"])
assert external_submission[TARGET_COLS].notna().all().all()
assert serial_submission[TARGET_COLS].notna().all().all()

submission_delta_rows = []
blend_submission = external_submission.copy()
for group in [1, 2, 3]:
    col = f"kpx_group_{group}"
    alpha = SELECTED_ALPHA[group - 1]
    external_values = external_submission[col].to_numpy(dtype=float)
    serial_values = serial_submission[col].to_numpy(dtype=float)
    blend_values = external_values + alpha * (serial_values - external_values)
    blend_submission[col] = np.clip(blend_values, 0, GROUP_CAPACITY[group])
    submission_delta_rows.append({
        "group": group,
        "alpha": alpha,
        "external_serial_correlation": float(np.corrcoef(external_values, serial_values)[0, 1]),
        "mean_abs_change": float(np.mean(np.abs(blend_values - external_values))),
        "mean_abs_change_cf": float(np.mean(np.abs(blend_values - external_values)) / GROUP_CAPACITY[group]),
        "max_abs_change": float(np.max(np.abs(blend_values - external_values))),
    })

submission_delta_summary = pd.DataFrame(submission_delta_rows)
submission_delta_summary.to_csv(RUN_ROOT / "submission_delta_summary.csv", index=False)

SUBMISSION_PATH = None
if BLEND_GATE_PASS and RUN_SUBMISSION_IF_PASS:
    SUBMISSION_PATH = RUN_ROOT / "submission_external_serial_blend.csv"
    blend_submission.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8-sig")
    print("Saved:", SUBMISSION_PATH)
else:
    print("Gate 미통과: 새 submission을 저장하지 않습니다.")

display(submission_delta_summary.round(6))
'''
cells.extend([markdown("### 8. Gate 통과 시 최종 submission 생성"), code(submission)])


save_config = r'''
run_config = {
    "experiment": "EXTERNAL_SERIAL_BLEND",
    "base_model": "hyejin_external_champion",
    "correction_model": "serial_control_unscaled",
    "model_retraining": False,
    "alpha_grid": ALPHA_GRID,
    "selected_alpha": {f"group_{group}": SELECTED_ALPHA[group - 1] for group in [1, 2, 3]},
    "external_gbm_weight": EXTERNAL_GBM_WEIGHT,
    "post_blend_scaling": False,
    "snap": False,
    "selection_data": FOLD_NAMES,
    "hold_usage": "gate_only",
    "thresholds": {
        "cv_delta": PASS_CV_DELTA,
        "meta_delta": PASS_META_DELTA,
        "positive_folds": PASS_MIN_POSITIVE_FOLDS,
        "min_group_delta": PASS_MIN_GROUP_DELTA,
        "min_nmae_delta": PASS_MIN_NMAE_DELTA,
        "hold_nonnegative": True,
        "ficr_positive": True,
    },
}
(RUN_ROOT / "run_config.json").write_text(
    json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
)

required_outputs = [
    RUN_ROOT / "candidate_scores.csv",
    RUN_ROOT / "meta_selection.csv",
    RUN_ROOT / "summary.csv",
    RUN_ROOT / "fold_scores.csv",
    RUN_ROOT / "group_scores.csv",
    RUN_ROOT / "blend_oof_predictions.parquet",
    RUN_ROOT / "gate_decision.json",
    RUN_ROOT / "submission_delta_summary.csv",
    RUN_ROOT / "run_config.json",
]
assert all(path.exists() for path in required_outputs)

if SUBMISSION_PATH is not None:
    saved = pd.read_csv(SUBMISSION_PATH, encoding="utf-8-sig")
    assert saved.shape == external_submission.shape == (8760, 5)
    assert saved["forecast_id"].equals(external_submission["forecast_id"])
    assert saved[TARGET_COLS].notna().all().all()
    for group in [1, 2, 3]:
        col = f"kpx_group_{group}"
        assert saved[col].between(0, GROUP_CAPACITY[group]).all()
    print(SUBMISSION_PATH.name, saved.shape, "검증 통과")

print("결과 저장 경로:", RUN_ROOT)
print("Gate:", "PASS" if BLEND_GATE_PASS else "REJECT")
'''
cells.extend([markdown("### 9. 설정 저장과 결과 검증"), code(save_config)])


cells.append(
    markdown(
        r'''
### 생성 파일

`/content/drive/MyDrive/BARAM2026/hyejin_external_serial_blend`에 저장됩니다.

- `candidate_scores.csv`: 그룹별 blend 후보 125개의 CV 진단
- `meta_selection.csv`: leave-one-fold-out에서 선택된 비율과 held-out 변화
- `summary.csv`: external control과 최종 blend의 CV·HOLD 비교
- `fold_scores.csv`: fold별 변화
- `group_scores.csv`: 그룹별 변화
- `gate_decision.json`: 엄격한 gate 판정
- `blend_oof_predictions.parquet`: 정렬된 external·serial·blend OOF
- `submission_delta_summary.csv`: 최종 예측 변화량
- `run_config.json`: 재현 설정
- gate 통과 시 `submission_external_serial_blend.csv`

이 노트북은 기존 예측 캐시만 읽으므로 A100이 필요하지 않습니다.
'''
    )
)


notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT)
