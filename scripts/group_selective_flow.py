import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "group_selective_flow.ipynb"


def lines(text):
    text = text.strip("\n")
    return [line + "\n" for line in text.splitlines()[:-1]] + [text.splitlines()[-1]]


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


cells = [
    markdown(
        r'''
## BARAM 2026 - Group-selective Flow

Flow 피처의 그룹별 효과를 분리하는 후속 실험입니다.

- 그룹 1·2: `F2_FLOW` 예측 사용
- 그룹 3: `C0` 예측 유지
- 기존 OOF와 제출 예측을 재사용하므로 모델을 다시 학습하지 않습니다.
- primary candidate는 사전에 `F2_G12`로 고정합니다.
''',
        "group_selective_intro",
    ),
]


sections = [
    (
        "1. Colab·Drive 설정",
        "group_selective_setup",
        r'''
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    FLOW_RUN_ROOT = Path("/content/drive/MyDrive/BARAM2026/hyejin_flow_spatial")
    RUN_ROOT = Path("/content/drive/MyDrive/BARAM2026/hyejin_group_selective_flow")
else:
    FLOW_RUN_ROOT = Path.cwd() / "runs" / "hyejin_flow_spatial"
    RUN_ROOT = Path.cwd() / "runs" / "hyejin_group_selective_flow"

RUN_ROOT.mkdir(parents=True, exist_ok=True)

TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPACITY_KWH = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}
GROUP_CAPACITY = {1: 21600, 2: 21600, 3: 21000}

PRIMARY_CANDIDATE = "F2_G12"
PRIMARY_FLOW_GROUPS = [1, 2]
PASS_CV_DELTA = 0.0005
PASS_MIN_POSITIVE_FOLDS = 4
PASS_MIN_SUBMETRIC_DELTA = -0.0005
BOOTSTRAP_SAMPLES = 3000
SEED = 42

print("flow input:", FLOW_RUN_ROOT)
print("output    :", RUN_ROOT)
''',
    ),
    (
        "2. 기존 Flow 결과 로드",
        "group_selective_load",
        r'''
INPUT_PATHS = {
    "oof": FLOW_RUN_ROOT / "oof_predictions.parquet",
    "c0_submission": FLOW_RUN_ROOT / "submission_flow_c0_control.csv",
    "f2_submission": FLOW_RUN_ROOT / "submission_flow_f2.csv",
}
missing = [str(path) for path in INPUT_PATHS.values() if not path.exists()]
assert not missing, "누락된 Flow 결과 파일:\n" + "\n".join(missing)

oof = pd.read_parquet(INPUT_PATHS["oof"])
oof["forecast_kst_dtm"] = pd.to_datetime(oof["forecast_kst_dtm"])
required_oof_cols = {"fold", "group", "forecast_kst_dtm", "target", "pred", "experiment"}
assert required_oof_cols.issubset(oof.columns)
assert set(oof["experiment"].unique()) == {"C0", "F2_FLOW"}
assert not oof[list(required_oof_cols)].isna().any().any()

print("OOF:", oof.shape)
print("folds:", sorted(oof["fold"].unique()))
''',
    ),
    (
        "3. 평가 지표",
        "group_selective_metric",
        r'''
def score_all_groups(frame, pred_col):
    group_nmae, group_ficr = [], []
    for group in [1, 2, 3]:
        sub = frame.loc[frame["group"] == group]
        actual = sub["target"].to_numpy(dtype=float)
        prediction = sub[pred_col].to_numpy(dtype=float)
        capacity = GROUP_CAPACITY[group]
        valid = actual >= capacity * 0.10
        actual, prediction = actual[valid], prediction[valid]
        error_rate = np.abs(prediction - actual) / capacity
        group_nmae.append(float(error_rate.mean()))
        unit_price = np.select(
            [error_rate <= 0.06, error_rate <= 0.08],
            [4.0, 3.0],
            default=0.0,
        )
        group_ficr.append(float(np.sum(actual * unit_price) / np.sum(actual * 4.0)))
    one_minus_nmae = 1.0 - float(np.mean(group_nmae))
    ficr = float(np.mean(group_ficr))
    return 0.5 * (one_minus_nmae + ficr), one_minus_nmae, ficr


def score_one_group(frame, pred_col, group):
    sub = frame.loc[frame["group"] == group]
    actual = sub["target"].to_numpy(dtype=float)
    prediction = sub[pred_col].to_numpy(dtype=float)
    capacity = GROUP_CAPACITY[group]
    valid = actual >= capacity * 0.10
    actual, prediction = actual[valid], prediction[valid]
    error_rate = np.abs(prediction - actual) / capacity
    one_minus_nmae = 1.0 - float(error_rate.mean())
    unit_price = np.select(
        [error_rate <= 0.06, error_rate <= 0.08],
        [4.0, 3.0],
        default=0.0,
    )
    ficr = float(np.sum(actual * unit_price) / np.sum(actual * 4.0))
    return 0.5 * (one_minus_nmae + ficr), one_minus_nmae, ficr
''',
    ),
    (
        "4. 그룹별 routing 후보 생성",
        "group_selective_candidates",
        r'''
KEYS = ["fold", "group", "forecast_kst_dtm", "target"]
wide = oof.pivot_table(
    index=KEYS,
    columns="experiment",
    values="pred",
    aggfunc="first",
).reset_index()
assert {"C0", "F2_FLOW"}.issubset(wide.columns)

CANDIDATE_ROUTES = {
    "C0": [],
    "F2_ALL": [1, 2, 3],
    "F2_G1": [1],
    "F2_G2": [2],
    "F2_G3": [3],
    "F2_G12": [1, 2],
}

for candidate, flow_groups in CANDIDATE_ROUTES.items():
    wide[candidate] = wide["C0"]
    for group in flow_groups:
        mask = wide["group"] == group
        wide.loc[mask, candidate] = wide.loc[mask, "F2_FLOW"]

assert wide[list(CANDIDATE_ROUTES)].notna().all().all()
print("primary:", PRIMARY_CANDIDATE, "/ flow groups:", PRIMARY_FLOW_GROUPS)
''',
    ),
    (
        "5. 전체·fold·그룹별 비교",
        "group_selective_scores",
        r'''
summary_rows, fold_rows, group_rows = [], [], []

for candidate in CANDIDATE_ROUTES:
    for partition, mask in [
        ("cv", wide["fold"].str.startswith("fold_")),
        ("hold", wide["fold"] == "hold_2024full"),
    ]:
        partition_frame = wide.loc[mask]
        score = score_all_groups(partition_frame, candidate)
        summary_rows.append({
            "candidate": candidate,
            "partition": partition,
            "Score": score[0],
            "1-NMAE": score[1],
            "FiCR": score[2],
        })
        for group in [1, 2, 3]:
            group_score = score_one_group(partition_frame, candidate, group)
            group_rows.append({
                "candidate": candidate,
                "partition": partition,
                "group": group,
                "Score": group_score[0],
                "1-NMAE": group_score[1],
                "FiCR": group_score[2],
            })

    for fold_name, fold_frame in wide.groupby("fold"):
        fold_score = score_all_groups(fold_frame, candidate)
        fold_rows.append({
            "candidate": candidate,
            "fold": fold_name,
            "Score": fold_score[0],
            "1-NMAE": fold_score[1],
            "FiCR": fold_score[2],
        })

summary = pd.DataFrame(summary_rows)
fold_scores = pd.DataFrame(fold_rows)
group_scores = pd.DataFrame(group_rows)


def add_control_deltas(table, key_cols):
    metrics = ["Score", "1-NMAE", "FiCR"]
    control = table.loc[table["candidate"] == "C0", key_cols + metrics].rename(
        columns={metric: f"control_{metric}" for metric in metrics}
    )
    result = table.merge(control, on=key_cols, how="left", validate="many_to_one")
    for metric in metrics:
        result[f"delta_{metric}"] = result[metric] - result[f"control_{metric}"]
    return result.drop(columns=[f"control_{metric}" for metric in metrics])


summary = add_control_deltas(summary, ["partition"])
fold_scores = add_control_deltas(fold_scores, ["fold"])
group_scores = add_control_deltas(group_scores, ["partition", "group"])

display(
    summary.loc[summary["candidate"].isin(["C0", "F2_ALL", PRIMARY_CANDIDATE])]
    .sort_values(["partition", "candidate"])
    .round(6)
)
display(
    fold_scores.loc[fold_scores["candidate"] == PRIMARY_CANDIDATE]
    .sort_values("fold")
    .round(6)
)
''',
    ),
    (
        "6. 사전 정의 gate 판정",
        "group_selective_gate",
        r'''
primary = summary.loc[summary["candidate"] == PRIMARY_CANDIDATE].set_index("partition")
primary_folds = fold_scores.loc[
    (fold_scores["candidate"] == PRIMARY_CANDIDATE)
    & fold_scores["fold"].str.startswith("fold_")
].copy()
positive_folds = int((primary_folds["delta_Score"] > 0).sum())

GROUP_SELECTIVE_GATE_PASS = bool(
    primary.loc["cv", "delta_Score"] >= PASS_CV_DELTA
    and positive_folds >= PASS_MIN_POSITIVE_FOLDS
    and primary.loc["hold", "delta_Score"] >= 0.0
    and primary.loc["cv", "delta_1-NMAE"] >= PASS_MIN_SUBMETRIC_DELTA
    and primary.loc["cv", "delta_FiCR"] > 0.0
)

gate_decision = {
    "passed": GROUP_SELECTIVE_GATE_PASS,
    "primary_candidate": PRIMARY_CANDIDATE,
    "flow_groups": PRIMARY_FLOW_GROUPS,
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

run_config = {
    "experiment": "GROUP_SELECTIVE_FLOW",
    "source_run": str(FLOW_RUN_ROOT),
    "primary_candidate": PRIMARY_CANDIDATE,
    "routing": {"group_1": "F2_FLOW", "group_2": "F2_FLOW", "group_3": "C0"},
    "retraining": False,
    "seed": SEED,
}

summary.to_csv(RUN_ROOT / "summary.csv", index=False)
fold_scores.to_csv(RUN_ROOT / "fold_scores.csv", index=False)
group_scores.to_csv(RUN_ROOT / "group_scores.csv", index=False)
wide[KEYS + list(CANDIDATE_ROUTES)].to_parquet(
    RUN_ROOT / "oof_group_selective.parquet", index=False
)
(RUN_ROOT / "gate_decision.json").write_text(
    json.dumps(gate_decision, ensure_ascii=False, indent=2), encoding="utf-8"
)
(RUN_ROOT / "run_config.json").write_text(
    json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
)

print("Gate:", "PASS" if GROUP_SELECTIVE_GATE_PASS else "REJECT")
print(json.dumps(gate_decision, ensure_ascii=False, indent=2))
''',
    ),
    (
        "7. 일자 단위 paired bootstrap",
        "group_selective_bootstrap",
        r'''
cv = wide.loc[wide["fold"].str.startswith("fold_")].copy()
cv["date"] = cv["forecast_kst_dtm"].dt.normalize()

for name in ["C0", PRIMARY_CANDIDATE]:
    capacity = cv["group"].map(GROUP_CAPACITY).to_numpy(dtype=float)
    actual = cv["target"].to_numpy(dtype=float)
    prediction = cv[name].to_numpy(dtype=float)
    error_rate = np.abs(prediction - actual) / capacity
    cv[f"valid_{name}"] = actual >= capacity * 0.10
    cv[f"error_{name}"] = error_rate
    cv[f"price_{name}"] = np.select(
        [error_rate <= 0.06, error_rate <= 0.08],
        [4.0, 3.0],
        default=0.0,
    )

valid = cv["valid_C0"] & cv[f"valid_{PRIMARY_CANDIDATE}"]
paired = cv.loc[valid].copy()
paired["actual4"] = paired["target"] * 4.0
paired["reward_C0"] = paired["target"] * paired["price_C0"]
paired["reward_primary"] = paired["target"] * paired[f"price_{PRIMARY_CANDIDATE}"]

daily = paired.groupby(["date", "group"], as_index=False).agg(
    n=("target", "size"),
    actual4=("actual4", "sum"),
    error_C0=("error_C0", "sum"),
    error_primary=(f"error_{PRIMARY_CANDIDATE}", "sum"),
    reward_C0=("reward_C0", "sum"),
    reward_primary=("reward_primary", "sum"),
)

dates = np.array(sorted(daily["date"].unique()))
lookup = {}
for group in [1, 2, 3]:
    group_daily = daily.loc[daily["group"] == group].set_index("date").reindex(dates).fillna(0)
    lookup[group] = {
        col: group_daily[col].to_numpy(dtype=float)
        for col in ["n", "actual4", "error_C0", "error_primary", "reward_C0", "reward_primary"]
    }

rng = np.random.default_rng(SEED)
deltas = []
for _ in range(BOOTSTRAP_SAMPLES):
    sampled = rng.integers(0, len(dates), len(dates))
    scores = {}
    for name in ["C0", "primary"]:
        nmae, ficr = [], []
        for group in [1, 2, 3]:
            item = lookup[group]
            nmae.append(item[f"error_{name}"][sampled].sum() / item["n"][sampled].sum())
            ficr.append(item[f"reward_{name}"][sampled].sum() / item["actual4"][sampled].sum())
        scores[name] = 0.5 * (1.0 - float(np.mean(nmae)) + float(np.mean(ficr)))
    deltas.append(scores["primary"] - scores["C0"])

deltas = np.asarray(deltas)
bootstrap_result = {
    "samples": BOOTSTRAP_SAMPLES,
    "seed": SEED,
    "mean_delta": float(deltas.mean()),
    "probability_delta_positive": float((deltas > 0).mean()),
    "probability_delta_at_least_gate": float((deltas >= PASS_CV_DELTA).mean()),
    "ci95": [float(value) for value in np.quantile(deltas, [0.025, 0.975])],
}
(RUN_ROOT / "bootstrap_result.json").write_text(
    json.dumps(bootstrap_result, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(bootstrap_result, ensure_ascii=False, indent=2))
''',
    ),
    (
        "8. Group-selective 제출 파일 생성",
        "group_selective_submission",
        r'''
SUBMISSION_PATH = None
if not GROUP_SELECTIVE_GATE_PASS:
    print("Gate 미통과: 제출 파일을 생성하지 않습니다.")
else:
    submission_c0 = pd.read_csv(INPUT_PATHS["c0_submission"], encoding="utf-8-sig")
    submission_f2 = pd.read_csv(INPUT_PATHS["f2_submission"], encoding="utf-8-sig")

    key_cols = ["forecast_id", "forecast_kst_dtm"]
    assert submission_c0[key_cols].equals(submission_f2[key_cols])
    assert submission_c0[TARGET_COLS].notna().all().all()
    assert submission_f2[TARGET_COLS].notna().all().all()

    submission = submission_c0.copy()
    submission["kpx_group_1"] = submission_f2["kpx_group_1"]
    submission["kpx_group_2"] = submission_f2["kpx_group_2"]
    submission["kpx_group_3"] = submission_c0["kpx_group_3"]

    for col in TARGET_COLS:
        assert submission[col].between(0, CAPACITY_KWH[col]).all()

    SUBMISSION_PATH = RUN_ROOT / "submission_flow_g12.csv"
    submission.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8-sig")

    delta_rows = []
    for group in [1, 2, 3]:
        col = f"kpx_group_{group}"
        delta = submission[col] - submission_c0[col]
        delta_rows.append({
            "group": group,
            "source": "F2_FLOW" if group in PRIMARY_FLOW_GROUPS else "C0",
            "mean_delta_kwh": float(delta.mean()),
            "mean_abs_delta_kwh": float(delta.abs().mean()),
            "max_abs_delta_kwh": float(delta.abs().max()),
        })
    pd.DataFrame(delta_rows).to_csv(
        RUN_ROOT / "submission_delta_summary.csv", index=False
    )
    print("Saved:", SUBMISSION_PATH)
''',
    ),
    (
        "9. 결과 파일 검증",
        "group_selective_verify",
        r'''
required_outputs = [
    RUN_ROOT / "summary.csv",
    RUN_ROOT / "fold_scores.csv",
    RUN_ROOT / "group_scores.csv",
    RUN_ROOT / "oof_group_selective.parquet",
    RUN_ROOT / "gate_decision.json",
    RUN_ROOT / "run_config.json",
    RUN_ROOT / "bootstrap_result.json",
]
if GROUP_SELECTIVE_GATE_PASS:
    required_outputs.extend([
        RUN_ROOT / "submission_flow_g12.csv",
        RUN_ROOT / "submission_delta_summary.csv",
    ])

missing_outputs = [str(path) for path in required_outputs if not path.exists()]
assert not missing_outputs, "누락된 결과 파일:\n" + "\n".join(missing_outputs)

if SUBMISSION_PATH is not None:
    checked = pd.read_csv(SUBMISSION_PATH, encoding="utf-8-sig")
    assert len(checked) == 8760
    assert checked[TARGET_COLS].notna().all().all()

print("결과 저장 경로:", RUN_ROOT)
print("Gate:", "PASS" if GROUP_SELECTIVE_GATE_PASS else "REJECT")
''',
    ),
]


for index, (title, cell_id, source) in enumerate(sections, start=1):
    cells.append(markdown(f"### {title}", f"{cell_id}_title"))
    cells.append(code(source, cell_id))


cells.append(
    markdown(
        r'''
### 생성 파일

`/content/drive/MyDrive/BARAM2026/hyejin_group_selective_flow`에 결과가 저장됩니다.

- `summary.csv`: C0, 전체 Flow, 그룹별 Flow routing 비교
- `fold_scores.csv`: fold별 변화
- `group_scores.csv`: 그룹별 변화
- `gate_decision.json`: `F2_G12` gate 판정
- `bootstrap_result.json`: 일자 단위 paired bootstrap
- `oof_group_selective.parquet`: 후보별 OOF 예측
- gate 통과 시 `submission_flow_g12.csv`
- `submission_delta_summary.csv`: C0 대비 제출값 변화
''',
        "group_selective_outputs",
    )
)


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT)
