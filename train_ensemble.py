from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPACITY_KWH = {
    "kpx_group_1": 21600.0,
    "kpx_group_2": 21600.0,
    "kpx_group_3": 21000.0,
}
VALID_START = pd.Timestamp("2024-01-01 00:00:00")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build: Callable[[int, bool], object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RF/XGB/Cat/LGBM models and create an ensemble submission."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing train/, test/, and sample_submission.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for metrics and model submissions. Defaults to data-dir/outputs/ensemble.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer estimators for a smoke test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--ensemble",
        choices=["weighted", "mean"],
        default="weighted",
        help="Use inverse-validation-MAE weights or a simple mean.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Worker threads for model libraries. Default 1 avoids Windows sandbox pipe errors.",
    )
    return parser.parse_args()


def require_boosting_packages() -> tuple[type, type, type]:
    missing = []
    try:
        from xgboost import XGBRegressor
    except Exception:
        XGBRegressor = None
        missing.append("xgboost")

    try:
        from catboost import CatBoostRegressor
    except Exception:
        CatBoostRegressor = None
        missing.append("catboost")

    try:
        from lightgbm import LGBMRegressor
    except Exception:
        LGBMRegressor = None
        missing.append("lightgbm")

    if missing:
        command = f"{sys.executable} -m pip install " + " ".join(missing)
        raise SystemExit(
            "Missing packages: "
            + ", ".join(missing)
            + "\nInstall them first, for example:\n"
            + command
        )

    return XGBRegressor, CatBoostRegressor, LGBMRegressor


def model_specs(quick: bool, jobs: int) -> list[ModelSpec]:
    XGBRegressor, CatBoostRegressor, LGBMRegressor = require_boosting_packages()

    rf_estimators = 120 if quick else 320
    xgb_estimators = 180 if quick else 650
    cat_iterations = 220 if quick else 800
    lgbm_estimators = 220 if quick else 800

    return [
        ModelSpec(
            "rf",
            lambda seed, quick: RandomForestRegressor(
                n_estimators=rf_estimators,
                max_depth=16,
                min_samples_leaf=6,
                max_features="sqrt",
                random_state=seed,
                n_jobs=jobs,
            ),
        ),
        ModelSpec(
            "xgb",
            lambda seed, quick: XGBRegressor(
                n_estimators=xgb_estimators,
                learning_rate=0.035,
                max_depth=5,
                min_child_weight=4,
                subsample=0.9,
                colsample_bytree=0.85,
                reg_alpha=0.05,
                reg_lambda=2.0,
                objective="reg:squarederror",
                tree_method="hist",
                eval_metric="mae",
                random_state=seed,
                n_jobs=jobs,
            ),
        ),
        ModelSpec(
            "cat",
            lambda seed, quick: CatBoostRegressor(
                iterations=cat_iterations,
                learning_rate=0.035,
                depth=6,
                l2_leaf_reg=5.0,
                loss_function="RMSE",
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
                thread_count=jobs,
            ),
        ),
        ModelSpec(
            "lgbm",
            lambda seed, quick: LGBMRegressor(
                n_estimators=lgbm_estimators,
                learning_rate=0.03,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.9,
                subsample_freq=1,
                colsample_bytree=0.85,
                reg_alpha=0.05,
                reg_lambda=2.0,
                objective="regression",
                random_state=seed,
                n_jobs=jobs,
                verbosity=-1,
            ),
        ),
    ]


def add_vector_speeds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pairs = [
        ("heightAboveGround_10_10u", "heightAboveGround_10_10v", "wind_speed_10m"),
        ("heightAboveGround_80_u", "heightAboveGround_80_v", "wind_speed_80m"),
        ("heightAboveGround_100_100u", "heightAboveGround_100_100v", "wind_speed_100m"),
        ("planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v", "wind_speed_pbl"),
        ("isobaricInhPa_850_u", "isobaricInhPa_850_v", "wind_speed_850hpa"),
        ("isobaricInhPa_700_u", "isobaricInhPa_700_v", "wind_speed_700hpa"),
        ("isobaricInhPa_500_u", "isobaricInhPa_500_v", "wind_speed_500hpa"),
    ]
    for u_col, v_col, out_col in pairs:
        if u_col in df.columns and v_col in df.columns:
            df[out_col] = np.hypot(df[u_col], df[v_col])
    return df


def aggregate_weather(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    df["data_available_kst_dtm"] = pd.to_datetime(df["data_available_kst_dtm"])
    df = add_vector_speeds(df)

    drop_cols = {"forecast_kst_dtm", "data_available_kst_dtm", "grid_id", "latitude", "longitude"}
    value_cols = [c for c in df.columns if c not in drop_cols]

    agg = df.groupby("forecast_kst_dtm", sort=True)[value_cols].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{prefix}_{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()

    meta = (
        df.groupby("forecast_kst_dtm", sort=True)["data_available_kst_dtm"]
        .first()
        .reset_index()
    )
    horizon_hours = (
        meta["forecast_kst_dtm"] - meta["data_available_kst_dtm"]
    ).dt.total_seconds() / 3600.0
    meta[f"{prefix}_forecast_horizon_hours"] = horizon_hours
    meta[f"{prefix}_data_available_hour"] = meta["data_available_kst_dtm"].dt.hour
    meta = meta.drop(columns=["data_available_kst_dtm"])

    return agg.merge(meta, on="forecast_kst_dtm", how="left")


def calendar_features(dt_series: pd.Series) -> pd.DataFrame:
    dt = pd.to_datetime(dt_series)
    out = pd.DataFrame(index=dt.index)
    out["month"] = dt.dt.month
    out["day"] = dt.dt.day
    out["hour"] = dt.dt.hour
    out["dayofweek"] = dt.dt.dayofweek
    out["dayofyear"] = dt.dt.dayofyear
    out["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    out["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 366.0)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 366.0)
    return out


def read_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    train_labels = pd.read_csv(train_dir / "train_labels.csv", encoding="utf-8-sig")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv", encoding="utf-8-sig")
    ldaps_train = pd.read_csv(train_dir / "ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv(train_dir / "gfs_train.csv", encoding="utf-8-sig")
    ldaps_test = pd.read_csv(test_dir / "ldaps_test.csv", encoding="utf-8-sig")
    gfs_test = pd.read_csv(test_dir / "gfs_test.csv", encoding="utf-8-sig")
    return train_labels, sample_submission, ldaps_train, gfs_train, ldaps_test, gfs_test


def build_feature_matrices(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_labels, sample_submission, ldaps_train, gfs_train, ldaps_test, gfs_test = read_inputs(data_dir)

    train_labels["kst_dtm"] = pd.to_datetime(train_labels["kst_dtm"])
    sample_submission["forecast_kst_dtm"] = pd.to_datetime(sample_submission["forecast_kst_dtm"])

    train_weather = aggregate_weather(ldaps_train, "ldaps").merge(
        aggregate_weather(gfs_train, "gfs"), on="forecast_kst_dtm", how="inner"
    )
    test_weather = aggregate_weather(ldaps_test, "ldaps").merge(
        aggregate_weather(gfs_test, "gfs"), on="forecast_kst_dtm", how="inner"
    )

    train_df = train_labels.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(
        train_weather, on="forecast_kst_dtm", how="left"
    )
    test_df = sample_submission[["forecast_id", "forecast_kst_dtm"]].merge(
        test_weather, on="forecast_kst_dtm", how="left"
    )

    X_train = pd.concat(
        [
            calendar_features(train_df["forecast_kst_dtm"]),
            train_df.drop(columns=["forecast_kst_dtm", *TARGET_COLS]),
        ],
        axis=1,
    )
    X_test = pd.concat(
        [
            calendar_features(test_df["forecast_kst_dtm"]),
            test_df.drop(columns=["forecast_id", "forecast_kst_dtm"]),
        ],
        axis=1,
    )

    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    X_test = X_test.reindex(columns=X_train.columns)
    return train_df, X_train, X_test, sample_submission


def fit_predict(
    model: object,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_pred: pd.DataFrame,
) -> np.ndarray:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_fit_imp = pd.DataFrame(imputer.fit_transform(X_fit), columns=X_fit.columns, index=X_fit.index)
    X_pred_imp = pd.DataFrame(imputer.transform(X_pred), columns=X_pred.columns, index=X_pred.index)
    model.fit(X_fit_imp, y_fit)
    return np.asarray(model.predict(X_pred_imp), dtype=float)


def score_prediction(y_true: pd.Series, pred: np.ndarray, target: str) -> dict[str, float]:
    clipped = np.clip(pred, 0.0, CAPACITY_KWH[target])
    err = y_true.to_numpy(dtype=float) - clipped
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return {
        "mae": mae,
        "rmse": rmse,
        "nmae_capacity": mae / CAPACITY_KWH[target],
    }


def normalized_inverse_mae_weights(metric_rows: list[dict[str, object]], target: str, model_names: list[str]) -> dict[str, float]:
    maes = {}
    for row in metric_rows:
        if row["target"] == target and row["model"] in model_names:
            maes[str(row["model"])] = float(row["mae"])

    if set(maes) != set(model_names):
        return {name: 1.0 / len(model_names) for name in model_names}

    inv = {name: 1.0 / max(value, 1e-9) for name, value in maes.items()}
    total = sum(inv.values())
    return {name: value / total for name, value in inv.items()}


def make_submission_frame(sample_submission: pd.DataFrame, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    submission = sample_submission[["forecast_id", "forecast_kst_dtm"]].copy()
    for target in TARGET_COLS:
        submission[target] = predictions[target]
    submission["forecast_kst_dtm"] = pd.to_datetime(submission["forecast_kst_dtm"]).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    return submission


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir or (data_dir / "outputs" / ("quick_ensemble" if args.quick else "ensemble"))
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = model_specs(args.quick, args.jobs)
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print("Building features...")
    train_df, X_train, X_test, sample_submission = build_feature_matrices(data_dir)
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    metric_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    final_model_predictions: dict[str, dict[str, np.ndarray]] = {spec.name: {} for spec in specs}
    ensemble_predictions: dict[str, np.ndarray] = {}

    valid_time_mask = train_df["forecast_kst_dtm"] >= VALID_START

    for target_index, target in enumerate(TARGET_COLS):
        labeled_mask = train_df[target].notna()
        train_mask = labeled_mask & ~valid_time_mask
        valid_mask = labeled_mask & valid_time_mask

        if train_mask.sum() == 0 or valid_mask.sum() == 0:
            ordered_idx = train_df.loc[labeled_mask].sort_values("forecast_kst_dtm").index
            split_at = int(len(ordered_idx) * 0.8)
            train_idx = ordered_idx[:split_at]
            valid_idx = ordered_idx[split_at:]
        else:
            train_idx = train_df.index[train_mask]
            valid_idx = train_df.index[valid_mask]

        print(
            f"\nTarget {target}: train={len(train_idx):,}, valid={len(valid_idx):,}, "
            f"final_train={int(labeled_mask.sum()):,}"
        )

        valid_preds: dict[str, np.ndarray] = {}
        for model_index, spec in enumerate(specs):
            seed = args.seed + target_index * 100 + model_index
            print(f"  validation fit: {spec.name}")
            valid_pred = fit_predict(
                spec.build(seed, args.quick),
                X_train.loc[train_idx],
                train_df.loc[train_idx, target],
                X_train.loc[valid_idx],
            )
            valid_pred = np.clip(valid_pred, 0.0, CAPACITY_KWH[target])
            valid_preds[spec.name] = valid_pred
            scores = score_prediction(train_df.loc[valid_idx, target], valid_pred, target)
            metric_rows.append(
                {
                    "target": target,
                    "model": spec.name,
                    "train_rows": int(len(train_idx)),
                    "valid_rows": int(len(valid_idx)),
                    **scores,
                }
            )
            print(
                f"    MAE={scores['mae']:.3f}, RMSE={scores['rmse']:.3f}, "
                f"NMAE(cap)={scores['nmae_capacity']:.5f}"
            )

        model_names = [spec.name for spec in specs]
        if args.ensemble == "weighted":
            weights = normalized_inverse_mae_weights(metric_rows, target, model_names)
        else:
            weights = {name: 1.0 / len(model_names) for name in model_names}

        ensemble_valid = sum(valid_preds[name] * weights[name] for name in model_names)
        ensemble_valid = np.clip(ensemble_valid, 0.0, CAPACITY_KWH[target])
        ensemble_scores = score_prediction(train_df.loc[valid_idx, target], ensemble_valid, target)
        metric_rows.append(
            {
                "target": target,
                "model": f"ensemble_{args.ensemble}",
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
                **ensemble_scores,
            }
        )
        print(
            f"  ensemble validation: MAE={ensemble_scores['mae']:.3f}, "
            f"RMSE={ensemble_scores['rmse']:.3f}, NMAE(cap)={ensemble_scores['nmae_capacity']:.5f}"
        )

        for name, weight in weights.items():
            weight_rows.append({"target": target, "model": name, "weight": weight})

        test_preds_for_target: dict[str, np.ndarray] = {}
        final_idx = train_df.index[labeled_mask]
        for model_index, spec in enumerate(specs):
            seed = args.seed + target_index * 100 + model_index
            print(f"  final fit: {spec.name}")
            test_pred = fit_predict(
                spec.build(seed, args.quick),
                X_train.loc[final_idx],
                train_df.loc[final_idx, target],
                X_test,
            )
            test_pred = np.clip(test_pred, 0.0, CAPACITY_KWH[target])
            final_model_predictions[spec.name][target] = test_pred
            test_preds_for_target[spec.name] = test_pred

        ensemble_test = sum(test_preds_for_target[name] * weights[name] for name in model_names)
        ensemble_predictions[target] = np.clip(ensemble_test, 0.0, CAPACITY_KWH[target])

    metrics = pd.DataFrame(metric_rows)
    weights_df = pd.DataFrame(weight_rows)
    metrics.to_csv(output_dir / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    weights_df.to_csv(output_dir / "ensemble_weights.csv", index=False, encoding="utf-8-sig")
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "data_dir": str(data_dir),
                "quick": args.quick,
                "seed": args.seed,
                "ensemble": args.ensemble,
                "jobs": args.jobs,
                "valid_start": str(VALID_START),
                "features": int(X_train.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for model_name, predictions in final_model_predictions.items():
        model_submission = make_submission_frame(sample_submission, predictions)
        model_submission.to_csv(output_dir / f"submission_{model_name}.csv", index=False, encoding="utf-8-sig")

    ensemble_submission = make_submission_frame(sample_submission, ensemble_predictions)
    ensemble_submission.to_csv(output_dir / "ensemble_submit.csv", index=False, encoding="utf-8-sig")
    ensemble_submission.to_csv(data_dir / "ensemble_submit.csv", index=False, encoding="utf-8-sig")

    print("\nSaved files:")
    print(f"  {output_dir / 'validation_metrics.csv'}")
    print(f"  {output_dir / 'ensemble_weights.csv'}")
    print(f"  {output_dir / 'ensemble_submit.csv'}")
    print(f"  {data_dir / 'ensemble_submit.csv'}")


if __name__ == "__main__":
    main()
