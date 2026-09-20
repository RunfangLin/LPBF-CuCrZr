"""Compare grouped XGBoost models for LPBF CuCrZr relative density.

Two candidate feature sets are compared on identical group-disjoint splits:
``power, speed, lt`` and ``power, speed, hd, lt``. The feature set and its
hyperparameters are selected using outer-CV scores before the winning model is
refitted on the complete training pool and evaluated once on the final test.
Volumetric energy density is not used.

The full database is retained for model development. The accompanying
prediction application restricts use to the specified engineering domain.

The preferred grouping variable is a publication or experiment identifier such
as ``source_id``. If no source column is available, identical process settings
are kept together as a fallback. This prevents replicate leakage but does not
establish generalisation to unseen publications.

Protocol
--------
1. Hold out complete groups for the final test set.
2. Search hyperparameters by group-disjoint outer cross-validation.
3. Within each outer training fold, use a separate group holdout only for
   early stopping. The outer validation fold is used only for scoring.
4. Select the median best boosting round across the winning CV folds.
5. Refit on the complete training pool and evaluate once on the held-out test.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import warnings
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator, Optional

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, KFold
from xgboost import XGBRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
FEATURE_SETS = {
    "three_parameter": ["power", "speed", "lt"],
    "four_parameter": ["power", "speed", "hd", "lt"],
}
GROUPING_FEATURES = ["power", "speed", "hd", "lt"]
TARGET = "density"
APPLICATION_DOMAIN = {
    "power": {"minimum": 300.0, "maximum": 370.0, "unit": "W"},
    "speed": {"minimum": 300.0, "maximum": 500.0, "unit": "mm/s"},
    "hd": {"minimum": 0.09, "maximum": 0.17, "unit": "mm"},
    "lt": {"allowed": [20.0, 30.0], "unit": "um"},
}
SOURCE_COLUMN_CANDIDATES = (
    "source_id",
    "publication_id",
    "paper_id",
    "experiment_id",
    "batch_id",
)


@dataclass
class Config:
    data_path: Path = SCRIPT_DIR / "Density_Data.csv"
    output_dir: Path = SCRIPT_DIR / "Density_Results"
    test_size: float = 0.20
    early_stopping_size: float = 0.15
    random_state: int = 42
    n_splits: int = 5
    early_stopping_rounds: int = 100
    param_grid: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.1, 0.05, 0.02],
            "n_estimators": [600, 900, 1200],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "min_child_weight": [1, 3],
            "reg_lambda": [1.0, 2.0],
            "reg_alpha": [0.0, 0.1],
        }
    )


@dataclass(frozen=True)
class LoadedData:
    features: pd.DataFrame
    target: pd.Series
    groups: pd.Series
    group_strategy: str
    group_column: Optional[str]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def resolve_groups(
    frame: pd.DataFrame,
    requested_column: Optional[str],
    require_source_groups: bool,
) -> tuple[pd.Series, str, Optional[str]]:
    if requested_column:
        if requested_column not in frame.columns:
            raise ValueError(f"Requested group column is missing: {requested_column}")
        group_column = requested_column
    else:
        group_column = next(
            (name for name in SOURCE_COLUMN_CANDIDATES if name in frame.columns),
            None,
        )

    if group_column is not None:
        groups = frame[group_column]
        invalid = groups.isna() | groups.astype(str).str.strip().eq("")
        if invalid.any():
            rows = invalid[invalid].index[:10].tolist()
            raise ValueError(
                f"Group column '{group_column}' is missing at rows: {rows}"
            )
        return groups.astype(str).copy(), f"source column: {group_column}", group_column

    if require_source_groups:
        accepted = ", ".join(SOURCE_COLUMN_CANDIDATES)
        raise ValueError(
            "No source-group column was found. Add one of the following columns "
            f"to the CSV, or omit --require-source-groups: {accepted}"
        )

    warnings.warn(
        "No publication or experiment identifier was found. Identical process "
        "conditions will be grouped together. This prevents replicate leakage "
        "but does not test generalisation to unseen publications.",
        stacklevel=2,
    )
    condition_groups = frame.groupby(
        GROUPING_FEATURES,
        sort=False,
        dropna=False,
    ).ngroup()
    groups = condition_groups.map(lambda value: f"condition_{value:04d}")
    return groups, "identical process condition fallback", None


def load_data(
    path: Path,
    group_column: Optional[str],
    require_source_groups: bool,
) -> LoadedData:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = GROUPING_FEATURES + [TARGET]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(missing)}")

    numeric = frame[required].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Required columns contain missing or non-finite values.")

    groups, strategy, resolved_column = resolve_groups(
        pd.concat(
            [numeric[GROUPING_FEATURES], frame.drop(columns=required)],
            axis=1,
        ),
        group_column,
        require_source_groups,
    )
    groups.index = frame.index
    return LoadedData(
        features=numeric[GROUPING_FEATURES].copy(),
        target=numeric[TARGET].copy(),
        groups=groups,
        group_strategy=strategy,
        group_column=resolved_column,
    )


def group_holdout(
    groups: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError("At least three groups are required for a group holdout.")
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_index, test_index = next(
        splitter.split(np.zeros(len(groups)), groups=groups)
    )
    overlap = set(groups[train_index]).intersection(groups[test_index])
    if overlap:
        raise RuntimeError("Group leakage detected in the holdout split.")
    return train_index, test_index


def iter_group_folds(
    groups: np.ndarray,
    n_splits: int,
    random_state: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.asarray(pd.unique(groups), dtype=object)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} groups; found {len(unique_groups)}."
        )
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_group_index, validation_group_index in splitter.split(unique_groups):
        train_groups = unique_groups[train_group_index]
        validation_groups = unique_groups[validation_group_index]
        train_index = np.flatnonzero(np.isin(groups, train_groups))
        validation_index = np.flatnonzero(np.isin(groups, validation_groups))
        if set(groups[train_index]).intersection(groups[validation_index]):
            raise RuntimeError("Group leakage detected in a CV fold.")
        yield train_index, validation_index


def make_model(
    params: dict[str, Any],
    cfg: Config,
    use_early_stopping: bool,
) -> XGBRegressor:
    options: dict[str, Any] = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": cfg.random_state,
        "n_jobs": -1,
        **params,
    }
    if use_early_stopping:
        options["early_stopping_rounds"] = cfg.early_stopping_rounds
    return XGBRegressor(**options)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    errors = predicted - actual
    return {
        "R2": float(r2_score(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MedianAE": float(np.median(np.abs(errors))),
    }


def fit_with_inner_early_stopping(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    params: dict[str, Any],
    cfg: Config,
    random_state: int,
) -> tuple[XGBRegressor, int]:
    fit_index, stop_index = group_holdout(
        groups,
        cfg.early_stopping_size,
        random_state,
    )
    model = make_model(params, cfg, use_early_stopping=True)
    model.fit(
        X.iloc[fit_index],
        y.iloc[fit_index],
        eval_set=[(X.iloc[stop_index], y.iloc[stop_index])],
        verbose=False,
    )
    best_iteration = getattr(model, "best_iteration", None)
    best_rounds = (
        int(best_iteration) + 1
        if best_iteration is not None
        else int(params["n_estimators"])
    )
    return model, best_rounds


def cv_search(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    cfg: Config,
) -> tuple[dict[str, Any], int, pd.DataFrame, np.ndarray]:
    if not cfg.param_grid or any(not values for values in cfg.param_grid.values()):
        raise ValueError("The parameter grid must contain non-empty value lists.")

    folds = list(iter_group_folds(groups, cfg.n_splits, cfg.random_state))
    fold_assignment = np.full(len(X), -1, dtype=int)
    for fold_number, (_, validation_index) in enumerate(folds, start=1):
        fold_assignment[validation_index] = fold_number

    keys = list(cfg.param_grid)
    combinations = itertools.product(*(cfg.param_grid[key] for key in keys))
    total = math.prod(len(values) for values in cfg.param_grid.values())
    records: list[dict[str, Any]] = []
    best_score = -np.inf
    best_params: Optional[dict[str, Any]] = None
    selected_rounds = 0

    print(f"Grid search: {total} combinations x {cfg.n_splits} outer folds")
    for number, values in enumerate(combinations, start=1):
        params = dict(zip(keys, values))
        scores: list[float] = []
        boosting_rounds: list[int] = []

        for fold_number, (outer_train, outer_validation) in enumerate(folds, start=1):
            model, rounds = fit_with_inner_early_stopping(
                X.iloc[outer_train].reset_index(drop=True),
                y.iloc[outer_train].reset_index(drop=True),
                groups[outer_train],
                params,
                cfg,
                cfg.random_state + fold_number,
            )
            predictions = model.predict(X.iloc[outer_validation])
            score = float(r2_score(y.iloc[outer_validation], predictions))
            scores.append(score)
            boosting_rounds.append(rounds)

        mean_score = float(np.mean(scores))
        median_rounds = int(np.median(boosting_rounds))
        record: dict[str, Any] = {
            **params,
            "mean_outer_R2": mean_score,
            "std_outer_R2": float(np.std(scores)),
            "median_best_rounds": median_rounds,
        }
        record.update(
            {f"fold_{index}_R2": value for index, value in enumerate(scores, 1)}
        )
        record.update(
            {
                f"fold_{index}_best_rounds": value
                for index, value in enumerate(boosting_rounds, 1)
            }
        )
        records.append(record)

        if np.isfinite(mean_score) and mean_score > best_score:
            best_score = mean_score
            best_params = params.copy()
            selected_rounds = median_rounds
        if number == 1 or number % 25 == 0 or number == total:
            print(f"[{number}/{total}] Best outer-CV R2: {best_score:.4f}")

    if best_params is None:
        raise ValueError("Parameter search did not produce a finite score.")
    results = pd.DataFrame(records).sort_values(
        ["mean_outer_R2", "std_outer_R2"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    return best_params, selected_rounds, results, fold_assignment


def save_figure(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_predictions(y_true, y_pred, label: str, output_dir: Path) -> None:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    scores = regression_metrics(actual, predicted)
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(actual, predicted, alpha=0.75)
    axis.plot([low, high], [low, high], "r--", label="Ideal prediction")
    axis.set(
        xlabel="Measured relative density (%)",
        ylabel="Predicted relative density (%)",
        title=f"{label}: R2 = {scores['R2']:.4f}",
    )
    axis.grid(alpha=0.3)
    axis.legend()
    save_figure(figure, output_dir / f"scatter_{label}.png")

    relative_error = np.divide(
        predicted - actual,
        actual,
        out=np.full(actual.shape, np.nan, dtype=float),
        where=actual != 0,
    ) * 100.0
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.bar(np.arange(len(actual)), relative_error, alpha=0.7)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(
        xlabel="Sample index",
        ylabel="Signed relative error (%)",
        title=f"{label}: relative error",
    )
    axis.grid(axis="y", alpha=0.3)
    save_figure(figure, output_dir / f"relative_error_{label}.png")


def plot_model_diagnostics(
    model: XGBRegressor,
    feature_names: list[str],
    fold_rounds: list[int],
    output_dir: Path,
) -> None:
    importance = model.feature_importances_
    order = np.argsort(importance)[::-1]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(np.asarray(feature_names)[order], importance[order])
    axis.set(ylabel="Normalized gain importance", title="Feature importance")
    save_figure(figure, output_dir / "feature_importance.png")

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(np.arange(1, len(fold_rounds) + 1), fold_rounds)
    axis.axhline(
        np.median(fold_rounds),
        color="red",
        linestyle="--",
        label="Median selected rounds",
    )
    axis.set(
        xlabel="Outer CV fold",
        ylabel="Best boosting rounds",
        title="Inner early-stopping results for the selected model",
    )
    axis.legend()
    save_figure(figure, output_dir / "selected_boosting_rounds.png")


def matching_cv_row(
    cv_results: pd.DataFrame,
    best_params: dict[str, Any],
) -> pd.Series:
    mask = np.ones(len(cv_results), dtype=bool)
    for name, value in best_params.items():
        mask &= cv_results[name].eq(value).to_numpy()
    if not mask.any():
        raise RuntimeError("The selected parameter row is missing from CV results.")
    return cv_results.loc[mask].iloc[0]


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    groups_test: np.ndarray,
    best_params: dict[str, Any],
    selected_rounds: int,
    cv_results: pd.DataFrame,
    feature_names: list[str],
    cfg: Config,
) -> tuple[XGBRegressor, dict[str, Any]]:
    final_params = best_params.copy()
    final_params["n_estimators"] = selected_rounds
    model = make_model(final_params, cfg, use_early_stopping=False)
    model.fit(X_train, y_train, verbose=False)

    subsets = {
        "training_in_sample": (X_train, y_train, groups_train),
        "test": (X_test, y_test, groups_test),
    }
    metrics: dict[str, dict[str, float]] = {}
    prediction_tables: list[pd.DataFrame] = []
    for name, (X_part, y_part, group_part) in subsets.items():
        predictions = model.predict(X_part)
        metrics[name] = regression_metrics(y_part, predictions)
        table = X_part.copy()
        table.insert(0, "source_row", X_part.index)
        table.insert(1, "subset", name)
        table.insert(2, "group_id", group_part)
        table["measured_density"] = y_part.to_numpy()
        table["predicted_density"] = predictions
        table["error_percentage_points"] = predictions - y_part.to_numpy()
        prediction_tables.append(table)
        plot_predictions(y_part, predictions, name, cfg.output_dir)

    pd.concat(prediction_tables).to_csv(
        cfg.output_dir / "predictions.csv", index=False
    )
    joblib.dump(model, cfg.output_dir / "density_model.pkl")

    best_cv_row = matching_cv_row(cv_results, best_params)
    fold_rounds = [
        int(best_cv_row[f"fold_{fold}_best_rounds"])
        for fold in range(1, cfg.n_splits + 1)
    ]
    plot_model_diagnostics(model, feature_names, fold_rounds, cfg.output_dir)
    return model, {
        "metrics": metrics,
        "final_params": final_params,
        "selected_boosting_rounds": selected_rounds,
        "selected_fold_rounds": fold_rounds,
    }


def write_summary(
    cfg: Config,
    data: LoadedData,
    train_index: np.ndarray,
    test_index: np.ndarray,
    selected_model: str,
    selected_features: list[str],
    model_comparison: list[dict[str, Any]],
    best_params: dict[str, Any],
    training_result: dict[str, Any],
) -> dict[str, Any]:
    train_groups = set(data.groups.iloc[train_index])
    test_groups = set(data.groups.iloc[test_index])
    summary: dict[str, Any] = {
        "data_file": str(cfg.data_path),
        "data_sha256": file_sha256(cfg.data_path),
        "candidate_feature_sets": FEATURE_SETS,
        "selected_model": selected_model,
        "selected_features": selected_features,
        "model_comparison": model_comparison,
        "application_domain": APPLICATION_DOMAIN,
        "target": TARGET,
        "grouping": {
            "strategy": data.group_strategy,
            "source_column": data.group_column,
            "total_groups": int(data.groups.nunique()),
            "training_groups": len(train_groups),
            "test_groups": len(test_groups),
            "train_test_group_overlap": len(train_groups.intersection(test_groups)),
        },
        "split": {
            "random_state": cfg.random_state,
            "requested_test_group_fraction": cfg.test_size,
            "total_rows": len(data.features),
            "training_rows": len(train_index),
            "test_rows": len(test_index),
        },
        "search": {
            "outer_group_folds": cfg.n_splits,
            "inner_early_stopping_group_fraction": cfg.early_stopping_size,
            "early_stopping_rounds": cfg.early_stopping_rounds,
            "hyperparameter_selection_metric": "mean outer-fold R2",
            "feature_set_selection_metric": "best mean outer-fold R2",
            "param_grid": cfg.param_grid,
        },
        "best_search_params": best_params,
        **training_result,
        "software": {
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scikit-learn": package_version("scikit-learn"),
            "xgboost": package_version("xgboost"),
        },
        "notes": [
            "Outer CV validation groups are never used for early stopping.",
            "Both feature sets use identical train-test groups and outer CV folds.",
            "The feature set is selected before the final test is evaluated.",
            "The final test groups are never used for model selection or early stopping.",
            "The final model is refitted on the complete training pool.",
            "Training metrics are in-sample and are not generalisation estimates.",
            "If process-condition fallback grouping is used, the test does not establish generalisation to unseen publications.",
            "The three-parameter model uses power, speed, and layer thickness.",
            "The four-parameter model additionally uses hatch spacing.",
            "VED is not included in either model.",
            "The application domain constrains deployment inputs; it does not filter the 81-row training database.",
            "Feature importance is predictive rather than causal.",
        ],
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
    (cfg.output_dir / "results_summary.json").write_text(
        text + "\n", encoding="utf-8"
    )
    lines = [
        "SELECTED MODEL",
        selected_model,
        "\nSELECTED FEATURES",
        json.dumps(selected_features),
        "\nMODEL COMPARISON",
        json.dumps(model_comparison, indent=2),
        "\nBEST SEARCH PARAMETERS",
        json.dumps(best_params, indent=2),
        "\nFINAL PARAMETERS",
        json.dumps(training_result["final_params"], indent=2),
    ]
    for name, scores in training_result["metrics"].items():
        lines.extend([f"\n{name.upper()} METRICS", json.dumps(scores, indent=2)])
    lines.extend(["\nNOTES", *summary["notes"]])
    (cfg.output_dir / "results_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=SCRIPT_DIR / "Density_Data.csv",
        help="CSV path (default: Density_Data.csv beside this script)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "Density_Results",
        help="Output directory",
    )
    parser.add_argument(
        "--group-column",
        default=None,
        help="Publication or experiment identifier column; detected automatically when omitted",
    )
    parser.add_argument(
        "--require-source-groups",
        action="store_true",
        help="Fail instead of falling back to identical-process-condition groups",
    )
    args = parser.parse_args()

    cfg = Config(
        data_path=args.data.expanduser().resolve(),
        output_dir=args.output.expanduser().resolve(),
    )
    data = load_data(
        cfg.data_path,
        args.group_column,
        args.require_source_groups,
    )
    all_groups = data.groups.to_numpy(dtype=object)
    train_index, test_index = group_holdout(
        all_groups,
        cfg.test_size,
        cfg.random_state,
    )

    X_train_all = data.features.iloc[train_index]
    y_train = data.target.iloc[train_index]
    groups_train = all_groups[train_index]
    X_test_all = data.features.iloc[test_index]
    y_test = data.target.iloc[test_index]
    groups_test = all_groups[test_index]

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data: {cfg.data_path}")
    print(
        f"Rows: {len(data.features)} | Training: {len(X_train_all)} | "
        f"Test: {len(X_test_all)}"
    )
    print(
        f"Grouping: {data.group_strategy} | Total groups: "
        f"{data.groups.nunique()}"
    )
    print(f"Candidate feature sets: {FEATURE_SETS}")
    print(f"Output: {cfg.output_dir}")

    runs: dict[str, dict[str, Any]] = {}
    comparison_rows: list[dict[str, Any]] = []
    common_fold_assignment: Optional[np.ndarray] = None

    for model_name, feature_names in FEATURE_SETS.items():
        print(f"\nSearching {model_name}: {', '.join(feature_names)}")
        best_params, selected_rounds, cv_results, fold_assignment = cv_search(
            X_train_all[feature_names],
            y_train,
            groups_train,
            cfg,
        )
        cv_results.to_csv(
            cfg.output_dir / f"cv_results_{model_name}.csv",
            index=False,
        )

        if common_fold_assignment is None:
            common_fold_assignment = fold_assignment
        elif not np.array_equal(common_fold_assignment, fold_assignment):
            raise RuntimeError("Feature sets received different outer CV folds.")

        best_row = matching_cv_row(cv_results, best_params)
        comparison_rows.append(
            {
                "model": model_name,
                "feature_count": len(feature_names),
                "features": ", ".join(feature_names),
                "best_mean_outer_R2": float(best_row["mean_outer_R2"]),
                "best_std_outer_R2": float(best_row["std_outer_R2"]),
                "selected_boosting_rounds": int(selected_rounds),
            }
        )
        runs[model_name] = {
            "features": feature_names,
            "best_params": best_params,
            "selected_rounds": selected_rounds,
            "cv_results": cv_results,
        }

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["best_mean_outer_R2", "best_std_outer_R2", "feature_count"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    comparison.insert(0, "rank", np.arange(1, len(comparison) + 1))
    comparison.to_csv(cfg.output_dir / "model_comparison.csv", index=False)

    selected_model = str(comparison.loc[0, "model"])
    selected_run = runs[selected_model]
    selected_features = list(selected_run["features"])
    best_params = selected_run["best_params"]
    selected_rounds = int(selected_run["selected_rounds"])
    cv_results = selected_run["cv_results"]
    cv_results.to_csv(cfg.output_dir / "cv_results.csv", index=False)

    if common_fold_assignment is None:
        raise RuntimeError("No CV fold assignment was generated.")
    pd.DataFrame(
        {
            "source_row": X_train_all.index,
            "group_id": groups_train,
            "outer_cv_fold": common_fold_assignment,
        }
    ).to_csv(cfg.output_dir / "cv_fold_assignments.csv", index=False)

    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    print(f"\nSelected model: {selected_model} ({', '.join(selected_features)})")

    _, training_result = train_final_model(
        X_train_all[selected_features],
        y_train,
        groups_train,
        X_test_all[selected_features],
        y_test,
        groups_test,
        best_params,
        selected_rounds,
        cv_results,
        selected_features,
        cfg,
    )
    comparison_records = json.loads(comparison.to_json(orient="records"))
    summary = write_summary(
        cfg,
        data,
        train_index,
        test_index,
        selected_model,
        selected_features,
        comparison_records,
        best_params,
        training_result,
    )

    print("\nFinal model metrics:")
    for name, scores in summary["metrics"].items():
        values = " | ".join(f"{key}={value:.4f}" for key, value in scores.items())
        print(f"  {name:18s} {values}")
    print(f"\nSaved results to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
