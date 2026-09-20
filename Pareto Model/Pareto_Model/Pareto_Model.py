"""Pareto active learning for LPBF CuCrZr (UTS and electrical conductivity).

The script trains two independent heteroscedastic Gaussian-process models,
combines them in a two-objective posterior, evaluates a discrete candidate
space with analytic EHVI, and exports the ranked candidates and diagnostics.

Expected project layout
-----------------------
Pareto Model/
|-- Data/
|   |-- data.csv
|   |-- UTSonly.csv
|   |-- EConly.csv
|   |-- generate_test.csv
`-- Parete_Model/
    `-- Pareto_Model.py   <- this file

Input CSV files should remain in physical units. BoTorch normalizes model
inputs and standardizes each output internally.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.multi_objective.analytic import (
    ExpectedHypervolumeImprovement,
)
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls.exact_marginal_log_likelihood import (
    ExactMarginalLogLikelihood,
)
from gpytorch.priors import GammaPrior


# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "Data"


@dataclass(frozen=True)
class Config:
    train_csv: Path = DATA_DIR / "data.csv"
    candidate_csv: Path = DATA_DIR / "generate_test.csv"
    uts_reference_csv: Path = DATA_DIR / "UTSonly.csv"
    ec_reference_csv: Path = DATA_DIR / "EConly.csv"
    output_dir: Path = PROJECT_DIR / "Output" / "Pareto_Result"

    use_uts_reference: bool = True
    use_ec_reference: bool = True
    use_physics_features: bool = True
    use_heteroscedastic_noise: bool = True

    show_reference_points: bool = True
    show_all_candidates: bool = True
    save_model_state: bool = True

    top_k: int = 5
    seed: int = 42
    cv_folds: int = 5
    knn_neighbors: int = 7
    prediction_batch_size: int = 8192

    temperature_is_kelvin: bool = True
    time_is_minutes: bool = True
    kelvin_offset: float = 273.15

    reference_uts_std: float = 60.0
    reference_ec_std: float = 8.0
    default_uts_noise_std: float = 50.0
    default_ec_noise_std: float = 5.0


CONFIG = Config()

BASE_FEATURES = ("power", "speed", "VED", "T", "t")
META_COLUMNS = (
    "_var_UTS",
    "_var_EC",
    "_is_ref_uts",
    "_is_ref_ec",
)

FEATURE_ALIASES = {
    "power": ("power", "P", "Power", "Power_W", "Power (W)"),
    "speed": ("speed", "v", "Speed", "speed_mm_s", "speed (mm/s)"),
    "VED": (
        "VED",
        "volume",
        "VD",
        "VD_J_mm3",
        "VED_Jmm3",
        "volume energy density",
    ),
    "T": ("T", "HT_T", "Temp", "Temperature", "Temp_C", "T(°C)", "T_C"),
    "t": ("t", "HT_t", "time", "Time", "duration"),
}

TARGET_ALIASES = {
    "UTS": ("UTS", "sigma_UTS", "strength", "UTS (MPa)"),
    "EC": (
        "EC",
        "%IACS",
        "electrical_conductivity",
        "EC_MS_m",
        "Conductivity",
        "EC (%)",
    ),
}


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def log(message: str) -> None:
    print(f"[INFO] {message}")


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    require_file(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"CSV file contains no rows: {path}")
    return frame


def find_column(frame: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    aliases = tuple(aliases)
    for alias in aliases:
        if alias in frame.columns:
            return alias

    lower_map = {str(column).lower(): column for column in frame.columns}
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]

    # Substring matching is retained for compatibility, but only for aliases
    # long enough to avoid accidental matches for one-character names.
    for alias in aliases:
        if len(alias) < 3:
            continue
        for column in frame.columns:
            if alias.lower() in str(column).lower():
                return column
    return None


def canonical_features(frame: pd.DataFrame) -> pd.DataFrame:
    columns: dict[str, str] = {}
    for canonical_name, aliases in FEATURE_ALIASES.items():
        column = find_column(frame, aliases)
        if column is None:
            raise ValueError(
                f"Could not identify feature '{canonical_name}'. "
                f"Accepted aliases: {aliases}. Available columns: {list(frame.columns)}"
            )
        columns[canonical_name] = column

    output = pd.DataFrame(index=frame.index)
    for canonical_name in BASE_FEATURES:
        output[canonical_name] = pd.to_numeric(
            frame[columns[canonical_name]], errors="coerce"
        )
    return output


def canonical_targets(frame: pd.DataFrame, require_both: bool = True) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index, columns=("UTS", "EC"), dtype=float)
    for target_name, aliases in TARGET_ALIASES.items():
        column = find_column(frame, aliases)
        if column is None:
            if require_both:
                raise ValueError(
                    f"Could not identify target '{target_name}' in columns "
                    f"{list(frame.columns)}"
                )
            output[target_name] = np.nan
        else:
            output[target_name] = pd.to_numeric(frame[column], errors="coerce")
    return output


def validate_features(frame: pd.DataFrame, label: str) -> None:
    values = frame.loc[:, BASE_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad_rows = np.where(~np.isfinite(values).all(axis=1))[0][:10].tolist()
        raise ValueError(f"{label} contains invalid feature values at rows {bad_rows}.")


# ---------------------------------------------------------------------------
# Training-data assembly and feature engineering
# ---------------------------------------------------------------------------

def make_base_training_frame(
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.concat(
        [features.reset_index(drop=True), targets.reset_index(drop=True)], axis=1
    )
    for column in META_COLUMNS:
        frame[column] = np.nan if column.startswith("_var") else 0
    return frame


def load_single_target_reference(
    path: Path,
    target: str,
    config: Config,
) -> pd.DataFrame:
    raw = read_csv(path)
    features = canonical_features(raw)
    validate_features(features, path.name)

    target_column = find_column(raw, TARGET_ALIASES[target])
    if target_column is None:
        raise ValueError(f"{path.name} does not contain the required {target} column.")
    target_values = pd.to_numeric(raw[target_column], errors="coerce")

    valid = features.notna().all(axis=1) & target_values.notna()
    frame = features.loc[valid].reset_index(drop=True)
    frame["UTS"] = np.nan
    frame["EC"] = np.nan
    frame[target] = target_values.loc[valid].to_numpy(dtype=float)
    frame["_var_UTS"] = (
        config.reference_uts_std**2 if target == "UTS" else np.nan
    )
    frame["_var_EC"] = (
        config.reference_ec_std**2 if target == "EC" else np.nan
    )
    frame["_is_ref_uts"] = int(target == "UTS")
    frame["_is_ref_ec"] = int(target == "EC")
    return frame


def engineer_features(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    features = frame.loc[:, BASE_FEATURES].copy()
    if not config.use_physics_features:
        return features

    temperature_c = (
        features["T"] - config.kelvin_offset
        if config.temperature_is_kelvin
        else features["T"]
    )
    time_h = features["t"] / 60.0 if config.time_is_minutes else features["t"]

    # The as-built state (t=0) has no ageing temperature. Setting its thermal
    # terms to zero avoids artificial values from division by a near-zero time.
    aged = time_h > 0
    effective_temperature_c = temperature_c.where(aged, 0.0)

    features["LED"] = features["power"] / np.clip(features["speed"], 1e-6, None)
    features["VEDxT"] = features["VED"] * effective_temperature_c
    features["VEDxt"] = features["VED"] * time_h
    features["T_div_t"] = np.divide(
        effective_temperature_c,
        time_h,
        out=np.zeros(len(features), dtype=float),
        where=aged,
    )
    features["T_sq"] = effective_temperature_c**2
    features["t_sq"] = time_h**2
    features["VED_div_T"] = np.divide(
        features["VED"],
        effective_temperature_c,
        out=np.zeros(len(features), dtype=float),
        where=aged & (np.abs(effective_temperature_c) > 1e-12),
    )

    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Feature engineering produced non-finite values.")
    return features


# ---------------------------------------------------------------------------
# Heteroscedastic noise and GP models
# ---------------------------------------------------------------------------

def estimate_heteroscedastic_noise(
    x: np.ndarray,
    y: np.ndarray,
    config: Config,
) -> Optional[np.ndarray]:
    if not config.use_heteroscedastic_noise or len(x) < 5:
        return None

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import KFold
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        warnings.warn(
            f"scikit-learn is unavailable; using constant noise instead: {error}"
        )
        return None

    x_scaled = StandardScaler().fit_transform(x)
    n_splits = max(2, min(config.cv_folds, len(x) // 2))
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=config.seed)

    oof_prediction = np.empty(len(x), dtype=float)
    for train_index, validation_index in folds.split(x_scaled):
        regressor = RandomForestRegressor(
            n_estimators=400,
            random_state=config.seed,
            n_jobs=-1,
        )
        regressor.fit(x_scaled[train_index], y[train_index])
        oof_prediction[validation_index] = regressor.predict(x_scaled[validation_index])

    squared_residual = (y - oof_prediction) ** 2
    neighbors = max(2, min(config.knn_neighbors, len(x)))
    smoother = KNeighborsRegressor(
        n_neighbors=neighbors,
        weights="distance",
        p=2,
    )
    smoother.fit(x_scaled, squared_residual)
    variance = smoother.predict(x_scaled)

    lower, upper = np.percentile(variance, (1, 97))
    return np.maximum(np.clip(variance, lower, upper), 1e-9)


def build_noise_vector(
    x: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    variance_column: str,
    default_std: float,
    config: Config,
) -> np.ndarray:
    variance = estimate_heteroscedastic_noise(
        x.to_numpy(dtype=float), y.to_numpy(dtype=float), config
    )
    if variance is None:
        variance = np.full(len(x), default_std**2, dtype=float)

    supplied = pd.to_numeric(metadata[variance_column], errors="coerce").to_numpy()
    supplied_mask = np.isfinite(supplied)
    variance[supplied_mask] = supplied[supplied_mask]
    return np.maximum(variance, 1e-9)


def make_mixed_kernel(dimension: int) -> ScaleKernel:
    matern = MaternKernel(
        nu=2.5,
        ard_num_dims=dimension,
        lengthscale_prior=GammaPrior(3.0, 6.0),
    )
    rbf = RBFKernel(
        ard_num_dims=dimension,
        lengthscale_prior=GammaPrior(3.0, 6.0),
    )
    return ScaleKernel(
        matern + rbf,
        outputscale_prior=GammaPrior(2.0, 0.5),
    )


def make_shared_normalization_bounds(
    training_inputs: pd.DataFrame,
    candidate_inputs: pd.DataFrame,
) -> torch.Tensor:
    """Build one fixed input transform shared by both GP submodels."""
    minimum = np.minimum(
        training_inputs.min(axis=0).to_numpy(dtype=float),
        candidate_inputs.min(axis=0).to_numpy(dtype=float),
    )
    maximum = np.maximum(
        training_inputs.max(axis=0).to_numpy(dtype=float),
        candidate_inputs.max(axis=0).to_numpy(dtype=float),
    )
    constant = maximum <= minimum
    maximum[constant] = minimum[constant] + 1.0
    return torch.as_tensor(np.vstack([minimum, maximum]), dtype=torch.double)


def fit_gp(
    x: torch.Tensor,
    y: torch.Tensor,
    y_variance: torch.Tensor,
    normalization_bounds: torch.Tensor,
) -> SingleTaskGP:
    dimension = x.shape[-1]
    model = SingleTaskGP(
        train_X=x,
        train_Y=y,
        train_Yvar=y_variance,
        covar_module=make_mixed_kernel(dimension),
        input_transform=Normalize(d=dimension, bounds=normalization_bounds.clone()),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def kernel_diagnostics(model: SingleTaskGP) -> dict[str, Any]:
    base_kernel = model.covar_module.base_kernel
    matern, rbf = base_kernel.kernels
    return {
        "outputscale": float(model.covar_module.outputscale.detach().cpu()),
        "matern_lengthscales": matern.lengthscale.detach().cpu().flatten().tolist(),
        "rbf_lengthscales": rbf.lengthscale.detach().cpu().flatten().tolist(),
    }


def ard_sensitivity(
    model: SingleTaskGP,
    feature_names: list[str],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    kernels = model.covar_module.base_kernel.kernels
    for kernel in kernels:
        label = kernel.__class__.__name__.replace("Kernel", "")
        lengthscales = kernel.lengthscale.detach().cpu().numpy().reshape(-1)
        inverse = 1.0 / np.maximum(lengthscales, 1e-12)
        normalized = inverse / inverse.max()
        result[label] = {
            feature: float(value)
            for feature, value in zip(feature_names, normalized)
        }
    return result


def regression_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    residual = observed - predicted
    denominator = np.sum((observed - observed.mean()) ** 2)
    r2 = np.nan if denominator == 0 else 1.0 - np.sum(residual**2) / denominator
    rmse = np.sqrt(np.mean(residual**2))
    mape = np.mean(
        np.abs(residual) / np.clip(np.abs(observed), 1e-8, None)
    ) * 100.0
    return {"R2": float(r2), "RMSE": float(rmse), "MAPE": float(mape)}


def posterior_arrays(
    model: SingleTaskGP,
    x: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        posterior = model.posterior(x)
        mean = posterior.mean.squeeze(-1).detach().cpu().numpy()
        std = posterior.variance.clamp_min(1e-18).sqrt().squeeze(-1)
        return mean, std.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Candidate scoring and Pareto filtering
# ---------------------------------------------------------------------------

def score_candidates(
    gp_uts: SingleTaskGP,
    gp_ec: SingleTaskGP,
    acquisition: ExpectedHypervolumeImprovement,
    candidates: torch.Tensor,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means: list[np.ndarray] = []
    standard_deviations: list[np.ndarray] = []
    scores: list[np.ndarray] = []

    for start in range(0, len(candidates), batch_size):
        stop = min(start + batch_size, len(candidates))
        batch = candidates[start:stop]
        with torch.no_grad():
            uts_posterior = gp_uts.posterior(batch)
            ec_posterior = gp_ec.posterior(batch)
            means.append(
                torch.stack(
                    [
                        uts_posterior.mean.squeeze(-1),
                        ec_posterior.mean.squeeze(-1),
                    ],
                    dim=1,
                )
                .cpu()
                .numpy()
            )
            standard_deviations.append(
                torch.stack(
                    [
                        uts_posterior.variance.squeeze(-1).clamp_min(1e-18).sqrt(),
                        ec_posterior.variance.squeeze(-1).clamp_min(1e-18).sqrt(),
                    ],
                    dim=1,
                )
                .cpu()
                .numpy()
            )
            scores.append(
                acquisition(batch.unsqueeze(-2)).reshape(-1).cpu().numpy()
            )
        log(f"Scored candidates {start:,}-{stop - 1:,} of {len(candidates):,}.")

    return (
        np.concatenate(means, axis=0),
        np.concatenate(standard_deviations, axis=0),
        np.concatenate(scores, axis=0),
    )


def non_dominated_mask_2d(points: np.ndarray) -> np.ndarray:
    """Return the non-dominated mask for two objectives that are maximized.

    The implementation is O(n log n), avoiding the O(n^2) pairwise comparison
    used by the historical script.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Expected an (n, 2) array of objective values.")
    if not np.isfinite(points).all():
        raise ValueError("Objective values contain NaN or infinity.")

    # Primary sort: objective 0 descending. Secondary: objective 1 descending.
    order = np.lexsort((-points[:, 1], -points[:, 0]))
    sorted_points = points[order]
    sorted_mask = np.zeros(len(points), dtype=bool)
    best_second = -np.inf

    index = 0
    while index < len(sorted_points):
        end = index + 1
        first_value = sorted_points[index, 0]
        while end < len(sorted_points) and sorted_points[end, 0] == first_value:
            end += 1

        group_second = sorted_points[index:end, 1]
        group_max = float(group_second.max())
        if group_max > best_second:
            sorted_mask[index:end] = group_second == group_max
        best_second = max(best_second, group_max)
        index = end

    mask = np.zeros(len(points), dtype=bool)
    mask[order] = sorted_mask
    return mask


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def save_feature_sensitivity(
    sensitivities: dict[str, dict[str, dict[str, float]]],
    feature_names: list[str],
    output_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for target, kernel_data in sensitivities.items():
        for kernel, feature_data in kernel_data.items():
            for feature, value in feature_data.items():
                rows.append(
                    {
                        "target": target,
                        "kernel": kernel,
                        "feature": feature,
                        "normalized_inverse_lengthscale": value,
                    }
                )
    pd.DataFrame(rows).to_csv(output_dir / "feature_sensitivity.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for axis, target in zip(axes, ("UTS", "EC")):
        kernels = sensitivities[target]
        x_position = np.arange(len(feature_names))
        width = 0.8 / max(1, len(kernels))
        for index, (kernel, values) in enumerate(kernels.items()):
            heights = [values[feature] for feature in feature_names]
            offset = (index - (len(kernels) - 1) / 2) * width
            axis.bar(x_position + offset, heights, width=width, label=kernel)
        axis.set_title(f"{target} ARD sensitivity")
        axis.set_xticks(x_position)
        axis.set_xticklabels(feature_names, rotation=55, ha="right")
        axis.set_ylabel("Normalized inverse lengthscale")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_pareto_results(
    observed: np.ndarray,
    candidate_mean: np.ndarray,
    candidate_std: np.ndarray,
    selected_indices: list[int],
    candidate_features: pd.DataFrame,
    reference_frame: pd.DataFrame,
    output_dir: Path,
    config: Config,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 8))
    axis.scatter(
        observed[:, 1],
        observed[:, 0],
        alpha=0.85,
        s=50,
        label="Paired UTS-EC observations",
    )

    if config.show_all_candidates:
        axis.scatter(
            candidate_mean[:, 1],
            candidate_mean[:, 0],
            alpha=0.08,
            s=14,
            color="grey",
            label=f"Candidate posterior means ({len(candidate_mean):,})",
        )

    if config.show_reference_points and not reference_frame.empty:
        uts_reference = reference_frame[reference_frame["_is_ref_uts"] == 1]
        ec_reference = reference_frame[reference_frame["_is_ref_ec"] == 1]
        if not uts_reference.empty:
            ec_placeholder = float(np.nanmedian(observed[:, 1]))
            axis.scatter(
                np.full(len(uts_reference), ec_placeholder),
                uts_reference["UTS"],
                alpha=0.35,
                marker="o",
                label="UTS-only references (shown at median EC)",
            )
        if not ec_reference.empty:
            uts_placeholder = float(np.nanmedian(observed[:, 0]))
            axis.scatter(
                ec_reference["EC"],
                np.full(len(ec_reference), uts_placeholder),
                alpha=0.35,
                marker="s",
                label="EC-only references (shown at median UTS)",
            )

    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(selected_indices))))
    markers = ("^", "s", "o", "D", "*", "P", "X", "v")
    for rank, candidate_index in enumerate(selected_indices, start=1):
        row = candidate_features.iloc[candidate_index]
        uts_mean, ec_mean = candidate_mean[candidate_index]
        uts_std, ec_std = candidate_std[candidate_index]
        temperature_c = (
            row["T"] - config.kelvin_offset
            if config.temperature_is_kelvin
            else row["T"]
        )
        time_h = row["t"] / 60.0 if config.time_is_minutes else row["t"]

        axis.errorbar(
            ec_mean,
            uts_mean,
            xerr=ec_std,
            yerr=uts_std,
            color=colors[rank - 1],
            marker=markers[(rank - 1) % len(markers)],
            markersize=9,
            capsize=3,
            linestyle="none",
            label=(
                f"Rank {rank}: P={row['power']:.0f} W, v={row['speed']:.0f} mm/s, "
                f"VED={row['VED']:.1f}\n"
                f"T={temperature_c:.0f} deg C, t={time_h:.1f} h; "
                f"UTS={uts_mean:.1f}+/-{uts_std:.1f} MPa, "
                f"EC={ec_mean:.1f}+/-{ec_std:.1f}% IACS"
            ),
        )

    axis.set_title(
        "Pareto active learning: UTS versus electrical conductivity\n"
        "Heteroscedastic SingleTaskGP (fixed observation noise) with analytic EHVI"
    )
    axis.set_xlabel("Electrical conductivity (% IACS)")
    axis.set_ylabel("Ultimate tensile strength (MPa)")
    axis.grid(alpha=0.3)
    axis.legend(bbox_to_anchor=(1.03, 1.0), loc="upper left", fontsize=8)
    figure.tight_layout()
    figure.savefig(
        output_dir / "pareto_candidates_real_units.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def serializable_config(config: Config) -> dict[str, Any]:
    output = asdict(config)
    for key, value in output.items():
        if isinstance(value, Path):
            output[key] = str(value)
    return output


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main(config: Config = CONFIG) -> None:
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_default_dtype(torch.double)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    log("Loading paired observations and candidate space.")
    paired_raw = read_csv(config.train_csv)
    candidate_raw = read_csv(config.candidate_csv)
    paired_features = canonical_features(paired_raw)
    paired_targets = canonical_targets(paired_raw, require_both=True)
    candidate_features = canonical_features(candidate_raw)

    validate_features(paired_features, config.train_csv.name)
    validate_features(candidate_features, config.candidate_csv.name)

    training_frames = [make_base_training_frame(paired_features, paired_targets)]

    reference_frames: list[pd.DataFrame] = []
    if config.use_uts_reference:
        uts_reference = load_single_target_reference(
            config.uts_reference_csv, "UTS", config
        )
        training_frames.append(uts_reference)
        reference_frames.append(uts_reference)
    if config.use_ec_reference:
        ec_reference = load_single_target_reference(
            config.ec_reference_csv, "EC", config
        )
        training_frames.append(ec_reference)
        reference_frames.append(ec_reference)

    training = pd.concat(training_frames, ignore_index=True, sort=False)
    references = (
        pd.concat(reference_frames, ignore_index=True, sort=False)
        if reference_frames
        else pd.DataFrame(columns=training.columns)
    )

    model_inputs = engineer_features(training, config)
    candidate_model_inputs = engineer_features(candidate_features, config)
    feature_names = model_inputs.columns.tolist()
    if feature_names != candidate_model_inputs.columns.tolist():
        raise RuntimeError("Training and candidate feature columns are inconsistent.")

    uts_mask = training["UTS"].notna()
    ec_mask = training["EC"].notna()
    paired_mask = uts_mask & ec_mask
    if not uts_mask.any() or not ec_mask.any() or not paired_mask.any():
        raise ValueError("Training data must contain UTS, EC, and paired UTS-EC rows.")

    x_uts_frame = model_inputs.loc[uts_mask].reset_index(drop=True)
    x_ec_frame = model_inputs.loc[ec_mask].reset_index(drop=True)
    y_uts_series = training.loc[uts_mask, "UTS"].reset_index(drop=True)
    y_ec_series = training.loc[ec_mask, "EC"].reset_index(drop=True)

    uts_noise = build_noise_vector(
        x_uts_frame,
        y_uts_series,
        training.loc[uts_mask].reset_index(drop=True),
        "_var_UTS",
        config.default_uts_noise_std,
        config,
    )
    ec_noise = build_noise_vector(
        x_ec_frame,
        y_ec_series,
        training.loc[ec_mask].reset_index(drop=True),
        "_var_EC",
        config.default_ec_noise_std,
        config,
    )

    # Explicit copies avoid exposing read-only pandas/NumPy buffers to PyTorch.
    x_uts = torch.tensor(
        x_uts_frame.to_numpy(dtype=float, copy=True), dtype=torch.double
    )
    y_uts = torch.tensor(
        y_uts_series.to_numpy(dtype=float, copy=True)[:, None], dtype=torch.double
    )
    yvar_uts = torch.tensor(uts_noise[:, None].copy(), dtype=torch.double)
    x_ec = torch.tensor(
        x_ec_frame.to_numpy(dtype=float, copy=True), dtype=torch.double
    )
    y_ec = torch.tensor(
        y_ec_series.to_numpy(dtype=float, copy=True)[:, None], dtype=torch.double
    )
    yvar_ec = torch.tensor(ec_noise[:, None].copy(), dtype=torch.double)
    x_candidates = torch.tensor(
        candidate_model_inputs.to_numpy(dtype=float, copy=True), dtype=torch.double
    )
    normalization_bounds = make_shared_normalization_bounds(
        model_inputs, candidate_model_inputs
    )

    log(
        f"Training rows: total={len(training)}, UTS={len(x_uts)}, "
        f"EC={len(x_ec)}, paired={int(paired_mask.sum())}."
    )
    log(f"Candidate rows: {len(x_candidates):,}; model features: {feature_names}.")

    log("Fitting the UTS Gaussian process.")
    gp_uts = fit_gp(x_uts, y_uts, yvar_uts, normalization_bounds)
    log("Fitting the EC Gaussian process.")
    gp_ec = fit_gp(x_ec, y_ec, yvar_ec, normalization_bounds)
    model_list = ModelListGP(gp_uts, gp_ec)
    model_list.eval()

    uts_train_prediction, uts_train_std = posterior_arrays(gp_uts, x_uts)
    ec_train_prediction, ec_train_std = posterior_arrays(gp_ec, x_ec)
    model_performance = {
        "UTS_submodel_in_sample": regression_metrics(
            y_uts_series.to_numpy(dtype=float), uts_train_prediction
        ),
        "EC_submodel_in_sample": regression_metrics(
            y_ec_series.to_numpy(dtype=float), ec_train_prediction
        ),
    }

    paired_inputs = torch.as_tensor(
        model_inputs.loc[paired_mask].to_numpy(dtype=float)
    )
    paired_observed = training.loc[paired_mask, ["UTS", "EC"]].to_numpy(dtype=float)
    paired_uts_prediction, _ = posterior_arrays(gp_uts, paired_inputs)
    paired_ec_prediction, _ = posterior_arrays(gp_ec, paired_inputs)
    model_performance["paired_rows_in_sample"] = {
        "UTS": regression_metrics(paired_observed[:, 0], paired_uts_prediction),
        "EC": regression_metrics(paired_observed[:, 1], paired_ec_prediction),
    }

    uts_z = (
        y_uts_series.to_numpy(dtype=float) - uts_train_prediction
    ) / np.clip(uts_train_std, 1e-9, None)
    ec_z = (
        y_ec_series.to_numpy(dtype=float) - ec_train_prediction
    ) / np.clip(ec_train_std, 1e-9, None)
    calibration = {
        "UTS_abs_z_median": float(np.median(np.abs(uts_z))),
        "UTS_abs_z_p95": float(np.percentile(np.abs(uts_z), 95)),
        "EC_abs_z_median": float(np.median(np.abs(ec_z))),
        "EC_abs_z_p95": float(np.percentile(np.abs(ec_z), 95)),
    }

    observed_tensor = torch.as_tensor(paired_observed)
    objective_span = (
        observed_tensor.max(dim=0).values - observed_tensor.min(dim=0).values
    ).clamp_min(1e-6)
    reference_point = (
        observed_tensor.min(dim=0).values - 0.01 * objective_span
    ).detach()
    partitioning = NondominatedPartitioning(
        ref_point=reference_point,
        Y=observed_tensor,
    )
    acquisition = ExpectedHypervolumeImprovement(
        model=model_list,
        ref_point=reference_point.tolist(),
        partitioning=partitioning,
    )

    log(
        f"Scoring candidates with analytic EHVI; reference point="
        f"({reference_point[0]:.3f}, {reference_point[1]:.3f})."
    )
    candidate_mean, candidate_std, ehvi = score_candidates(
        gp_uts,
        gp_ec,
        acquisition,
        x_candidates,
        config.prediction_batch_size,
    )
    # EHVI is theoretically non-negative. Remove negligible negative values
    # caused by floating-point round-off before ranking and export.
    ehvi = np.maximum(ehvi, 0.0)

    union_objectives = np.vstack([paired_observed, candidate_mean])
    union_mask = non_dominated_mask_2d(union_objectives)
    candidate_front_mask = union_mask[len(paired_observed) :]

    # Rank the complete discrete candidate space by pointwise analytic EHVI.
    # The posterior-mean Pareto mask is retained only as a diagnostic output;
    # it does not restrict acquisition because EHVI already accounts for both
    # improvement magnitude and posterior uncertainty.
    ranked = np.argsort(-ehvi, kind="stable")
    selected_indices = ranked[: min(config.top_k, len(ranked))].astype(int).tolist()
    selected_ehvi = ehvi[selected_indices]
    log(
        f"Posterior-mean non-dominated candidates (diagnostic only): "
        f"{int(candidate_front_mask.sum()):,}; full-space EHVI selections: "
        f"{len(selected_indices)}."
    )

    baseline_hv = float(partitioning.compute_hypervolume())
    selected_means = candidate_mean[selected_indices]
    augmented = torch.as_tensor(np.vstack([paired_observed, selected_means]))
    augmented_hv = float(
        NondominatedPartitioning(
            ref_point=reference_point,
            Y=augmented,
        ).compute_hypervolume()
    )
    delta_hv_mean = augmented_hv - baseline_hv

    candidate_results = candidate_raw.reset_index(drop=True).copy()
    candidate_results.insert(0, "candidate_index", np.arange(len(candidate_results)))
    candidate_results["UTS"] = candidate_mean[:, 0]
    candidate_results["UTS_error"] = candidate_std[:, 0]
    candidate_results["EC"] = candidate_mean[:, 1]
    candidate_results["EC_error"] = candidate_std[:, 1]
    candidate_results["EHVI_value"] = ehvi
    candidate_results["is_mean_nondominated_wrt_union"] = (
        candidate_front_mask.astype(int)
    )
    ehvi_rank = np.empty(len(ranked), dtype=int)
    ehvi_rank[ranked] = np.arange(1, len(ranked) + 1)
    candidate_results["EHVI_rank"] = ehvi_rank
    candidate_results.sort_values("EHVI_rank").to_csv(
        config.output_dir / "all_candidates_ehvi.csv", index=False
    )

    selected_results = candidate_results.iloc[selected_indices].copy()
    selected_results.insert(1, "rank", np.arange(1, len(selected_results) + 1))
    selected_results.to_csv(config.output_dir / "selected_candidates.csv", index=False)

    sensitivities = {
        "UTS": ard_sensitivity(gp_uts, feature_names),
        "EC": ard_sensitivity(gp_ec, feature_names),
    }
    save_feature_sensitivity(sensitivities, feature_names, config.output_dir)
    plot_pareto_results(
        paired_observed,
        candidate_mean,
        candidate_std,
        selected_indices,
        candidate_features,
        references,
        config.output_dir,
        config,
    )

    kernel_parameters = {
        "UTS": kernel_diagnostics(gp_uts),
        "EC": kernel_diagnostics(gp_ec),
    }

    if config.save_model_state:
        torch.save(
            {
                "feature_names": feature_names,
                "normalization_bounds": normalization_bounds,
                "UTS_state_dict": gp_uts.state_dict(),
                "EC_state_dict": gp_ec.state_dict(),
                "UTS_train_X": x_uts,
                "UTS_train_Y": y_uts,
                "UTS_train_Yvar": yvar_uts,
                "EC_train_X": x_ec,
                "EC_train_Y": y_ec,
                "EC_train_Yvar": yvar_ec,
            },
            config.output_dir / "gp_model_state.pt",
        )

    summary = {
        "configuration": serializable_config(config),
        "input_files": {
            "paired_training": {
                "path": str(config.train_csv),
                "rows": len(paired_raw),
                "sha256": file_sha256(config.train_csv),
            },
            "candidates": {
                "path": str(config.candidate_csv),
                "rows": len(candidate_raw),
                "sha256": file_sha256(config.candidate_csv),
            },
            "UTS_reference": (
                {
                    "path": str(config.uts_reference_csv),
                    "sha256": file_sha256(config.uts_reference_csv),
                }
                if config.use_uts_reference
                else None
            ),
            "EC_reference": (
                {
                    "path": str(config.ec_reference_csv),
                    "sha256": file_sha256(config.ec_reference_csv),
                }
                if config.use_ec_reference
                else None
            ),
        },
        "training_rows": {
            "total": len(training),
            "UTS": len(x_uts),
            "EC": len(x_ec),
            "paired": int(paired_mask.sum()),
        },
        "candidate_rows": len(candidate_raw),
        "feature_names": feature_names,
        "normalization_bounds": {
            feature: {
                "minimum": float(normalization_bounds[0, index]),
                "maximum": float(normalization_bounds[1, index]),
            }
            for index, feature in enumerate(feature_names)
        },
        "selected_indices": selected_indices,
        "selected_ehvi": selected_ehvi.astype(float).tolist(),
        "reference_point": reference_point.cpu().numpy().astype(float).tolist(),
        "delta_hv_mean_assumption": float(delta_hv_mean),
        "acquisition": (
            "analytic ExpectedHypervolumeImprovement (q=1), "
            "ranked over the complete candidate space"
        ),
        "candidate_selection": "full-space pointwise EHVI ranking",
        "posterior_mean_front_count_diagnostic": int(candidate_front_mask.sum()),
        "used_qEHVI": False,
        "model_performance_in_sample": model_performance,
        "calibration_in_sample": calibration,
        "kernel_parameters": kernel_parameters,
        "ard_sensitivity": sensitivities,
        "notes": (
            "Two independent fixed-noise GPs; CV/KNN heteroscedastic noise; "
            "Matern-5/2 plus RBF kernels; full-space pointwise EHVI ranking."
        ),
    }
    with (config.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    log(f"Outputs saved to: {config.output_dir}")
    log(f"Mean-posterior hypervolume increase: {delta_hv_mean:.6f}")
    print("\nSelected candidates:")
    print(selected_results.to_string(index=False))


if __name__ == "__main__":
    main()
