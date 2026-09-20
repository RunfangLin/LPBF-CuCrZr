"""Desktop predictor for the final LPBF CuCrZr relative-density model."""

from __future__ import annotations

import json
import math
import tkinter as tk
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Optional

import joblib
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "Density_Results"
MODEL_PATH = RESULT_DIR / "density_model.pkl"
SUMMARY_PATH = RESULT_DIR / "results_summary.json"
LOW_DENSITY_WARNING = "Warning: model predictions below 97% are less reliable."

DEFAULT_DOMAIN = {
    "power": {"minimum": 300.0, "maximum": 370.0, "unit": "W"},
    "speed": {"minimum": 300.0, "maximum": 500.0, "unit": "mm/s"},
    "hd": {"minimum": 0.09, "maximum": 0.17, "unit": "mm"},
    "lt": {"allowed": [20.0, 30.0], "unit": "um"},
}
ALLOWED_FEATURE_SETS = {
    ("power", "speed", "lt"),
    ("power", "speed", "hd", "lt"),
}


def installed_version(distribution: str) -> Optional[str]:
    """Return the installed package version, or None when it is unavailable."""
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return None


def validate_runtime_compatibility(summary: dict[str, Any]) -> dict[str, str]:
    """Require the same model-library versions used during training.

    The fitted estimator is stored with joblib. XGBoost does not guarantee that
    pickled estimators will produce valid predictions across library versions.
    Refusing an incompatible model prevents plausible-looking but incorrect
    density values.
    """
    recorded = summary.get("software", {})
    packages = {
        "xgboost": "xgboost",
        "scikit-learn": "scikit-learn",
    }
    runtime: dict[str, str] = {}
    problems: list[str] = []

    for summary_name, distribution in packages.items():
        expected = str(recorded.get(summary_name, "")).strip()
        current = installed_version(distribution)
        runtime[summary_name] = current or "not installed"

        if not expected:
            problems.append(
                f"The training summary does not record {summary_name}."
            )
        elif current is None:
            problems.append(f"{summary_name} {expected} is required but not installed.")
        elif current != expected:
            problems.append(
                f"{summary_name}: model trained with {expected}, "
                f"application is running {current}."
            )

    if problems:
        details = "\n".join(f"- {problem}" for problem in problems)
        raise RuntimeError(
            "Incompatible model environment:\n"
            f"{details}\n\n"
            "Run Density_Model.py and Density_Predictor.py with the same "
            "Python environment, then restart the application."
        )

    return runtime


def load_artifacts(
) -> tuple[Any, dict[str, Any], list[str], dict[str, Any], dict[str, str]]:
    """Load the fitted model and its recorded modelling metadata."""
    missing = [path for path in (MODEL_PATH, SUMMARY_PATH) if not path.is_file()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"Missing {names}. Run Density_Model.py before using the predictor."
        )

    with SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    selected_features = list(summary.get("selected_features", []))
    if tuple(selected_features) not in ALLOWED_FEATURE_SETS:
        raise ValueError(
            f"Unsupported feature set in results_summary.json: {selected_features}"
        )

    domain = summary.get("application_domain", DEFAULT_DOMAIN)
    runtime_versions = validate_runtime_compatibility(summary)
    model = joblib.load(MODEL_PATH)

    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None and list(model_features) != selected_features:
        raise RuntimeError(
            "The model feature order does not match results_summary.json. "
            "Run Density_Model.py again before using the predictor."
        )

    return model, summary, selected_features, domain, runtime_versions


def validate_parameters(
    parameters: dict[str, float],
    domain: dict[str, Any],
) -> None:
    """Reject inputs outside the approved engineering application domain."""
    labels = {
        "power": "Power",
        "speed": "Scan speed",
        "hd": "Hatch spacing",
    }
    for name in ("power", "speed", "hd"):
        lower = float(domain[name]["minimum"])
        upper = float(domain[name]["maximum"])
        value = parameters[name]
        if not lower <= value <= upper:
            unit = domain[name]["unit"]
            raise ValueError(
                f"{labels[name]} must be between {lower:g} and {upper:g} {unit}."
            )

    allowed_layers = [float(value) for value in domain["lt"]["allowed"]]
    if parameters["lt"] not in allowed_layers:
        allowed_text = " or ".join(f"{value:g}" for value in allowed_layers)
        raise ValueError(f"Layer thickness must be {allowed_text} um.")


def predict_density(
    model: Any,
    selected_features: list[str],
    parameters: dict[str, float],
) -> float:
    """Predict relative density using the selected three- or four-input model."""
    model_input = pd.DataFrame(
        [{name: parameters[name] for name in selected_features}],
        columns=selected_features,
    )
    prediction = float(model.predict(model_input)[0])
    if not math.isfinite(prediction):
        raise RuntimeError("The model returned a non-finite density prediction.")
    if not 0.0 <= prediction <= 100.0:
        raise RuntimeError(
            f"The model returned an invalid relative density ({prediction:.3f}%). "
            "Check that training and prediction use the same software environment."
        )
    return prediction


class DensityPredictorApp(ttk.Frame):
    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master, padding=18)
        self.master = master
        self.model = None
        self.summary: dict[str, Any] = {}
        self.selected_features: list[str] = []
        self.domain: dict[str, Any] = DEFAULT_DOMAIN
        self.runtime_versions: dict[str, str] = {}
        self.load_error = ""

        self.power = tk.StringVar(value="350")
        self.speed = tk.StringVar(value="400")
        self.hatch_spacing = tk.StringVar(value="0.11")
        self.layer_thickness = tk.StringVar(value="20")
        self.result = tk.StringVar(value="Predicted relative density: --")
        self.warning = tk.StringVar(value="")
        self.model_status = tk.StringVar(value="Loading model...")

        self._build_interface()
        self._load_model()

    def _build_interface(self) -> None:
        self.master.title("LPBF CuCrZr Density Predictor")
        self.master.minsize(610, 430)
        self.grid(sticky="nsew")
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        ttk.Label(
            self,
            text="LPBF CuCrZr Relative-Density Predictor",
            font=("Segoe UI", 16, "bold"),
        ).grid(row=0, column=0, columnspan=3, pady=(0, 6))
        ttk.Label(
            self,
            text="Inputs are restricted to the approved application domain.",
        ).grid(row=1, column=0, columnspan=3, pady=(0, 16))

        fields = (
            ("Laser power", self.power, "W (300-370)"),
            ("Scan speed", self.speed, "mm/s (300-500)"),
            ("Hatch spacing", self.hatch_spacing, "mm (0.09-0.17)"),
        )
        for row, (label, variable, unit) in enumerate(fields, start=2):
            ttk.Label(self, text=label).grid(row=row, column=0, sticky="w", pady=6)
            ttk.Entry(self, textvariable=variable, width=22).grid(
                row=row, column=1, sticky="ew", padx=10, pady=6
            )
            ttk.Label(self, text=unit).grid(row=row, column=2, sticky="w", pady=6)

        ttk.Label(self, text="Layer thickness").grid(
            row=5, column=0, sticky="w", pady=6
        )
        ttk.Combobox(
            self,
            textvariable=self.layer_thickness,
            values=("20", "30"),
            state="readonly",
            width=19,
        ).grid(row=5, column=1, sticky="ew", padx=10, pady=6)
        ttk.Label(self, text="um").grid(row=5, column=2, sticky="w", pady=6)

        button_frame = ttk.Frame(self)
        button_frame.grid(row=6, column=0, columnspan=3, pady=(18, 12))
        ttk.Button(button_frame, text="Predict density", command=self._predict).grid(
            row=0, column=0, padx=6
        )
        ttk.Button(button_frame, text="Reload model", command=self._load_model).grid(
            row=0, column=1, padx=6
        )

        ttk.Separator(self).grid(
            row=7, column=0, columnspan=3, sticky="ew", pady=8
        )
        ttk.Label(
            self,
            textvariable=self.result,
            font=("Segoe UI", 14, "bold"),
        ).grid(row=8, column=0, columnspan=3, pady=(8, 4))
        tk.Label(
            self,
            textvariable=self.warning,
            fg="#b22222",
            font=("Microsoft YaHei UI", 11, "bold"),
        ).grid(row=9, column=0, columnspan=3, pady=4)
        ttk.Label(
            self,
            textvariable=self.model_status,
            wraplength=560,
            justify="center",
        ).grid(row=10, column=0, columnspan=3, pady=(12, 0))

    def _load_model(self) -> None:
        try:
            (
                self.model,
                self.summary,
                self.selected_features,
                self.domain,
                self.runtime_versions,
            ) = load_artifacts()
            selected_model = self.summary.get("selected_model", "unknown")
            feature_text = ", ".join(self.selected_features)
            xgboost_version = self.runtime_versions.get("xgboost", "unknown")
            self.model_status.set(
                f"Loaded: {selected_model} | Inputs: {feature_text} | "
                f"XGBoost {xgboost_version}"
            )
            self.load_error = ""
        except Exception as error:
            self.model = None
            self.load_error = str(error)
            self.model_status.set(self.load_error)
            self.result.set("Predicted relative density: --")
            self.warning.set("")

    def _read_parameters(self) -> dict[str, float]:
        try:
            parameters = {
                "power": float(self.power.get()),
                "speed": float(self.speed.get()),
                "hd": float(self.hatch_spacing.get()),
                "lt": float(self.layer_thickness.get()),
            }
        except ValueError as error:
            raise ValueError("All process parameters must be numeric.") from error
        validate_parameters(parameters, self.domain)
        return parameters

    def _predict(self) -> None:
        try:
            if self.model is None:
                self._load_model()
            if self.model is None:
                raise RuntimeError(
                    self.load_error
                    or "The fitted model is unavailable. Run Density_Model.py first."
                )
            parameters = self._read_parameters()
            prediction = predict_density(
                self.model,
                self.selected_features,
                parameters,
            )
        except Exception as error:
            messagebox.showerror("Prediction error", str(error))
            return

        self.result.set(f"Predicted relative density: {prediction:.2f}%")
        self.warning.set(LOW_DENSITY_WARNING if prediction < 97.0 else "")


def main() -> None:
    root = tk.Tk()
    DensityPredictorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
