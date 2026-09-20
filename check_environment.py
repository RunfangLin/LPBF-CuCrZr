# check_environment.py
# Check the Python environment for LPBF CuCrZr Density + Pareto models.

from __future__ import annotations

import ast
import importlib
import importlib.metadata
import os
import sys
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Core packages used by the current Density / Pareto workflows.
PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "joblib": "joblib",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "torch": "torch",
    "botorch": "botorch",
    "gpytorch": "gpytorch",

    # Useful dependencies / legacy scripts.
    "scipy": "scipy",
    "tqdm": "tqdm",
}


# Specific APIs currently used by Pareto_Model.py.
API_CHECKS = [
    ("botorch", "fit_gpytorch_mll"),

    ("botorch.models.gp_regression", "FixedNoiseGP"),
    ("botorch.models.model_list_gp_regression", "ModelListGP"),

    ("botorch.models.transforms.input", "Normalize"),
    ("botorch.models.transforms.outcome", "Standardize"),

    (
        "botorch.acquisition.multi_objective.analytic",
        "ExpectedHypervolumeImprovement",
    ),
    (
        "botorch.utils.multi_objective.box_decompositions.non_dominated",
        "NondominatedPartitioning",
    ),

    ("gpytorch.kernels", "MaternKernel"),
    ("gpytorch.kernels", "RBFKernel"),
    ("gpytorch.kernels", "ScaleKernel"),

    ("gpytorch.priors", "GammaPrior"),

    (
        "gpytorch.mlls.sum_marginal_log_likelihood",
        "SumMarginalLogLikelihood",
    ),
    (
        "gpytorch.mlls.exact_marginal_log_likelihood",
        "ExactMarginalLogLikelihood",
    ),

    # Density model APIs.
    ("sklearn.model_selection", "KFold"),
    ("sklearn.model_selection", "train_test_split"),
    ("sklearn.metrics", "r2_score"),
    ("sklearn.metrics", "mean_squared_error"),
    ("sklearn.metrics", "mean_absolute_error"),
    ("xgboost", "XGBRegressor"),
]


EXCLUDED_DIRS = {
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    "site-packages",
}


# ============================================================
# Helpers
# ============================================================

def get_version(import_name: str, pip_name: str) -> str:
    """Return an installed package version if available."""
    try:
        module = importlib.import_module(import_name)

        version = getattr(module, "__version__", None)
        if version:
            return str(version)

        return importlib.metadata.version(pip_name)

    except Exception:
        try:
            return importlib.metadata.version(pip_name)
        except Exception:
            return "unknown"


def check_packages():
    """Check that all required top-level packages can be imported."""
    print("\n" + "=" * 72)
    print("1. PACKAGE CHECK")
    print("=" * 72)

    missing = []

    for import_name, pip_name in PACKAGES.items():
        try:
            importlib.import_module(import_name)
            version = get_version(import_name, pip_name)

            print(
                f"[ OK ] {import_name:<15} "
                f"version = {version}"
            )

        except Exception as exc:
            print(
                f"[FAIL] {import_name:<15} "
                f"({pip_name}) -> {type(exc).__name__}: {exc}"
            )
            missing.append(pip_name)

    return sorted(set(missing))


def check_specific_apis():
    """
    Check APIs used explicitly by Density_Model.py and Pareto_Model.py.

    This catches package-version incompatibilities even when the package
    itself is installed.
    """
    print("\n" + "=" * 72)
    print("2. API COMPATIBILITY CHECK")
    print("=" * 72)

    failures = []

    for module_name, symbol_name in API_CHECKS:
        try:
            module = importlib.import_module(module_name)
            getattr(module, symbol_name)

            print(
                f"[ OK ] {module_name}.{symbol_name}"
            )

        except Exception as exc:
            print(
                f"[FAIL] {module_name}.{symbol_name}"
            )
            print(
                f"       {type(exc).__name__}: {exc}"
            )

            failures.append(
                f"{module_name}.{symbol_name}"
            )

    return failures


def check_torch():
    """Print PyTorch and CUDA information."""
    print("\n" + "=" * 72)
    print("3. PYTORCH / CUDA CHECK")
    print("=" * 72)

    try:
        import torch

        print(f"PyTorch version : {torch.__version__}")
        print(f"CUDA available  : {torch.cuda.is_available()}")
        print(f"Torch CUDA build: {torch.version.cuda}")

        if torch.cuda.is_available():
            print(f"GPU count       : {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                print(
                    f"GPU {i:<2}          : "
                    f"{torch.cuda.get_device_name(i)}"
                )

            # Tiny GPU computation test.
            try:
                x = torch.tensor(
                    [1.0, 2.0, 3.0],
                    dtype=torch.double,
                    device="cuda",
                )
                y = x ** 2

                print(
                    f"CUDA tensor test : PASS "
                    f"({y.cpu().tolist()})"
                )

            except Exception as exc:
                print(
                    f"CUDA tensor test : FAIL -> {exc}"
                )

        else:
            print(
                "CUDA tensor test : skipped "
                "(CPU execution is still valid)"
            )

    except Exception as exc:
        print(
            f"[FAIL] PyTorch cannot be imported: "
            f"{type(exc).__name__}: {exc}"
        )


def collect_python_imports(root: Path):
    """
    Scan project .py files and collect imported top-level module names.

    The code itself is NOT executed.
    """
    imports = set()
    syntax_errors = []

    for file_path in root.rglob("*.py"):

        # Ignore virtual environments and irrelevant folders.
        if any(
            part.lower() in {x.lower() for x in EXCLUDED_DIRS}
            for part in file_path.parts
        ):
            continue

        try:
            source = file_path.read_text(
                encoding="utf-8-sig",
                errors="replace",
            )

            tree = ast.parse(source)

        except SyntaxError as exc:
            syntax_errors.append(
                (file_path, str(exc))
            )
            continue

        except Exception:
            continue

        for node in ast.walk(tree):

            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(
                        alias.name.split(".")[0]
                    )

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(
                        node.module.split(".")[0]
                    )

    return imports, syntax_errors


def check_project_imports():
    """
    Automatically scan every Python script in the project directory.

    This can catch dependencies not included in the hard-coded list.
    """
    print("\n" + "=" * 72)
    print("4. PROJECT-WIDE IMPORT SCAN")
    print("=" * 72)

    imports, syntax_errors = collect_python_imports(
        PROJECT_ROOT
    )

    # Local Python modules should not be treated as external packages.
    local_modules = {
        p.stem
        for p in PROJECT_ROOT.rglob("*.py")
        if not any(
            part.lower() in {x.lower() for x in EXCLUDED_DIRS}
            for part in p.parts
        )
    }

    stdlib = set(
        getattr(sys, "stdlib_module_names", set())
    )

    external = sorted(
        name
        for name in imports
        if name not in stdlib
        and name not in local_modules
        and name != "__future__"
    )

    print("Detected external imports:")

    for name in external:
        try:
            importlib.import_module(name)
            print(f"[ OK ] {name}")

        except Exception as exc:
            print(
                f"[FAIL] {name:<20} "
                f"{type(exc).__name__}: {exc}"
            )

    if syntax_errors:
        print("\nPython files with syntax errors:")

        for path, error in syntax_errors:
            try:
                rel = path.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = path

            print(f"[WARN] {rel}")
            print(f"       {error}")

    return external


def print_python_environment():
    """Show exactly which Python interpreter is being used."""
    print("=" * 72)
    print("LPBF CuCrZr ENVIRONMENT CHECK")
    print("=" * 72)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python       : {sys.version}")
    print(f"Executable   : {sys.executable}")
    print(f"Prefix       : {sys.prefix}")

    virtual_env = os.environ.get("VIRTUAL_ENV")

    if virtual_env:
        print(f"Virtual env  : {virtual_env}")
    else:
        print(
            "Virtual env  : NOT detected from VIRTUAL_ENV"
        )

    expected_venv = PROJECT_ROOT / ".venv"

    if expected_venv.exists():
        print(f".venv folder : FOUND ({expected_venv})")
    else:
        print(".venv folder : not found")


def main():
    print_python_environment()

    missing_packages = check_packages()
    api_failures = check_specific_apis()
    check_torch()
    check_project_imports()

    print("\n" + "=" * 72)
    print("5. SUMMARY")
    print("=" * 72)

    if not missing_packages and not api_failures:
        print(
            "[PASS] Core Density + Pareto environment "
            "appears ready."
        )

    else:
        if missing_packages:
            print("\nMissing packages:")
            for package in missing_packages:
                print(f"  - {package}")

            print("\nInstall missing packages with:")
            print(
                "python -m pip install "
                + " ".join(missing_packages)
            )

        if api_failures:
            print("\nPackage/API compatibility problems:")
            for failure in api_failures:
                print(f"  - {failure}")

            print(
                "\nNOTE: An API failure means the package may be installed, "
                "but its version may not match the current code."
            )

    print("\nEnvironment check completed.")


if __name__ == "__main__":
    main()