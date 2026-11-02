"""Indian Car Market Analysis and Price Prediction dashboard.

Run from this folder:
    streamlit run app.py

The dataset is deliberately read from an external folder.  It is never copied
into this project, so large source data stays out of version control.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Requirement reference: 4856969-Machine_Learning_Major_Project.pdf
# (Data Science Major Project / Expected Deliverables).  This app provides
# data cleaning, exploratory analysis, a trained regression model and metrics.
TRAINING_FILE = "Cap_Training_Data_2025.csv"
DATASET_FOLDER = "12573233-Indian_Car_Market_Capstone_2025_Dataset"
DATASET_PATH_ENV = "CAR_DATASET_PATH"
DATASET_DIR_ENV = "CAR_DATASET_DIR"
APP_ROOT: Final = Path(__file__).resolve().parent
RANDOM_STATE = 42
TARGET = "price"


def normalise_location(value: str | Path) -> Path:
    """Expand a user supplied path without relying on the current working directory."""
    text = os.path.expandvars(str(value).strip().strip('"').strip("'"))
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = APP_ROOT / path
    return path.resolve(strict=False)


def training_file_in(location: str | Path) -> Path | None:
    """Return the expected CSV from either a file path or an external directory."""
    try:
        path = normalise_location(location)
        if path.is_file():
            return path
        if not path.is_dir():
            return None

        exact_file = path / TRAINING_FILE
        if exact_file.is_file():
            return exact_file.resolve()

        # Support a case-insensitive match without recursively searching user files.
        for csv_file in path.glob("*.csv"):
            if csv_file.name.casefold() == TRAINING_FILE.casefold() and csv_file.is_file():
                return csv_file.resolve()
    except OSError:
        # An inaccessible drive/folder is simply not a usable data source.
        return None
    return None


def unique_candidates(candidates: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    """Keep candidate order while removing equivalent Windows/macOS/Linux paths."""
    unique: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for path, source in candidates:
        resolved = path.resolve(strict=False)
        key = str(resolved).casefold()
        if key not in seen:
            seen.add(key)
            unique.append((resolved, source))
    return unique


def discover_dataset_candidates() -> list[tuple[Path, str]]:
    """Find the named training CSV in safe, likely external locations only.

    Users can override discovery with CAR_DATASET_PATH (an exact CSV path) or
    CAR_DATASET_DIR (a folder or CSV path). No data is copied into this project.
    """
    explicit_locations = [
        (os.getenv(DATASET_PATH_ENV), DATASET_PATH_ENV),
        (os.getenv(DATASET_DIR_ENV), DATASET_DIR_ENV),
    ]
    explicit_candidates = [
        (found, f"Environment variable {name}")
        for value, name in explicit_locations
        if value and (found := training_file_in(value)) is not None
    ]
    if explicit_candidates:
        return unique_candidates(explicit_candidates)

    download_roots = [Path.home() / "Downloads"]
    for environment_name in ("USERPROFILE", "OneDrive", "OneDriveConsumer"):
        if environment_value := os.getenv(environment_name):
            download_roots.append(normalise_location(environment_value) / "Downloads")

    automatic_locations: list[Path] = [
        APP_ROOT.parent / DATASET_FOLDER,
        APP_ROOT / DATASET_FOLDER,
    ]
    for download_root in download_roots:
        automatic_locations.extend((download_root / DATASET_FOLDER, download_root))

    found_candidates = [
        (found, "Automatic discovery")
        for location in automatic_locations
        if (found := training_file_in(location)) is not None
    ]
    return unique_candidates(found_candidates)


def dataset_fingerprint(path: Path) -> tuple[str, int, int]:
    """Change cache keys when an external CSV is replaced or edited."""
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


@st.cache_data(max_entries=4, show_spinner="Loading and cleaning the training data...")
def load_data(path_as_text: str, fingerprint: tuple[str, int, int]) -> pd.DataFrame:
    """Load once per file path and normalize columns without changing the source CSV."""
    del fingerprint  # It is intentionally part of the cache key; see dataset_fingerprint().
    path = Path(path_as_text)
    if not path.is_file():
        raise FileNotFoundError(f"Training CSV was not found: {path}")

    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    if TARGET not in frame:
        raise ValueError(f"The CSV needs a '{TARGET}' column. Found: {', '.join(frame.columns)}")

    # Convert values that should be numeric. Invalid entries become missing and
    # are imputed by the model pipeline, avoiding row loss and data leakage.
    for column in ["price", "manufacture_year", "age_of_car", "engine_displacement",
                   "engine_power", "vroom_audit_rating", "door_count", "seat_count"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "manufacture_year" in frame:
        frame["vehicle_age"] = (pd.Timestamp.now().year - frame["manufacture_year"]).clip(0, 60)
    frame = frame[frame[TARGET].gt(0)].copy()
    if frame.empty:
        raise ValueError("No valid positive prices are available after cleaning.")
    return frame


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Choose only useful, available predictors and exclude record identifiers."""
    excluded = {TARGET, "id", "model"}  # model has high cardinality and overfits a small data set.
    candidates = [column for column in frame.columns if column not in excluded]
    numeric = [column for column in candidates if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in candidates if column not in numeric]
    return numeric, categorical


@st.cache_resource(max_entries=4, show_spinner="Training and selecting the price model...")
def train_best_model(
    path_as_text: str, fingerprint: tuple[str, int, int]
) -> tuple[Pipeline, dict[str, float]]:
    """Automatically select the stronger of a fast baseline and a nonlinear model."""
    frame = load_data(path_as_text, fingerprint)
    if len(frame) < 10:
        raise ValueError("At least 10 valid-priced cars are required to train a model.")
    numeric, categorical = feature_columns(frame)
    if not numeric and not categorical:
        raise ValueError("No usable feature columns were found for price prediction.")
    features, target = frame[numeric + categorical], frame[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ],
        remainder="drop",
    )
    candidates = {
        "Ridge baseline": Ridge(alpha=10.0),
        "Random forest": RandomForestRegressor(
            # A bounded worker count keeps the dashboard responsive on laptops
            # and avoids process-start overhead during the built-in self-test.
            n_estimators=120, min_samples_leaf=2, max_features=0.8,
            random_state=RANDOM_STATE, n_jobs=1,
        ),
    }
    scored: list[tuple[float, str, Pipeline, np.ndarray]] = []
    for name, estimator in candidates.items():
        model = Pipeline([("prepare", preprocessing), ("model", estimator)])
        model.fit(x_train, y_train)
        # Pipeline.predict is typed as possibly multi-output by scikit-learn.
        # This dashboard has one target (price), so normalize to a 1-D array.
        prediction = np.asarray(model.predict(x_test), dtype=float).reshape(-1)
        scored.append((root_mean_squared_error(y_test, prediction), name, model, prediction))
    rmse, selected_name, selected_model, prediction = min(scored, key=lambda item: item[0])
    metrics = {
        "model": selected_name,
        "mae": mean_absolute_error(y_test, prediction),
        "rmse": rmse,
        "r2": r2_score(y_test, prediction),
    }
    return selected_model, metrics


def main() -> None:
    st.set_page_config(page_title="Indian Car Market", layout="wide")
    st.title("Indian Car Market Analysis & Price Prediction")
    st.caption("External dataset | cached analysis | automatic model selection")

    with st.sidebar:
        st.header("Data source")
        manual_location = st.text_input(
            "Training CSV or dataset folder (optional)",
            help=(
                "Paste the full CSV path or the folder that contains "
                f"{TRAINING_FILE}. The data remains outside this project."
            ),
        )
        if manual_location.strip():
            manual_file = training_file_in(manual_location)
            candidates = [] if manual_file is None else [(manual_file, "Manual path")]
        else:
            candidates = discover_dataset_candidates()

        if candidates:
            if len(candidates) == 1:
                path, source = candidates[0]
            else:
                labels = [f"{source}: {path}" for path, source in candidates]
                selected_label = st.selectbox("Multiple datasets found", labels)
                path, source = candidates[labels.index(selected_label)]
            st.success(f"Using {source.lower()}")
            st.caption(str(path))
        else:
            path = None
            st.warning("Training CSV not found")

    if path is None:
        st.error(f"Could not find {TRAINING_FILE}.")
        st.info(
            "Set CAR_DATASET_PATH to the CSV, set CAR_DATASET_DIR to its folder, "
            "or paste either location into the sidebar."
        )
        st.code(
            '$env:CAR_DATASET_PATH = "C:\\path\\to\\Cap_Training_Data_2025.csv"',
            language="powershell",
        )
        return

    try:
        fingerprint = dataset_fingerprint(path)
        frame = load_data(str(path), fingerprint)
    except (OSError, UnicodeDecodeError, ValueError, pd.errors.ParserError) as error:
        st.error(str(error))
        st.info("Choose a readable CSV with a Price column, then try again.")
        return

    page = st.sidebar.radio("View", ["Overview", "Market analysis", "Price prediction", "Data quality"])

    if page == "Overview":
        columns = st.columns(4)
        columns[0].metric("Cars", f"{len(frame):,}")
        columns[1].metric("Average price", f"INR {frame[TARGET].mean():,.0f}")
        columns[2].metric("Median price", f"INR {frame[TARGET].median():,.0f}")
        columns[3].metric("Manufacturers", frame.get("maker", pd.Series(dtype=str)).nunique())
        if "maker" in frame:
            maker_counts = frame["maker"].value_counts().head(12).rename_axis("maker").reset_index(name="cars")
            st.plotly_chart(
                px.bar(maker_counts, x="maker", y="cars", title="Most represented makers"),
                width="stretch",
            )
        st.plotly_chart(
            px.histogram(frame, x=TARGET, nbins=40, title="Price distribution"), width="stretch"
        )

    elif page == "Market analysis":
        group_options = [column for column in ["maker", "location", "fuel_type", "transmission", "body_type"] if column in frame]
        if not group_options:
            st.warning("This CSV has no supported market-analysis columns.")
            return
        group = st.selectbox("Compare prices by", group_options)
        summary = (frame.groupby(group, dropna=False)[TARGET].agg(["count", "mean", "median"])
                   .sort_values("mean", ascending=False).head(20).reset_index())
        st.plotly_chart(
            px.bar(summary, x=group, y="mean", hover_data=["count", "median"], title="Average price"),
            width="stretch",
        )
        st.dataframe(summary, width="stretch", hide_index=True)

    elif page == "Price prediction":
        if st.button("Train / refresh model", type="primary"):
            # Explicit refresh clears both levels, even when the file timestamp is unchanged.
            load_data.clear()
            train_best_model.clear()
            st.rerun()
        try:
            _model, metrics = train_best_model(str(path), fingerprint)
        except ValueError as error:
            st.error(str(error))
            return
        first, second, third = st.columns(3)
        first.metric("Selected model", metrics["model"])
        second.metric("MAE", f"INR {metrics['mae']:,.0f}")
        third.metric("R2", f"{metrics['r2']:.3f}")
        st.caption(
            f"Test RMSE: INR {metrics['rmse']:,.0f}. "
            "The lower-error candidate is selected automatically."
        )
        st.info("The trained model is cached for the current data path, avoiding unnecessary retraining on each interaction.")

    else:
        missing = frame.isna().sum().sort_values(ascending=False).rename("missing").to_frame()
        st.subheader("Data quality")
        st.write(f"Rows: {len(frame):,}  |  Columns: {len(frame.columns)}  |  Duplicate rows: {frame.duplicated().sum():,}")
        missing_values = missing[missing["missing"] > 0]
        if missing_values.empty:
            st.success("No missing values after cleaning.")
        else:
            st.dataframe(missing_values, width="stretch")
        st.dataframe(frame.head(100), width="stretch", hide_index=True)


def run_self_test() -> None:
    """Run fast checks without Streamlit UI or the real dataset.

    Usage: python app.py --self-test
    The temporary CSV is created outside the project and removed automatically.
    """
    with tempfile.TemporaryDirectory() as temporary_dir:
        path = Path(temporary_dir) / TRAINING_FILE
        rows = 40
        pd.DataFrame(
            {
                "ID": range(1, rows + 1),
                "Maker": ["maker_a", "maker_b"] * (rows // 2),
                "model": ["model_x", "model_y"] * (rows // 2),
                "Location": ["Delhi", "Mumbai"] * (rows // 2),
                "manufacture_year": [2015, 2018] * (rows // 2),
                "engine_displacement": [1200, 1500] * (rows // 2),
                "fuel_type": ["petrol", "diesel"] * (rows // 2),
                "transmission": ["manual", "automatic"] * (rows // 2),
                "Price": [400_000 + index * 10_000 for index in range(rows)],
            }
        ).to_csv(path, index=False)

        load_data.clear()
        train_best_model.clear()
        fingerprint = dataset_fingerprint(path)
        frame = load_data(str(path), fingerprint)
        numeric, categorical = feature_columns(frame)
        model, metrics = train_best_model(str(path), fingerprint)

        assert "price" in frame and "vehicle_age" in frame
        assert not {"id", "model", "price"}.intersection(numeric + categorical)
        assert model is not None and metrics["rmse"] >= 0
        assert training_file_in(path.parent) == path
        try:
            missing_path = path.with_name("missing.csv")
            load_data(str(missing_path), (str(missing_path), 0, 0))
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("A missing training file must raise FileNotFoundError.")
    print("Self-test passed: loading, cleaning, feature selection, error handling, and model training.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        run_self_test()
    else:
        main()
