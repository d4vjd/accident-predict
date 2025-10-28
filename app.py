import os
import io
import json
import time
import pickle
from typing import Dict, List, Optional, Tuple
import importlib


def _safe_import_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


# Dynamically import optional dependencies to avoid static import diagnostics
_np_mod = _safe_import_module("numpy")
np = _np_mod

_pd_mod = _safe_import_module("pandas")
pd = _pd_mod

_st_mod = _safe_import_module("streamlit")
st = _st_mod

_sk_pre = _safe_import_module("sklearn.preprocessing")
LabelEncoder = getattr(_sk_pre, "LabelEncoder", None) if _sk_pre else None

_sk_ms = _safe_import_module("sklearn.model_selection")
KFold = getattr(_sk_ms, "KFold", None) if _sk_ms else None

_sk_metrics = _safe_import_module("sklearn.metrics")
mean_squared_error = getattr(_sk_metrics, "mean_squared_error", None) if _sk_metrics else None

_xgb_mod = _safe_import_module("xgboost")
XGBRegressor = getattr(_xgb_mod, "XGBRegressor", None) if _xgb_mod else None


def _dependencies_ok() -> bool:
    missing = []
    checks = [
        ("numpy", np),
        ("pandas", pd),
        ("streamlit", st),
        ("sklearn.preprocessing.LabelEncoder", LabelEncoder),
        ("sklearn.model_selection.KFold", KFold),
        ("sklearn.metrics.mean_squared_error", mean_squared_error),
        ("xgboost.XGBRegressor", XGBRegressor),
    ]
    for name, obj in checks:
        if obj is None:
            missing.append(name)
    if missing:
        msg = (
            "Missing Python packages: "
            + ", ".join(missing)
            + "\nInstall with: pip install streamlit xgboost scikit-learn pandas numpy"
        )
        if st:
            try:
                st.error(msg)
                st.stop()
            except Exception:
                pass
        else:
            print(msg)
        return False
    return True


# -----------------------------
# Streamlit App Configuration
# -----------------------------
if st:
    st.set_page_config(
        page_title="Accident Risk Predictor",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )


# -----------------------------
# Constants and Paths
# -----------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PKL = os.path.join(MODEL_DIR, "accident_xgb_model.pkl")
# Use a v2 artifact that stores a plain dict to avoid importing this module on unpickle
PREPROCESSOR_PKL = os.path.join(MODEL_DIR, "preprocessor_v2.pkl")
LEGACY_PREPROCESSOR_PKL = os.path.join(MODEL_DIR, "preprocessor.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# Preprocessor replicating notebook steps
# - Label encoding for categorical
# - Quantile binning for numeric (10 bins)
# - Frequency encoding for categorical
# -----------------------------
class Preprocessor:
    def __init__(self):
        self.cat_cols: List[str] = []
        self.num_cols: List[str] = []
        self.bool_cols: List[str] = []
        self.target_col: str = "accident_risk"

        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.freq_maps: Dict[str, Dict[str, float]] = {}
        self.bin_edges: Dict[str, np.ndarray] = {}
        self.feature_names_: List[str] = []

    def _identify_columns(self, df: pd.DataFrame):
        # Detect columns by dtype and known names
        self.cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        # Treat booleans separately
        self.bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
        # Numeric (excluding id and target)
        self.num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.num_cols = [c for c in self.num_cols if c not in ("id", self.target_col)]

        # Ensure expected columns exist even if dtypes differ
        for expected in [
            "road_type",
            "lighting",
            "weather",
            "time_of_day",
        ]:
            if expected in df.columns and expected not in self.cat_cols:
                self.cat_cols.append(expected)

        for expected in [
            "num_lanes",
            "curvature",
            "speed_limit",
            "num_reported_accidents",
        ]:
            if expected in df.columns and expected not in self.num_cols and expected not in self.bool_cols:
                self.num_cols.append(expected)

        for expected in [
            "road_signs_present",
            "public_road",
            "holiday",
            "school_season",
        ]:
            if expected in df.columns and expected not in self.bool_cols:
                self.bool_cols.append(expected)

    def fit(self, df: pd.DataFrame):
        self._identify_columns(df)

        # Label encoders
        for col in self.cat_cols:
            le = LabelEncoder()
            # Fit on string values
            le.fit(df[col].astype(str))
            self.label_encoders[col] = le

        # Frequency maps (relative frequencies)
        for col in self.cat_cols:
            vc = df[col].astype(str).value_counts(normalize=True)
            self.freq_maps[col] = vc.to_dict()

        # Quantile-based bin edges for numeric columns
        for col in self.num_cols:
            values = df[col].astype(float).to_numpy()
            # 11 quantiles yield 10 bins
            qs = np.linspace(0, 1, 11)
            edges = np.quantile(values, qs)
            # Ensure strictly increasing edges to avoid bin errors
            for i in range(1, len(edges)):
                if edges[i] <= edges[i - 1]:
                    edges[i] = edges[i - 1] + 1e-6
            self.bin_edges[col] = edges

        # Build feature names in the order that will be produced
        feature_names = []
        # Original numeric and boolean (as ints)
        feature_names += self.num_cols
        feature_names += self.bool_cols
        # Label-encoded categorical
        feature_names += [f"{c}_le" for c in self.cat_cols]
        # Numeric bins
        feature_names += [f"{c}_bin" for c in self.num_cols]
        # Frequency-encoded categorical
        feature_names += [f"{c}_freq" for c in self.cat_cols]

        self.feature_names_ = feature_names
        return self

    def _encode_row(self, row: pd.Series) -> pd.Series:
        data = {}
        # Numeric
        for col in self.num_cols:
            data[col] = float(row[col])
        # Boolean as ints
        for col in self.bool_cols:
            v = row[col]
            if isinstance(v, str):
                v = v.strip().lower() in ("true", "1", "yes", "y")
            data[col] = int(bool(v))
        # Label-encoded categorical
        for col in self.cat_cols:
            val = str(row[col])
            le = self.label_encoders[col]
            # Handle unseen values by mapping to most frequent class
            if val not in le.classes_.tolist():
                val = le.classes_[0]
            data[f"{col}_le"] = int(le.transform([val])[0])
        # Numeric bins via digitize
        for col in self.num_cols:
            edges = self.bin_edges[col]
            # digitize returns indices 1..len(edges)-1 for interior values
            idx = np.digitize([float(row[col])], edges, right=True)[0] - 1
            idx = int(np.clip(idx, 0, len(edges) - 2))
            data[f"{col}_bin"] = idx
        # Frequency encoding
        for col in self.cat_cols:
            val = str(row[col])
            freq_map = self.freq_maps[col]
            data[f"{col}_freq"] = float(freq_map.get(val, 0.0))
        return pd.Series(data)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in df.iterrows():
            rows.append(self._encode_row(row))
        X = pd.DataFrame(rows)
        # Reorder columns to match training order
        X = X[self.feature_names_]
        return X

    def transform_single(self, row_dict: Dict) -> pd.DataFrame:
        # Build a single-row DataFrame with original expected columns
        cols = set(self.num_cols + self.cat_cols + self.bool_cols)
        single = {c: row_dict.get(c) for c in cols}
        df = pd.DataFrame([single])
        return self.transform(df)


# -----------------------------
# Utilities for model persistence
# -----------------------------
def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def serialize_preprocessor(pre: 'Preprocessor') -> Dict:
    # Convert to a pure-serializable dict to avoid importing this module during load
    return {
        "cat_cols": pre.cat_cols,
        "num_cols": pre.num_cols,
        "bool_cols": pre.bool_cols,
        "target_col": pre.target_col,
        "label_encoders": {col: le.classes_.tolist() for col, le in pre.label_encoders.items()},
        "freq_maps": pre.freq_maps,
        "bin_edges": {col: edges.tolist() for col, edges in pre.bin_edges.items()},
        "feature_names_": pre.feature_names_,
    }


def deserialize_preprocessor(data: Dict) -> 'Preprocessor':
    pre = Preprocessor()
    pre.cat_cols = data.get("cat_cols", [])
    pre.num_cols = data.get("num_cols", [])
    pre.bool_cols = data.get("bool_cols", [])
    pre.target_col = data.get("target_col", "accident_risk")
    # Rebuild label encoders with preserved classes order
    pre.label_encoders = {}
    for col, classes in data.get("label_encoders", {}).items():
        le = LabelEncoder()
        # Directly assign classes_ to preserve encoding mapping
        if np is not None:
            le.classes_ = np.array(classes)
        else:
            # Fallback: attempt dynamic import, otherwise use list
            _np = _safe_import_module("numpy")
            if _np is not None:
                le.classes_ = _np.array(classes)
            else:
                le.classes_ = classes
        pre.label_encoders[col] = le
    pre.freq_maps = data.get("freq_maps", {})
    pre.bin_edges = {col: np.array(edges) for col, edges in data.get("bin_edges", {}).items()}
    pre.feature_names_ = data.get("feature_names_", [])
    return pre


def save_preprocessor(pre: 'Preprocessor', path: str):
    save_pickle(serialize_preprocessor(pre), path)


def load_preprocessor(path: str) -> Optional['Preprocessor']:
    try:
        data = load_pickle(path)
        if isinstance(data, dict):
            return deserialize_preprocessor(data)
        # If not a dict, it's likely an old artifact; avoid importing this module again.
        return None
    except Exception:
        return None


def try_load_model_and_preprocessor():
    model = None
    pre = None
    if os.path.exists(MODEL_PKL):
        try:
            model = load_pickle(MODEL_PKL)
        except Exception:
            model = None
    if os.path.exists(PREPROCESSOR_PKL):
        pre = load_preprocessor(PREPROCESSOR_PKL)
    return model, pre


# -----------------------------
# Training pipeline (aligned with notebook hyperparameters)
# -----------------------------
XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}


def _rmse(y_true, y_pred) -> float:
    """Compute RMSE compatible with older scikit-learn versions.

    Tries `mean_squared_error(..., squared=False)` first; if unsupported or
    unavailable, falls back to sqrt of MSE computed either via sklearn or
    pure Python/numpy.
    """
    try:
        if mean_squared_error is not None:
            # Newer sklearn supports the `squared` argument
            return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        # Older sklearn without `squared` parameter
        pass

    # Fallback paths
    if mean_squared_error is not None:
        try:
            return float(np.sqrt(mean_squared_error(y_true, y_pred))) if np is not None else float(mean_squared_error(y_true, y_pred) ** 0.5)
        except Exception:
            pass

    # Final fallback using numpy or pure Python
    try:
        if np is not None:
            y_true_arr = np.asarray(y_true, dtype=float)
            y_pred_arr = np.asarray(y_pred, dtype=float)
            return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))
        # Pure Python
        diffs = [float(a) - float(b) for a, b in zip(list(y_true), list(y_pred))]
        mse = sum(d * d for d in diffs) / max(1, len(diffs))
        return float(mse ** 0.5)
    except Exception:
        # If everything fails, return NaN-like value to signal an issue
        return float("nan")


def train_model(train_df: pd.DataFrame, progress_placeholder=None) -> Tuple[XGBRegressor, Preprocessor, pd.DataFrame, pd.Series]:
    pre = Preprocessor().fit(train_df)
    X = pre.transform(train_df)
    y = train_df[pre.target_col].astype(float)

    model = XGBRegressor(**XGB_PARAMS)
    if progress_placeholder:
        progress_placeholder.write("Training XGBoost model...")
    model.fit(X, y)

    # Persist artifacts
    save_pickle(model, MODEL_PKL)
    save_preprocessor(pre, PREPROCESSOR_PKL)

    return model, pre, X, y


def quick_cv_rmse(model: XGBRegressor, X: pd.DataFrame, y: pd.Series, n_splits: int = 3) -> float:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmses = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_val)
        rmse = _rmse(y_val, pred)
        rmses.append(rmse)
    return float(np.mean(rmses)) if np is not None else float(sum(rmses) / max(1, len(rmses)))


# -----------------------------
# Default options for UI (used before training)
# -----------------------------
DEFAULT_CATEGORIES = {
    "road_type": ["urban", "rural", "highway"],
    "lighting": ["daylight", "dim", "dark"],
    "weather": ["clear", "rainy", "foggy", "snowy"],
    "time_of_day": ["morning", "afternoon", "evening", "night"],
}


def make_input_form(pre: Optional[Preprocessor], key_prefix: str = "") -> Dict:
    # Derive choices from preprocessor if available
    def choices(col):
        if pre and col in pre.cat_cols:
            # Use original classes order
            return pre.label_encoders[col].classes_.tolist()
        return DEFAULT_CATEGORIES.get(col, [])

    c1, c2, c3 = st.columns(3)
    with c1:
        road_type = st.selectbox("Road type", choices("road_type"), index=0, key=f"{key_prefix}_road_type")
        num_lanes = st.number_input("Number of lanes", min_value=1, max_value=8, value=3, key=f"{key_prefix}_num_lanes")
        curvature = st.slider("Curvature", min_value=0.0, max_value=1.0, value=0.35, step=0.01, key=f"{key_prefix}_curvature")
        speed_limit = st.number_input("Speed limit (mph)", min_value=15, max_value=85, value=45, key=f"{key_prefix}_speed_limit")
    with c2:
        lighting = st.selectbox("Lighting", choices("lighting"), index=0, key=f"{key_prefix}_lighting")
        weather = st.selectbox("Weather", choices("weather"), index=0, key=f"{key_prefix}_weather")
        time_of_day = st.selectbox("Time of day", choices("time_of_day"), index=2, key=f"{key_prefix}_time_of_day")
        num_reported_accidents = st.number_input("Recent reported accidents", min_value=0, max_value=20, value=1, key=f"{key_prefix}_num_reported_accidents")
    with c3:
        road_signs_present = st.checkbox("Road signs present", value=True, key=f"{key_prefix}_road_signs_present")
        public_road = st.checkbox("Public road", value=True, key=f"{key_prefix}_public_road")
        holiday = st.checkbox("Holiday", value=False, key=f"{key_prefix}_holiday")
        school_season = st.checkbox("School season", value=True, key=f"{key_prefix}_school_season")

    return {
        "road_type": road_type,
        "num_lanes": num_lanes,
        "curvature": curvature,
        "speed_limit": speed_limit,
        "lighting": lighting,
        "weather": weather,
        "road_signs_present": road_signs_present,
        "public_road": public_road,
        "time_of_day": time_of_day,
        "holiday": holiday,
        "school_season": school_season,
        "num_reported_accidents": num_reported_accidents,
    }


def predict_single(model: XGBRegressor, pre: Preprocessor, row: Dict) -> float:
    Xs = pre.transform_single(row)
    yhat = float(model.predict(Xs)[0])
    return float(np.clip(yhat, 0.0, 1.0))


# -----------------------------
# Sidebar Navigation
# -----------------------------
PAGES = ["Overview", "Predict", "Explore", "Compare", "Batch", "Insights"]
page = st.sidebar.selectbox("Navigate", PAGES, index=1, key="nav_page")


# -----------------------------
# Model Setup Section (always visible in sidebar)
# -----------------------------
st.sidebar.markdown("### Model setup")
loaded_model, loaded_pre = try_load_model_and_preprocessor()
if "model" not in st.session_state:
    st.session_state.model = loaded_model
    st.session_state.pre = loaded_pre
    st.session_state.training_info = None

model_status = (
    "Loaded from disk" if st.session_state.model is not None else "Not loaded"
)
st.sidebar.write(f"Status: {model_status}")

# Auto-load or auto-train on first run so users can predict immediately
AUTO_TRAIN_ON_FIRST_RUN = True
default_train_path = os.path.join(os.path.dirname(__file__), "train.csv")
if (st.session_state.model is None or st.session_state.pre is None) and AUTO_TRAIN_ON_FIRST_RUN:
    # Try loading again in case files appeared (e.g., after a previous run)
    m2, p2 = try_load_model_and_preprocessor()
    if m2 is not None and p2 is not None:
        st.session_state.model = m2
        st.session_state.pre = p2
        st.sidebar.success("Model auto-loaded from disk.")
    elif os.path.exists(default_train_path) and pd is not None:
        try:
            ph = st.sidebar.empty()
            ph.write("Auto-training model from bundled train.csv…")
            train_df = pd.read_csv(default_train_path)
            model, pre, X, y = train_model(train_df, progress_placeholder=ph)
            st.session_state.model = model
            st.session_state.pre = pre
            rmse = quick_cv_rmse(model, X, y, n_splits=3)
            st.session_state.training_info = {"samples": len(train_df), "features": X.shape[1], "rmse": rmse}
            ph.write("Auto-training complete.")
            st.sidebar.success(f"Auto-trained model. Quick CV RMSE: {rmse:.5f}")
        except Exception as e:
            st.sidebar.error(f"Auto-training failed: {e}")
    else:
        st.sidebar.info("Upload or provide path to train.csv to enable auto-training.")

# Inform user if a legacy preprocessor artifact exists
if not os.path.exists(PREPROCESSOR_PKL) and os.path.exists(LEGACY_PREPROCESSOR_PKL):
    st.sidebar.warning(
        "Legacy preprocessor.pkl detected. Please retrain to create preprocessor_v2.pkl (fixes duplicate widget keys)."
    )

with st.sidebar.expander("Load saved artifacts"):
    if st.button("Load model & preprocessor from disk"):
        m, p = try_load_model_and_preprocessor()
        if m and p:
            st.session_state.model = m
            st.session_state.pre = p
            st.success("Model and preprocessor loaded.")
        else:
            st.warning("Saved artifacts not found. Train a model first.")

with st.sidebar.expander("Train from CSV"):
    uploaded_train = st.file_uploader("Upload Kaggle train.csv", type=["csv"], key="train_upload")
    path_input = st.text_input("Or provide local path to train.csv", value="")
    if st.button("Train model"):
        try:
            if uploaded_train is not None:
                train_df = pd.read_csv(uploaded_train)
            elif path_input and os.path.exists(path_input):
                train_df = pd.read_csv(path_input)
            else:
                st.error("Please upload a CSV or provide a valid path.")
                train_df = None

            if train_df is not None:
                ph = st.empty()
                ph.write("Preparing data and training model...")
                model, pre, X, y = train_model(train_df, progress_placeholder=ph)
                st.session_state.model = model
                st.session_state.pre = pre
                rmse = quick_cv_rmse(model, X, y, n_splits=3)
                st.session_state.training_info = {"samples": len(train_df), "features": X.shape[1], "rmse": rmse}
                ph.write("Training complete.")
                st.success(f"Model trained. Quick CV RMSE: {rmse:.5f}")
        except Exception as e:
            st.error(f"Training failed: {e}")


# -----------------------------
# Pages
# -----------------------------
st.title("Accident Risk Prediction")
st.caption(
    "Interact with the model from the Kaggle Playground Series S5E10 to explore how road design, weather, and context impact predicted accident risk."
)


if page == "Overview":
    st.subheader("Model & Data Overview")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Artifacts**")
        st.write({
            "Model path": MODEL_PKL,
            "Preprocessor path": PREPROCESSOR_PKL,
            "Loaded": st.session_state.model is not None,
        })
    with col2:
        st.markdown("**Training summary**")
        info = st.session_state.training_info or {}
        st.write({
            "Samples": info.get("samples", "-"),
            "Features": info.get("features", "-"),
            "Quick CV RMSE": info.get("rmse", "-"),
        })

    st.markdown("**Instructions**")
    st.write(
        "Use the sidebar to load saved artifacts or train from a CSV."
        " Then explore predictions, compare conditions, and run batch scoring."
    )

elif page == "Predict":
    st.subheader("Single Scenario Prediction")
    if st.session_state.model is None or st.session_state.pre is None:
        st.info("Load a saved model or train from CSV in the sidebar.")
    else:
        user_row = make_input_form(st.session_state.pre, key_prefix="predict")
        if st.button("Predict risk"):
            try:
                yhat = predict_single(st.session_state.model, st.session_state.pre, user_row)
                st.metric(label="Predicted accident risk", value=f"{yhat:.3f}")
                st.write("Inputs")
                st.json(user_row)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

elif page == "Explore":
    st.subheader("Variable Effect Explorer")
    if st.session_state.model is None or st.session_state.pre is None:
        st.info("Load a saved model or train from CSV in the sidebar.")
    else:
        base_row = make_input_form(st.session_state.pre, key_prefix="explore_base")
        variable = st.selectbox(
            "Variable to vary",
            ["num_lanes", "curvature", "speed_limit", "num_reported_accidents"],
            index=1,
            key="explore_variable",
        )
        # Define ranges
        ranges = {
            "num_lanes": (1, 8, 1),
            "curvature": (0.0, 1.0, 0.02),
            "speed_limit": (15, 85, 1),
            "num_reported_accidents": (0, 20, 1),
        }
        lo, hi, step = ranges[variable]
        values = []
        preds = []
        v = lo
        while v <= hi + 1e-9:
            row = dict(base_row)
            row[variable] = v
            try:
                yhat = predict_single(st.session_state.model, st.session_state.pre, row)
            except Exception:
                yhat = np.nan
            values.append(v)
            preds.append(yhat)
            v = round(v + step, 6)
        df_plot = pd.DataFrame({variable: values, "predicted_risk": preds})
        st.line_chart(df_plot.set_index(variable))
        st.caption("Predicted risk as a function of the selected variable, holding other inputs fixed.")

elif page == "Compare":
    st.subheader("Compare Two Conditions")
    if st.session_state.model is None or st.session_state.pre is None:
        st.info("Load a saved model or train from CSV in the sidebar.")
    else:
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("**Scenario A**")
            row_a = make_input_form(st.session_state.pre, key_prefix="compare_a")
        with c_right:
            st.markdown("**Scenario B**")
            row_b = make_input_form(st.session_state.pre, key_prefix="compare_b")
        if st.button("Compare"):
            ya = predict_single(st.session_state.model, st.session_state.pre, row_a)
            yb = predict_single(st.session_state.model, st.session_state.pre, row_b)
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Scenario A risk", f"{ya:.3f}")
            with c2:
                st.metric("Scenario B risk", f"{yb:.3f}")

elif page == "Batch":
    st.subheader("Batch Predictions from CSV")
    if st.session_state.model is None or st.session_state.pre is None:
        st.info("Load a saved model or train from CSV in the sidebar.")
    else:
        up = st.file_uploader("Upload CSV with columns matching the Kaggle dataset (without the target)", type=["csv"], key="batch_csv")
        if up is not None:
            try:
                df = pd.read_csv(up)
                # If accident_risk present, drop it for prediction
                if "accident_risk" in df.columns:
                    df = df.drop(columns=["accident_risk"])  # predicting fresh
                X = st.session_state.pre.transform(df)
                preds = st.session_state.model.predict(X)
                preds = np.clip(preds, 0.0, 1.0)
                out = df.copy()
                out["predicted_accident_risk"] = preds
                st.dataframe(out.head(30))
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

elif page == "Insights":
    st.subheader("Model Feature Insights")
    if st.session_state.model is None or st.session_state.pre is None:
        st.info("Load a saved model or train from CSV in the sidebar.")
    else:
        importances = getattr(st.session_state.model, "feature_importances_", None)
        if importances is None:
            st.warning("Feature importances not available for this model.")
        else:
            fi = pd.DataFrame({
                "feature": st.session_state.pre.feature_names_,
                "importance": importances,
            }).sort_values("importance", ascending=False)
            st.dataframe(fi.head(30))
            st.bar_chart(fi.head(30).set_index("feature"))


st.markdown("---")
st.caption("This app uses the training approach and hyperparameters from the Kaggle Playground Series S5E10 notebook to provide interactive accident risk predictions.")