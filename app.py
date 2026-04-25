#!/usr/bin/env python3
"""
Streamlit app for the final 8-predictor HCC extrahepatic metastasis
risk-stratification platform.

Scientific choices
------------------
- The displayed value is a model-derived risk score, not a calibrated absolute
  cumulative-incidence probability.
- Fixed score cutoffs are used for high/low-risk classification:
    LM cutoff = 0.487; BM cutoff = 0.490.
- No random noise, cosmetic smoothing, or arbitrary 10%-90% rescaling is used.
- The UI uses lower_snake_case canonical keys; the app also accepts older bundles with display-style training-time column names.
"""

from __future__ import annotations

from pathlib import Path
import re
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st


HORIZONS = ["6 months", "9 months", "12 months"]
ENFORCE_MONOTONIC_HORIZON_SCORES = True

LUNG_FEATURES = [
    "afp_ge400",
    "tumor_size_ge5",
    "pvtt",
    "child_pugh_bc",
    "no_antiviral_therapy",
    "tumor_number_multiple",
    "albi_grade_23",
    "smoking_history",
]

BONE_FEATURES = [
    "tumor_size_ge5",
    "pvtt",
    "afp_ge400",
    "alp_ge100",
    "no_antiviral_therapy",
    "cirrhosis",
    "albi_grade_23",
    "child_pugh_bc",
]

# Compatibility layer -------------------------------------------------------
# The UI and app logic use lower_snake_case canonical keys.
# Some earlier model bundles were trained with display-style column names
# such as "Tumor_Size_ge5", "PVTT", "AFP_ge400", and "Child_Pugh_BC".
# XGBoost/sklearn may require those original training-time column names at
# prediction, so we compare feature names after canonicalization but still
# pass the raw bundle column names into the model.
FEATURE_ALIASES = {
    "tumor_size": "tumor_size_ge5",
    "tumor_size_cm": "tumor_size_ge5",
    "tumor_size_ge_5": "tumor_size_ge5",
    "tumor_size_ge5": "tumor_size_ge5",
    "afp": "afp_ge400",
    "afp_ge_400": "afp_ge400",
    "afp_ge400": "afp_ge400",
    "alp": "alp_ge100",
    "alp_ge_100": "alp_ge100",
    "alp_ge100": "alp_ge100",
    "pvtt": "pvtt",
    "macrovascular_invasion": "pvtt",
    "child_pugh": "child_pugh_bc",
    "child_pugh_bc": "child_pugh_bc",
    "child_pugh_b_c": "child_pugh_bc",
    "albi_grade": "albi_grade_23",
    "albi_grade_23": "albi_grade_23",
    "albi_grade_2_3": "albi_grade_23",
    "antiviral_therapy": "no_antiviral_therapy",
    "no_antiviral_therapy": "no_antiviral_therapy",
    "no_antiviral_treatment": "no_antiviral_therapy",
    "tumor_number": "tumor_number_multiple",
    "tumor_number_multiple": "tumor_number_multiple",
    "multiple_tumors": "tumor_number_multiple",
    "smoking": "smoking_history",
    "smoking_history": "smoking_history",
    "cirrhosis": "cirrhosis",
}


def canonical_feature_name(name: str) -> str:
    clean = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    clean = re.sub(r"_+", "_", clean)
    return FEATURE_ALIASES.get(clean, clean)


def get_bundle_feature_pairs(bundle: dict) -> list[tuple[str, str]]:
    actual_features = (
        bundle.get("features")
        or bundle.get("feature_names")
        or bundle.get("feature_names_in_")
    )
    if actual_features is None:
        raise ValueError("Model bundle does not contain a feature list.")
    return [(str(raw), canonical_feature_name(str(raw))) for raw in list(actual_features)]

st.set_page_config(
    page_title="Tongji HCC EHM Prediction Platform",
    page_icon="🎯",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_bundle(path: str) -> dict:
    bundle_path = Path(path)
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the training script first, for example: "
            "python train_model_lung_8var.py --csv lung.csv --out-dir ."
        )
    bundle = joblib.load(bundle_path)
    if "models" not in bundle or "cutoff" not in bundle:
        raise ValueError(f"{path} is not a valid model bundle: missing models or cutoff.")
    if "features" not in bundle:
        if "feature_names" in bundle:
            bundle["features"] = list(bundle["feature_names"])
        elif "feature_names_in_" in bundle:
            bundle["features"] = list(bundle["feature_names_in_"])
        else:
            raise ValueError(f"{path} is not a valid model bundle: missing feature list.")
    bundle["_feature_pairs"] = get_bundle_feature_pairs(bundle)
    return bundle


def validate_bundle_features(bundle: dict, expected_features: list[str], model_name: str) -> None:
    pairs = bundle.get("_feature_pairs") or get_bundle_feature_pairs(bundle)
    actual_raw = [raw for raw, _ in pairs]
    actual_canonical = [canonical for _, canonical in pairs]
    if actual_canonical != expected_features:
        raise ValueError(
            f"{model_name} feature mismatch after canonicalization. "
            f"Expected {expected_features}, but bundle contains {actual_raw}, "
            f"which canonicalizes to {actual_canonical}."
        )


def predict_scores(bundle: dict, feature_values: dict[str, int]) -> list[tuple[str, str, float]]:
    """Return [(horizon, risk label, score), ...].

    DataFrame columns use the raw feature names stored in the bundle/model,
    while values are looked up using canonical lower_snake_case feature keys.
    """
    pairs = bundle.get("_feature_pairs") or get_bundle_feature_pairs(bundle)
    missing = [canonical for _, canonical in pairs if canonical not in feature_values]
    if missing:
        raise KeyError(f"Missing feature values: {missing}")

    X = pd.DataFrame(
        [{raw: int(feature_values[canonical]) for raw, canonical in pairs}],
        columns=[raw for raw, _ in pairs],
    )
    scores: list[float] = []
    for horizon in HORIZONS:
        model = bundle["models"][horizon]
        score = float(model.predict_proba(X)[:, 1][0])
        scores.append(score)

    if ENFORCE_MONOTONIC_HORIZON_SCORES:
        scores = np.maximum.accumulate(scores).tolist()

    cutoff = float(bundle["cutoff"])
    return [
        (horizon, "High risk" if score >= cutoff else "Low risk", float(score))
        for horizon, score in zip(HORIZONS, scores)
    ]


def yes_no_radio(label: str, default: str = "No", help_text: str | None = None) -> str:
    options = ["No", "Yes"]
    return st.sidebar.radio(label, options, index=options.index(default), help=help_text)


def display_result_card(label: str, risk: str, score: float, cutoff: float) -> None:
    bg = "#ffe5e5" if risk == "High risk" else "#e7f6e7"
    border = "#e76f51" if risk == "High risk" else "#2a9d8f"
    st.markdown(
        f"""
        <div style="
            background-color:{bg};
            border-left:6px solid {border};
            padding:14px 16px;
            border-radius:8px;
            margin-bottom:10px;">
          <div style="font-size:18px;font-weight:700;">{label}: {risk}</div>
          <div style="font-size:15px;margin-top:4px;">
            Model-derived risk score: <b>{score * 100:.1f}%</b>
            &nbsp;|&nbsp; Fixed score cutoff: <b>{cutoff * 100:.1f}%</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_prediction_button(button_label: str) -> bool:
    clicked = st.button(button_label, type="primary")
    if clicked:
        with st.spinner("Analyzing patient data..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.006)
                progress_bar.progress(i + 1)
    return clicked


def input_change_reset(key_prefix: str, feature_values: dict[str, int]) -> None:
    current_input = repr(sorted(feature_values.items()))
    last_key = f"{key_prefix}_last_input"
    pred_key = f"{key_prefix}_predicted"
    results_key = f"{key_prefix}_results"

    if last_key not in st.session_state:
        st.session_state[last_key] = current_input
    elif st.session_state[last_key] != current_input:
        st.session_state[pred_key] = False
        st.session_state[results_key] = None
        st.session_state[last_key] = current_input

    st.session_state.setdefault(pred_key, False)
    st.session_state.setdefault(results_key, None)


def show_export(results: list[tuple[str, str, float]], filename: str) -> None:
    result_df = pd.DataFrame({
        "time_point": [x[0] for x in results],
        "risk_category": [x[1] for x in results],
        "model_derived_risk_score_percent": [round(x[2] * 100, 1) for x in results],
    })
    st.download_button("📥 Download results", result_df.to_csv(index=False), filename, "text/csv")


def show_interpretation(results: list[tuple[str, str, float]], metastasis_site: str) -> None:
    labels = {h: risk for h, risk, _ in results}
    if labels["12 months"] == "High risk":
        st.warning(
            f"The patient is classified as high risk for {metastasis_site} metastasis "
            "by the locked 12-month operating cutoff. Consider closer imaging follow-up "
            "according to the study protocol and local clinical judgment."
        )
    elif labels["9 months"] == "High risk":
        st.info(
            f"The patient crosses the high-risk cutoff by 9 months for {metastasis_site} metastasis. "
            "Clinical vigilance is advised."
        )
    else:
        st.success(
            f"The patient remains below the fixed operating cutoff for {metastasis_site} metastasis "
            "through 12 months."
        )


def lung_page() -> None:
    st.title("🎯 Lung Metastasis Risk Prediction Tool")
    st.markdown(
        """
        This page uses the locked **8-predictor LM model**. The output is a
        **model-derived risk score** for risk stratification, not a calibrated
        absolute probability.
        """
    )

    try:
        bundle = load_bundle("lung_model_bundle.pkl")
        validate_bundle_features(bundle, LUNG_FEATURES, "LM")
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    st.sidebar.header("📝 Patient information: LM model")
    afp = yes_no_radio("AFP ≥ 400 ng/mL")
    tumor_size = yes_no_radio("Tumor size ≥ 5 cm")
    pvtt = yes_no_radio("PVTT / macrovascular invasion")
    child_pugh = st.sidebar.radio("Child-Pugh class", ["A", "B", "C"], index=0)
    antiviral = st.sidebar.radio("Antiviral therapy", ["Yes", "No"], index=0)
    tumor_number = st.sidebar.radio("Tumor number", ["Single", "Multiple"], index=0)
    albi = st.sidebar.radio("ALBI grade", ["1", "2", "3"], index=0)
    smoking = yes_no_radio("Smoking history")

    feature_values = {
        "afp_ge400": 1 if afp == "Yes" else 0,
        "tumor_size_ge5": 1 if tumor_size == "Yes" else 0,
        "pvtt": 1 if pvtt == "Yes" else 0,
        "child_pugh_bc": 1 if child_pugh in ["B", "C"] else 0,
        "no_antiviral_therapy": 1 if antiviral == "No" else 0,
        "tumor_number_multiple": 1 if tumor_number == "Multiple" else 0,
        "albi_grade_23": 1 if albi in ["2", "3"] else 0,
        "smoking_history": 1 if smoking == "Yes" else 0,
    }

    input_change_reset("lung", feature_values)
    if not st.session_state["lung_predicted"]:
        if run_prediction_button("Start prediction: Lung metastasis"):
            st.session_state["lung_results"] = predict_scores(bundle, feature_values)
            st.session_state["lung_predicted"] = True
            st.success("✅ Prediction completed")

    if st.session_state["lung_predicted"] and st.session_state["lung_results"]:
        st.subheader("📊 Prediction results")
        results = st.session_state["lung_results"]
        cutoff = float(bundle["cutoff"])
        for horizon, risk, score in results:
            display_result_card(horizon, risk, score, cutoff)
        show_export(results, "lung_metastasis_prediction_results.csv")
        show_interpretation(results, "lung")


def bone_page() -> None:
    st.title("🦴 Bone Metastasis Risk Prediction Tool")
    st.markdown(
        """
        This page uses the locked **8-predictor BM model**. The output is a
        **model-derived risk score** for risk stratification, not a calibrated
        absolute probability.
        """
    )

    try:
        bundle = load_bundle("bone_model_bundle.pkl")
        validate_bundle_features(bundle, BONE_FEATURES, "BM")
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    st.sidebar.header("📝 Patient information: BM model")
    tumor_size = yes_no_radio("Tumor size ≥ 5 cm")
    pvtt = yes_no_radio("PVTT / macrovascular invasion")
    afp = yes_no_radio("AFP ≥ 400 ng/mL")
    alp = yes_no_radio("ALP ≥ 100 U/L")
    antiviral = st.sidebar.radio("Antiviral therapy", ["Yes", "No"], index=0)
    cirrhosis = yes_no_radio("Cirrhosis")
    albi = st.sidebar.radio("ALBI grade", ["1", "2", "3"], index=0)
    child_pugh = st.sidebar.radio("Child-Pugh class", ["A", "B", "C"], index=0)

    feature_values = {
        "tumor_size_ge5": 1 if tumor_size == "Yes" else 0,
        "pvtt": 1 if pvtt == "Yes" else 0,
        "afp_ge400": 1 if afp == "Yes" else 0,
        "alp_ge100": 1 if alp == "Yes" else 0,
        "no_antiviral_therapy": 1 if antiviral == "No" else 0,
        "cirrhosis": 1 if cirrhosis == "Yes" else 0,
        "albi_grade_23": 1 if albi in ["2", "3"] else 0,
        "child_pugh_bc": 1 if child_pugh in ["B", "C"] else 0,
    }

    input_change_reset("bone", feature_values)
    if not st.session_state["bone_predicted"]:
        if run_prediction_button("Start prediction: Bone metastasis"):
            st.session_state["bone_results"] = predict_scores(bundle, feature_values)
            st.session_state["bone_predicted"] = True
            st.success("✅ Prediction completed")

    if st.session_state["bone_predicted"] and st.session_state["bone_results"]:
        st.subheader("📊 Prediction results")
        results = st.session_state["bone_results"]
        cutoff = float(bundle["cutoff"])
        for horizon, risk, score in results:
            display_result_card(horizon, risk, score, cutoff)
        show_export(results, "bone_metastasis_prediction_results.csv")
        show_interpretation(results, "bone")


if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    st.markdown(
        "<h1 style='color:#2a9d8f;'>Tongji HCC Extrahepatic Metastasis Prediction Platform</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        This platform provides score-based risk stratification for first-site
        lung and bone metastasis after HCC treatment.

        The fixed operating cutoffs are applied to model-derived risk scores:
        **LM = 0.487**, **BM = 0.490**.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎯 Lung metastasis prediction", use_container_width=True):
            st.session_state.page = "lung"
            st.rerun()
    with col2:
        if st.button("🦴 Bone metastasis prediction", use_container_width=True):
            st.session_state.page = "bone"
            st.rerun()

elif st.session_state.page == "lung":
    st.sidebar.button("🏠 Return to home", on_click=lambda: st.session_state.update(page="home"))
    lung_page()

elif st.session_state.page == "bone":
    st.sidebar.button("🏠 Return to home", on_click=lambda: st.session_state.update(page="home"))
    bone_page()
