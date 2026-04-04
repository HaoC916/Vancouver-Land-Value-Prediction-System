from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.infer.predict import LandValuePredictor


def _fmt_currency(x: float) -> str:
    return f"${x:,.0f}"


st.set_page_config(page_title="Vancouver Land Value Estimator (Demo)", layout="wide")

st.title("Vancouver Land Value Estimator (Demo)")
st.caption(
    "Course-project demo. This app estimates assessed land value (not guaranteed sale price)."
)

@st.cache_resource
def load_predictor():
    return LandValuePredictor()


try:
    predictor = load_predictor()
except Exception as e:
    st.error(str(e))
    st.info("Please run `python -m src.models.train_model` before launching the app.")
    st.stop()

st.markdown(
    """
This demo uses the project’s final trained model and a small set of user inputs.
Internal model features are completed using derived fields and lookup values from
the merged model table.
"""
)

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Property Inputs")
    with st.form("predict_form"):
        postal_code = st.text_input("PROPERTY_POSTAL_CODE", value="V6B1A1")

        legal_options = predictor.get_top_options("LEGAL_TYPE", top_n=100)
        zoning_dist_options = predictor.get_top_options("ZONING_DISTRICT", top_n=150)
        zoning_class_options = predictor.get_top_options("ZONING_CLASSIFICATION", top_n=200)
        neigh_options = predictor.get_top_options("NEIGHBOURHOOD_CODE", top_n=200)

        legal_type = st.selectbox(
            "LEGAL_TYPE",
            options=legal_options if legal_options else ["Unknown"],
        )
        zoning_district = st.selectbox(
            "ZONING_DISTRICT",
            options=zoning_dist_options if zoning_dist_options else ["Unknown"],
        )
        zoning_classification = st.selectbox(
            "ZONING_CLASSIFICATION",
            options=zoning_class_options if zoning_class_options else ["Unknown"],
        )
        neighbourhood_code = st.selectbox(
            "NEIGHBOURHOOD_CODE",
            options=neigh_options if neigh_options else ["Unknown"],
        )

        year_built = st.number_input(
            "YEAR_BUILT",
            min_value=1800,
            max_value=2100,
            value=1990,
            step=1,
        )

        st.caption(
            "Near-range estimation only: choose REPORT_YEAR from 2024 to 2027. "
            "Far-future years are intentionally disabled because lookup features "
            "are only reliable near the observed data horizon."
        )
        report_year = st.number_input(
            "REPORT_YEAR (optional)",
            min_value=2024,
            max_value=2027,
            value=2026,
            step=1,
        )

        has_big_improvement = st.checkbox("Provide BIG_IMPROVEMENT_YEAR", value=False)
        if has_big_improvement:
            big_improvement_year = st.number_input(
                "BIG_IMPROVEMENT_YEAR",
                min_value=1800,
                max_value=2100,
                value=2010,
                step=1,
            )
        else:
            big_improvement_year = None

        submit = st.form_submit_button("Estimate Land Value")

with col_right:
    st.subheader("Estimated Result")
    if submit:
        normalized_postal = predictor.normalize_postal_code(postal_code)
        if not predictor.is_valid_canadian_postal_code(normalized_postal):
            st.error(
                "Please enter a valid Canadian postal code (example: V6H2J4 or V6H 2J4)."
            )
        else:
            if not predictor.is_postal_code_seen(normalized_postal):
                st.warning(
                    "This postal code was not seen in the training data. "
                    "The estimate may be less reliable."
                )

            user_input = {
                "PROPERTY_POSTAL_CODE": normalized_postal,
                "LEGAL_TYPE": legal_type,
                "ZONING_DISTRICT": zoning_district,
                "ZONING_CLASSIFICATION": zoning_classification,
                "NEIGHBOURHOOD_CODE": neighbourhood_code,
                "YEAR_BUILT": year_built,
                "BIG_IMPROVEMENT_YEAR": big_improvement_year,
                "REPORT_YEAR": report_year,
            }
            try:
                result = predictor.predict(user_input)
                st.metric("Point Estimate (Assessed Land Value)", _fmt_currency(result.point_estimate))
                st.markdown(
                    f"**Estimated Range:** {_fmt_currency(result.lower_bound)} to {_fmt_currency(result.upper_bound)}"
                )
                st.caption(
                    f"Range uses error band: {_fmt_currency(result.error_band)} "
                    f"(source: `{result.error_band_source}`)."
                )

                st.markdown("### Interpretation")
                st.write(
                    "This estimate is driven mainly by legal type, postal/location context, zoning, "
                    "and neighbourhood-related signals in the model."
                )

                with st.expander("Derived and lookup details"):
                    st.json(result.used_features)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.markdown("---")
st.markdown(
    """
**Disclaimer**  
This is a course-project demo based on historical assessment data and engineered features.  
It estimates land assessment value, not a guaranteed market sale price.  
Results should be interpreted as approximate.
"""
)

artifacts_path = Path("artifacts/land_value_model.joblib")
if artifacts_path.exists():
    st.caption(f"Loaded model artifact: `{artifacts_path}`")
