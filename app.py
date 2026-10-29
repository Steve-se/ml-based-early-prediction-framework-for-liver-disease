
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

# Locate the saved model relative to this file
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "liver_disease_pipeline.pkl"

st.set_page_config(
    page_title="Liver Disease Prediction",
    page_icon="🩺",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("Liver Disease Prediction System")
st.write(
    "Enter the patient's information and laboratory values "
    "to obtain an experimental machine learning prediction."
)

st.warning(
    "For educational and research purposes only. "
    "This model is not a validated diagnostic tool and must "
    "not replace professional medical assessment."
)

with st.form("patient_form"):
    st.subheader("Patient Information")

    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=35
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    st.subheader("Laboratory Results")

    col1, col2 = st.columns(2)

    with col1:
        tot_bilirubin = st.number_input(
            "Total Bilirubin",
            min_value=0.0,
            value=0.8
        )

        alkphos = st.number_input(
            "Alkaline Phosphatase",
            min_value=0.0,
            value=200.0
        )

        sgot = st.number_input(
            "SGOT / AST",
            min_value=0.0,
            value=25.0
        )

        tot_proteins = st.number_input(
            "Total Proteins",
            min_value=0.0,
            value=6.8
        )

    with col2:
        direct_bilirubin = st.number_input(
            "Direct Bilirubin",
            min_value=0.0,
            value=0.2
        )

        sgpt = st.number_input(
            "SGPT / ALT",
            min_value=0.0,
            value=25.0
        )

        albumin = st.number_input(
            "Albumin",
            min_value=0.0,
            value=3.5
        )

        albumin_globulin_ratio = st.number_input(
            "Albumin and Globulin Ratio",
            min_value=0.0,
            value=1.0
        )

    submitted = st.form_submit_button(
        "Predict"
    )

if submitted:
    patient = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "tot_bilirubin": tot_bilirubin,
        "direct_bilirubin": direct_bilirubin,
        "alkphos": alkphos,
        "sgpt": sgpt,
        "sgot": sgot,
        "tot_proteins": tot_proteins,
        "albumin": albumin,
        "ag_ratio": albumin_globulin_ratio
    }])

    try:
        prediction = int(model.predict(patient)[0])

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("Model result: Liver-disease class predicted.")
        else:
            st.success("Model result: No-liver-disease class predicted.")
        
        st.info(
            "This result is an experimental machine learning prediction, "
            "not a medical diagnosis. A qualified healthcare professional "
            "should interpret the patient's laboratory results."
        )

        st.caption(
            "This is a model prediction, not a diagnosis. "
            "A healthcare professional should interpret "
            "laboratory results and determine any next steps."
        )

    except Exception as exc:
        st.error(
            "Prediction failed. Please verify that the model "
            "file and input feature names match the training pipeline."
        )
        st.exception(exc)


        
        report = patient.copy()
        
        report["predicted_class"] = (
            "Liver disease"
            if prediction == 1
            else "No liver disease"
        )
        
        report["notice"] = (
            "Experimental model output; not a medical diagnosis."
        )
        
        csv_data = report.to_csv(index=False)
        
        st.download_button(
            label="Download Prediction Report (CSV)",
            data=csv_data,
            file_name="liver_disease_prediction.csv",
            mime="text/csv"
        )
