import streamlit as st
import pandas as pd
import joblib


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)


# =====================================================
# LOAD MODEL AND SCALER
# =====================================================

model = joblib.load("diabetes_prediction.pkl")
scaler = joblib.load("scaler.pkl")


# =====================================================
# TITLE
# =====================================================

st.title("🩺 Diabetes Prediction System")

st.write(
    "This application uses Logistic Regression "
    "to predict the likelihood of diabetes."
)


# =====================================================
# USER INPUT
# =====================================================

st.subheader("Enter Patient Details")


Pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1
)


Glucose = st.number_input(
    "Glucose",
    min_value=0,
    max_value=300,
    value=120
)


BloodPressure = st.number_input(
    "Blood Pressure",
    min_value=0,
    max_value=200,
    value=70
)


SkinThickness = st.number_input(
    "Skin Thickness",
    min_value=0,
    max_value=100,
    value=20
)


Insulin = st.number_input(
    "Insulin",
    min_value=0,
    max_value=900,
    value=80
)


BMI = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)


DiabetesPedigreeFunction = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)


Age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)


# =====================================================
# PREDICTION
# =====================================================

if st.button("🔍 Predict Diabetes"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Pregnancies": [Pregnancies],
        "Glucose": [Glucose],
        "BloodPressure": [BloodPressure],
        "SkinThickness": [SkinThickness],
        "Insulin": [Insulin],
        "BMI": [BMI],
        "DiabetesPedigreeFunction": [
            DiabetesPedigreeFunction
        ],
        "Age": [Age]
    })


    # Scale input
    input_scaled = scaler.transform(input_data)


    # Prediction
    prediction = model.predict(input_scaled)[0]


    # Probability
    probability = model.predict_proba(input_scaled)[0][1]


    # =================================================
    # RESULT
    # =================================================

    st.subheader("Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ The model predicts a higher likelihood "
            "of diabetes."
        )

    else:

        st.success(
            "✅ The model predicts a lower likelihood "
            "of diabetes."
        )


    st.info(
        f"Predicted Probability: "
        f"{probability * 100:.2f}%"
    )


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Logistic Regression | Diabetes Prediction Project"
)
