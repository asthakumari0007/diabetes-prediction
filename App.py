import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.app-title {
    font-size: 36px;
    font-weight: 700;
    color: #17324d;
    margin-bottom: 2px;
}

.app-subtitle {
    font-size: 16px;
    color: #6b7280;
    margin-bottom: 30px;
}

.section-heading {
    font-size: 22px;
    font-weight: 650;
    color: #17324d;
    margin-top: 20px;
    margin-bottom: 5px;
}

.section-text {
    font-size: 14px;
    color: #6b7280;
    margin-bottom: 18px;
}

.result-positive {
    background-color: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 12px;
    padding: 25px;
    text-align: center;
    margin-top: 25px;
}

.result-negative {
    background-color: #ecfdf5;
    border: 1px solid #bbf7d0;
    border-radius: 12px;
    padding: 25px;
    text-align: center;
    margin-top: 25px;
}

.result-title {
    font-size: 30px;
    font-weight: 700;
}

.result-probability {
    font-size: 18px;
    margin-top: 8px;
}

.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 12px;
    margin-top: 45px;
    padding-top: 18px;
    border-top: 1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load("diabetes_prediction.pkl")
    scaler = joblib.load("scaler.pkl")

    return model, scaler


model, scaler = load_model()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="app-title">🩺 Diabetes Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-subtitle">'
    'Machine Learning Based Diabetes Prediction System'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# PATIENT INFORMATION
# =========================================================

st.markdown(
    '<div class="section-heading">Patient Information</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-text">'
    'Enter the patient details to predict the diabetes outcome.'
    '</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)


with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        max_value=250,
        value=120,
        step=1
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=150,
        value=70,
        step=1
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20,
        step=1
    )


with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=900,
        value=80,
        step=1
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=32.0,
        step=0.1
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.47,
        step=0.01
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=33,
        step=1
    )


# =========================================================
# PATIENT SUMMARY
# =========================================================

st.markdown(
    '<div class="section-heading">Patient Summary</div>',
    unsafe_allow_html=True
)


summary1, summary2, summary3, summary4 = st.columns(4)


with summary1:

    st.metric(
        "Age",
        age
    )


with summary2:

    st.metric(
        "Glucose",
        glucose
    )


with summary3:

    st.metric(
        "BMI",
        f"{bmi:.1f}"
    )


with summary4:

    st.metric(
        "Blood Pressure",
        blood_pressure
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.write("")

predict_button = st.button(
    "Predict Diabetes",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # Create input data
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree],
        "Age": [age]
    })


    # Scale the input
    scaled_data = scaler.transform(input_data)


    # Prediction
    prediction = model.predict(scaled_data)[0]


    # Prediction probability
    probability = model.predict_proba(scaled_data)[0]

    diabetes_probability = probability[1] * 100


    # =====================================================
    # RESULT
    # =====================================================

    if prediction == 1:

        st.markdown(
            f"""
            <div class="result-positive">

                <div class="result-title">
                    ⚠️ Diabetes Positive
                </div>

                <div class="result-probability">
                    Prediction Probability:
                    <b>{diabetes_probability:.2f}%</b>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="result-negative">

                <div class="result-title">
                    ✅ Diabetes Negative
                </div>

                <div class="result-probability">
                    Diabetes Probability:
                    <b>{diabetes_probability:.2f}%</b>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # PREDICTION DETAILS
    # =====================================================

    st.markdown(
        '<div class="section-heading">Prediction Details</div>',
        unsafe_allow_html=True
    )


    detail1, detail2, detail3 = st.columns(3)


    with detail1:

        st.metric(
            "Glucose",
            glucose
        )


    with detail2:

        st.metric(
            "BMI",
            f"{bmi:.1f}"
        )


    with detail3:

        st.metric(
            "Diabetes Probability",
            f"{diabetes_probability:.2f}%"
        )


    # =====================================================
    # SIMPLE INSIGHT
    # =====================================================

    if glucose >= 126:

        st.warning(
            "The entered glucose value is relatively high. "
            "The prediction should be interpreted carefully."
        )

    elif glucose < 100:

        st.info(
            "The entered glucose value is in a lower range. "
            "The final prediction is based on all input features."
        )

    else:

        st.info(
            "The prediction is based on the patient's "
            "health and demographic information."
        )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("Diabetes Prediction")

    st.write(
        "A machine learning based system for diabetes prediction."
    )

    st.divider()

    st.subheader("Model")

    st.write("Logistic Regression")

    st.divider()

    st.subheader("Input Features")

    st.write("• Pregnancies")
    st.write("• Glucose")
    st.write("• Blood Pressure")
    st.write("• Skin Thickness")
    st.write("• Insulin")
    st.write("• BMI")
    st.write("• Diabetes Pedigree Function")
    st.write("• Age")

    st.divider()

    st.caption(
        "For educational and demonstration purposes."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

        Diabetes Prediction • Machine Learning System

        <br>

        Built with Python, Scikit-learn and Streamlit

    </div>
    """,
    unsafe_allow_html=True
)
