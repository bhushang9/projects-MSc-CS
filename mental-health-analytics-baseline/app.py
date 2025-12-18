import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import joblib
from textblob import TextBlob
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

#python -m streamlit run app.py

st.set_page_config(
    page_title="Predictive Analytics for Mental Health",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for calm theme
st.markdown("""
    <style>
        /* Optional: Style only buttons slightly */
        .stButton>button {
            background-color: #2e6f95;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1.5em;
        }
        .stButton>button:hover {
            background-color: #4f9bbf;
        }
    </style>
""", unsafe_allow_html=True)


# Load trained model, scaler, and encoders
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

st.title("Predictive Analytics for Mental Health")
st.write("Please fill out the information below to assess your mental health risk:")

#For clear button
# Add session state initialization for clear button functionality
if 'form_cleared' not in st.session_state:
    st.session_state.form_cleared = False

# Function to clear form values
def clear_form():
    st.session_state["name_input"] = ""
    st.session_state["age_input"] = 25
    st.session_state["gender_input"] = "Male"
    st.session_state["family_history_input"] = "Yes"
    st.session_state["benefits_input"] = "Yes"
    st.session_state["care_options_input"] = "Yes"
    st.session_state["anonymity_input"] = "Yes"
    st.session_state["leave_input"] = "Very easy"
    st.session_state["work_interfere_input"] = "Never"
    st.session_state["journal_input"] = ""


# Clear form button - place it at the top of the form
col1, col2 = st.columns([4, 1])
with col2:
    st.button("Clear Form", on_click=clear_form, key="clear_button")


# Input Fields
# Collect input from user
name = st.text_input("What is your Name?", key="name_input")
age = st.number_input("How old are you?", min_value=0, max_value=100, value=25, key="age_input")
gender = st.selectbox("What is your Gender?", ["Male", "Female", "Other"], key="gender_input")
family_history = st.selectbox("Do you have a Family history of mental health issues?", ["Yes", "No"], key="family_history_input")
benefits = st.selectbox("Does your employer offer mental health benefits?", ["Yes", "No", "Don't know"], key="benefits_input")
care_options = st.selectbox("Are you aware of the mental health care options available through your employer?", ["Yes", "No", "Not sure"], key="care_options_input")
anonymity = st.selectbox("Would your anonymity be protected if you seek help for a mental health issue?", ["Yes", "No", "Don't know"], key="anonymity_input")
leave = st.selectbox("How easy is it for you to take time off for mental health reasons?", ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"], key="leave_input")
work_interfere = st.selectbox("How often does your mental health affect your ability to work?", ["Never", "Rarely", "Sometimes", "Often"], key="work_interfere_input")

journal_entry = st.text_area("Optional: Write how you're feeling (for sentiment analysis)")

# Progress bar
total_fields = 9
inputs = [age, gender, family_history, benefits, care_options, anonymity, leave, work_interfere, name]
completed = sum([1 for i in inputs if i and (str(i).strip() != "")])
completion_ratio = completed / total_fields
completion_percent = int(completion_ratio * 100)
st.markdown(f"**Form Completion: {completion_percent}%**")
st.progress(completion_ratio)


# Prediction Logic
# Calculate sentiment score
def get_sentiment_score(text):
    if not text.strip():
        return 0.0
    analysis = TextBlob(text)
    return round(analysis.sentiment.polarity, 2)

sentiment_score = get_sentiment_score(journal_entry)
st.write(f"Journal Sentiment Score: {sentiment_score}")

# Prepare input data
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'family_history': [family_history],
    'benefits': [benefits],
    'care_options': [care_options],
    'anonymity': [anonymity],
    'leave': [leave],
    'work_interfere': [work_interfere]
})

# Encode categorical features
for col in input_data.columns:
    if col in encoders:
        input_data[col] = encoders[col].transform(input_data[[col]])


# Scale numerical input
input_data_scaled = pd.DataFrame(
    scaler.transform(input_data),
    columns=input_data.columns
)


# Predict
if st.button("Predict"):
    try:
        prediction = model.predict(input_data_scaled)
        proba = model.predict_proba(input_data_scaled)[0]

        # Debug outputs
        st.write(f"Raw prediction: {prediction[0]}")
        st.write(f"Prediction Confidence: Yes → {proba[1]:.2%}, No → {proba[0]:.2%}")

        result = (
            "Yes, mental health treatment is advised." if prediction[0] in [1, "Yes"]
            else "No, treatment may not be necessary."
        )

        # Show prediction result in color
        st.subheader("Prediction Result:")
        if prediction[0] in [1, "Yes"]:
            st.markdown(f"<div style='color:red; font-size:20px; font-weight:bold'>{result}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color:green; font-size:20px; font-weight:bold'>{result}</div>", unsafe_allow_html=True)

        # Show sentiment score if journal is filled
        if journal_entry.strip():
            st.info(f"Sentiment Score from Journal: {sentiment_score}")

        # Show confidence bar chart
        # Feature importance (if supported)
        if hasattr(model, "feature_importances_"):
            st.subheader("Feature Contributions (Importance)")
            importance_df = pd.DataFrame({
                "Feature": input_data.columns,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            st.bar_chart(importance_df.set_index("Feature"))
        
        # Downloadable report
        st.subheader("Download Your Report")
        report_df = pd.DataFrame({
            "Name": [name],
            "Age": [age],
            "Prediction": [result],
            "Confidence_Yes (%)": [round(proba[1]*100, 2)],
            "Confidence_No (%)": [round(proba[0]*100, 2)],
            "Sentiment Score": [sentiment_score]
        })
        st.download_button(
            label="Download Report (CSV)",
            data=report_df.to_csv(index=False),
            file_name=f"{name.replace(' ', '_')}_mental_health_report.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


  





