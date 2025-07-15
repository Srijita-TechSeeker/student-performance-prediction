import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('student_model.pkl')

st.title("🎓 Student Pass/Fail Predictor")

# Input fields
study_hours = st.slider("Study Hours per Week", 0, 40, 10)
attendance = st.slider("Attendance Rate (%)", 0, 100, 75)
assignment_completion = st.slider("Assignment Completion (%)", 0, 100, 80)
online_courses = st.number_input("Online Courses Completed", 0, 10, 1)

# Combine inputs for prediction
features = np.array([[study_hours, attendance, assignment_completion, online_courses]])

# Predict
if st.button("Predict"):
    result = model.predict(features)
    st.success("✅ Prediction: Passed" if result[0] == 1 else "❌ Prediction: Failed")
