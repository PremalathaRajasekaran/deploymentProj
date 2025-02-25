import numpy as np
import joblib
import streamlit as st

# Load the model from the file
model = joblib.load('model_scaled.pkl')
scaler = joblib.load('scaled.pkl')

#streamlit application title
st.title('Diabetes Prediction App')
st.write('Enter your details to predict the diabetes')

#defining the input fields
st.sidebar.header('User Input Parameters')

feature_1 = st.sidebar.slider('Pregnancies', min_value=0.0, max_value=17.0, value=3.0, step=0.1)
feature_2 = st.sidebar.slider('Glucose', min_value=0.0, max_value=199.0, value=117.0, step=0.1)
feature_3 = st.sidebar.slider('Blood Pressure', min_value=0.0, max_value=122.0, value=72.0, step=0.1)
feature_4 = st.sidebar.slider('Skin Thickness', min_value=0.0, max_value=99.0, value=23.0, step=0.1)
feature_5 = st.sidebar.slider('Insulin', min_value=0.0, max_value=846.0, value=30.0, step=0.1)
feature_6 = st.sidebar.slider('BMI', min_value=0.0, max_value=67.1, value=32.0, step=0.1)
feature_7 = st.sidebar.slider('Diabetes Pedigree Function', min_value=0.078, max_value=2.42, value=0.3725, step=0.001)
feature_8 = st.sidebar.slider('Age', min_value=21.0, max_value=81.0, value=29.0, step=0.1)

input_data = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]])
scaled_input = scaler.transform(input_data)

if st.button('Predict'):
    # Predict the output
    result = model.predict(scaled_input)
    if result[0] == 1:
        st.write('You have diabetes')
        st.success(f'Please consult a doctor:::{result[0]}')
    else:
        st.write('You do not have diabetes')
        st.success(f'You are healthy:::{result[0]}')