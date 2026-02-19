import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

st.title("Breast Cancer Stratification App")

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

st.subheader("Enter Tumor Details")

# Take user inputs (only few features for simplicity)
mean_radius = st.number_input("Mean Radius", min_value=0.0)
mean_texture = st.number_input("Mean Texture", min_value=0.0)
mean_perimeter = st.number_input("Mean Perimeter", min_value=0.0)
mean_area = st.number_input("Mean Area", min_value=0.0)

if st.button("Predict"):

    # Create input array (must match number of features)
    input_data = np.zeros((1, 30))
    input_data[0][0] = mean_radius
    input_data[0][1] = mean_texture
    input_data[0][2] = mean_perimeter
    input_data[0][3] = mean_area

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("The tumor is BENIGN")
    else:
        st.error("The tumor is MALIGNANT")
