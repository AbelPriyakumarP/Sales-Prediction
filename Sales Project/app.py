import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Display the dataset
st.title("Sales Prediction App")
st.write("Sales Data")




# User input for prediction
st.sidebar.header("User Input")
years = st.sidebar.number_input("Advertising Budget", min_value=0, step=1, value=1)
jobrate = st.sidebar.number_input("Price", min_value=0.0, step = 0.5, value=3.5)
model = joblib.load('Linearmodel.pkl')
predict = st.button('Press the button for salary prediction')
X=[years, jobrate]

if predict:
    st.balloons()
    X1 = np.array([X])
    prediction = model.predict(X1)
    st.write(f'Salary Predciton is {prediction}')
    

# End of app
st.sidebar.markdown("---")
st.sidebar.write("Make Predictions")
