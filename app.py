import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# Title and description
st.title("Calorie Prediction App")
st.write("This app predicts the calories burned during exercise based on user data.")

# Upload CSV files
st.sidebar.header("Upload your CSV files")
exercise_file = st.sidebar.file_uploader("Upload Exercise Data", type=["csv"])
calories_file = st.sidebar.file_uploader("Upload Calories Data", type=["csv"])

if exercise_file and calories_file:
    # Load datasets
    exercise_data = pd.read_csv(exercise_file)
    calories_data = pd.read_csv(calories_file)
    
    # Merge datasets
    combined_data = pd.concat([exercise_data, calories_data['Calories']], axis=1)
    st.write("### Combined Data Preview")
    st.dataframe(combined_data.head())

    # Data preprocessing
    combined_data.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)

    # Display basic stats
    st.write("### Data Summary")
    st.write(combined_data.describe())

    # Correlation heatmap
    if st.checkbox("Show Correlation Heatmap"):
        st.write("### Correlation Heatmap")
        correlation = combined_data.corr()
        plt.figure(figsize=(10, 10))
        sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
        st.pyplot(plt)

    # Prepare data for training
    X = combined_data.drop(columns=['User_ID', 'Calories'], axis=1)
    Y = combined_data['Calories']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    # Model training
    model = XGBRegressor()
    model.fit(X_train, Y_train)

    # Predictions
    test_data_prediction = model.predict(X_test)

    # Evaluation
    mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
    st.write("### Model Performance")
    st.write(f"Mean Absolute Error: {mae}")

    # User input for prediction
    st.write("### Predict Calories")
    gender = st.radio("Gender", options=['Male', 'Female'])
    age = st.number_input("Age", min_value=0, max_value=100, value=25)
    height = st.number_input("Height (in cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("Weight (in kg)", min_value=20, max_value=200, value=70)
    duration = st.number_input("Duration of Exercise (in minutes)", min_value=0, max_value=300, value=30)
    heart_rate = st.number_input("Heart Rate", min_value=50, max_value=200, value=100)
    body_temp = st.number_input("Body Temperature (in Celsius)", min_value=30, max_value=45, value=36.5)

    if st.button("Predict"):
        user_data = np.array([[0 if gender == 'Male' else 1, age, height, weight, duration, heart_rate, body_temp]])
        prediction = model.predict(user_data)
        st.write(f"### Predicted Calories Burned: {prediction[0]:.2f}")
else:
    st.write("Please upload both Exercise and Calories CSV files.")
