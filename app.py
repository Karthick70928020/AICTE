
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb
import plotly.express as px
import xgboost as xgb
import pickle

st.set_page_config(page_title="EV Forecasting App", layout="wide")
st.title("Electric Vehicle Forecasting and Analysis")

# Upload CSV
uploaded_file = st.file_uploader("Upload your EV dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Basic EDA
    st.subheader("Basic Statistics")
    st.write(df.describe())

    # Feature selection and preprocessing
    st.subheader("Model Training and Forecasting")
    if st.button("Train Model"):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
        from sklearn.ensemble import RandomForestRegressor

        # Example: Assume last column is the target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.success(f"Model trained! RMSE: {rmse:.2f}")

        # Save model
        with open("ev_model.pkl", "wb") as f:
            pickle.dump(model, f)

        st.download_button("Download Trained Model", data=open("ev_model.pkl", "rb"), file_name="ev_model.pkl")

        # Plot actual vs predicted
        plt.figure(figsize=(10, 4))
        plt.plot(y_test.values, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.legend()
        st.pyplot(plt)

        # SHAP explainability
        st.subheader("Model Explainability with SHAP")
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot()


