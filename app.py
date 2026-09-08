import streamlit as st
import pandas as pd 
import mlflow 
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model = mlflow.sklearn.load_model(
    "models:/sales_prediction_model@champion"
)


st.title("Advertising sales predictor")

TV = st.number_input("TV Budget")
radio = st.number_input("Radio Budget")
newspaper = st.number_input("Newspaper Budget")

if st.button("Predict sales"):

    input_data = pd.DataFrame(
        {
            "TV": [TV],
            "Radio": [radio],
            "Newspaper": [newspaper]
        }
    )

    prediction = model.predict(input_data)

    st.success(f"Predicted sales: {prediction[0]:.2f}")