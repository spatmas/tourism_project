import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# TITLE
st.title("Tourism Purchase Prediction")

st.write("Enter customer details to predict purchase behavior")

# LOAD MODEL

repo_id = "Sachinpp04/tourism-model"
model_path = hf_hub_download(
    repo_id=repo_id,
    filename="best_model.pkl",
    repo_type="model"
)

data = joblib.load(model_path)

model = data["model"]
model_columns = data["columns"]

# INPUT FORM
with st.form("input_form"):

    age = st.number_input("Age", 10, 100, 30)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    occupation = st.selectbox("Occupation", ["Salaried", "FreeLancer", "Small Business"])
    product = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "King"])

    submit = st.form_submit_button("Predict")

# PREDICTION
if submit:

    input_data = {
        "Age": age,
        "CityTier": city_tier,
        "Gender_" + gender: 1,
        "MaritalStatus_" + marital_status: 1,
        "Occupation_" + occupation: 1,
        "ProductPitched_" + product: 1
    }

    df = pd.DataFrame([input_data])

    # Align columns with training data
    df = df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(df)[0]

    # OUTPUT
    if prediction == 1:
        st.success("Customer will purchase")
    else:
        st.error("Customer will NOT purchase")

    st.write("Input data used:")
    st.dataframe(df)
