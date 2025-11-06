import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Leadership Pipeline Digital Twin Simulation", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/Data.csv")

@st.cache_resource
def load_model():
    model_path = "models/attrition_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.warning(f"Model file not found at {model_path}. Please upload it to enable predictions.")
        return None

data = load_data()
model = load_model()

st.title("Leadership Pipeline Digital Twin Simulation App")

#-----------------------------------------#
# 1. Data Overview
#-----------------------------------------#
st.header("🔍 1. Data Overview")
st.write("Preview of the Dataset:")
st.dataframe(data.head())

st.write("Summary Statistics:")
st.write(data.describe())

#-----------------------------------------#
# 2. Model Status
#-----------------------------------------#
st.header("🧠 2. Attrition Model Status")
if model:
    st.success("✅ Attrition model successfully loaded!")
else:
    st.error("❌ Model not loaded. Please add `attrition_model.pkl` to the `models` directory.")

#-----------------------------------------#
# 3. Research Question
#-----------------------------------------#
st.header("🎯 3. Research Question")
st.markdown("**Can digital twin simulations more accurately predict mid-level leadership gaps in the tech industry compared to traditional succession planning?**")

st.info("Further visualization, simulation, and insights will be added in the next version.")
