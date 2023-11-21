import os
import streamlit as st
import numpy as np
from src.Mlflow_Project.pipeline.prediction import ScalingPipeline, PredictionPipeline

st.title('Induction Furnace Carbon Sulphur Response')

@st.cache_data
def train_pipeline():
    # Train your pipeline here if required
    os.system("python main.py")
    return "Training Successful"

@st.cache_data
def make_prediction(data):
    # Load and use your prediction pipeline here
    scl = ScalingPipeline()
    data_scaled = scl.scale(data)

    obj = PredictionPipeline()
    predict = obj.predict(data_scaled)
    return predict

def main():
    menu = ["Home", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home Page")
        # You can display any content for the home page here
        st.write("Furnace User Interface")
        st.write("This interface we will use for data collection from the site ")
        

    elif choice == "Prediction":
        st.subheader("Carbon Sulphur Prediction")
        st.write("Enter the values to make a prediction")
        # Form inputs
        laddle_temp = st.number_input("LADDLE_TEMP", value=0, min_value=None, max_value=None, step=1)
        casting_temp = st.number_input("CASTING_TEMP", value=0, min_value=None, max_value=None, step=1)
        furnace_cooling_pressure = st.number_input("FURNACE_COOLING_PRESSURE", value=0.0, min_value=None, max_value=None,
                                                   step=0.1)
        convertor_cooling_pressure = st.number_input("CONVERTOR_COOLING_PRESSURE", value=0.0, min_value=None, max_value=None, step=0.1)
        sponge_iron_MT = st.number_input("SPONGE_IRON_MT", value=0.0, min_value=None, max_value=None, step=0.1)
        scrap_MT = st.number_input("SCRAP_MT", value=0.0, min_value=None, max_value=None, step=0.1)
        pig_iron_KG = st.number_input("PIG_IRON_KG", value=0, min_value=None, max_value=None, step=1)
        cpc_KG = st.number_input("CPC_KG", value=0, min_value=None, max_value=None, step=1)
        simn_MT = st.number_input("SiMn_KG", value=0.0, min_value=None, max_value=None, step=0.1)
        aluminium_KG = st.number_input("AluminiumBar_KG", value=0.0, min_value=None, max_value=None, step=0.1)



        if st.button("Predict"):
            try:
                # Prepare data for prediction
                data = np.array([laddle_temp, casting_temp, furnace_cooling_pressure,
                                convertor_cooling_pressure, sponge_iron_MT, scrap_MT,
                                pig_iron_KG , cpc_KG,  simn_MT, aluminium_KG])  
                predictions = make_prediction(data.reshape(1, -1))
                # Extract Carbon and Sulphur predictions
                carbon_prediction = [pred[0] for pred in predictions]
                sulphur_prediction = [pred[1] for pred in predictions]
                st.success(f"Carbon Prediction: {carbon_prediction}")
                st.success(f"Sulphur Prediction: {sulphur_prediction}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == '__main__':
    main()
