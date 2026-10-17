import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title='Delivery Delay Prediction App', page_icon='📦', layout='centered'
)

st.title('📦 ABC Logistics - Delivery Delay Prediction System')
st.markdown(
    'Provide the delivery details below to predict whether the package will'
    ' experience a delay.'
)


# Load the trained model safely
@st.cache_resource
def load_model():
  # Ensure 'delivery_model.sav' is in the same folder as this script in your GitHub repo
  return joblib.load('delivery_model.sav')


try:
  model = load_model()
except Exception as e:
  st.error(
      'Error loading model file (`delivery_model.sav`). Please ensure it is'
      ' uploaded to your repository.'
  )
  st.stop()

# Create input widgets matching your feature columns
st.subheader('Enter Delivery Parameters')

col1, col2 = st.columns(2)

with col1:
  delivery_distance = st.number_input(
      'Delivery Distance (km)', min_value=0.0, max_value=100.0, value=15.0
  )
  traffic_congestion = st.slider(
      'Traffic Congestion Score (1-5)', min_value=1, max_value=5, value=3
  )
  weather_condition = st.selectbox(
      'Weather Condition (1: Good, 2: Moderate, 3: Bad)', [1, 2, 3]
  )
  delivery_slot = st.selectbox('Delivery Slot (1, 2, 3)', [1, 2, 3])
  driver_experience = st.number_input(
      'Driver Experience (Years)', min_value=0, max_value=30, value=5
  )
  num_stops = st.number_input('Number of Stops', min_value=1, max_value=15, value=2)

with col2:
  vehicle_age = st.number_input('Vehicle Age (Years)', min_value=0, max_value=20, value=3)
  road_condition_score = st.slider(
      'Road Condition Score (1-5)', min_value=1, max_value=5, value=2
  )
  package_weight = st.number_input(
      'Package Weight (kg)', min_value=0.0, max_value=100.0, value=10.0
  )
  fuel_efficiency = st.number_input(
      'Fuel Efficiency (km/l)', min_value=1.0, max_value=30.0, value=12.0
  )
  warehouse_processing_time = st.number_input(
      'Warehouse Processing Time (mins)', min_value=0, max_value=200, value=30
  )

# Organize input into a DataFrame with correct feature names
input_data = pd.DataFrame(
    [[
        delivery_distance,
        traffic_congestion,
        weather_condition,
        delivery_slot,
        driver_experience,
        num_stops,
        vehicle_age,
        road_condition_score,
        package_weight,
        fuel_efficiency,
        warehouse_processing_time,
    ]],
    columns=[
        'Delivery_Distance',
        'Traffic_Congestion',
        'Weather_Condition',
        'Delivery_Slot',
        'Driver_Experience',
        'Num_Stops',
        'Vehicle_Age',
        'Road_Condition_Score',
        'Package_Weight',
        'Fuel_Efficiency',
        'Warehouse_Processing_Time',
    ],
)

# Prediction button
if st.button('Predict Delivery Status', type='primary'):
  prediction = model.predict(input_data)[0]

  st.subheader('Prediction Result:')
  if prediction == 1:
    st.error(
        '⚠️ **Warning:** This delivery is predicted to be **Delayed**.'
    )
  else:
    st.success(
        '✅ **Success:** This delivery is predicted to arrive **On Time**.'
    )
