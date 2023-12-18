import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle
from sklearn.preprocessing import OneHotEncoder

encoder = pickle.load(open('road_encoder.pkl', 'rb'))
oe = pickle.load(open('road_oe.pkl', 'rb'))
df = pickle.load(open('road_df.pkl', 'rb'))
model = pickle.load(open('dlmodel.pkl', 'rb'))

Time = st.selectbox("Select what the time of accident", df['Time'].unique())
Day_of_week = st.selectbox("Select Week", df['Day_of_week'].unique())
Age_band_of_driver = st.selectbox("Select Age Band", df['Age_band_of_driver'].unique())
Sex_of_driver= st.selectbox("Sex of a driver",df['Sex_of_driver'].unique())
Educational_level=st.selectbox("Select Educational Level",df['Educational_level'].unique())
Vehicle_driver_relation=st.selectbox("Select Vehicle Driver Relationship",df['Vehicle_driver_relation'].unique())
Driving_experience = st.selectbox("Select Driving Experience", df['Driving_experience'].unique())
Type_of_vehicle=st.selectbox("Select the type of vehicle",df['Type_of_vehicle'].unique())
Owner_of_vehicle=st.selectbox("Select Owner of Vehicle",df['Owner_of_vehicle'].unique())
Service_year_of_vehicle=st.selectbox("Select the service year of vehicle",df['Service_year_of_vehicle'].unique())
Area_accident_occured = st.selectbox("Select an area where accident was occured", df['Area_accident_occured'].unique())
Lanes_or_Medians=st.selectbox("Select Lanes or Medians",df['Lanes_or_Medians'].unique())
Road_allignment=st.selectbox("Select Road allignment",df['Road_allignment'].unique())
Types_of_Junction = st.selectbox("Select a type of Junction", df['Types_of_Junction'].unique())
Road_surface_type=st.selectbox("Select Road surface type",df['Road_surface_type'].unique())
Road_surface_conditions=st.selectbox("Select Road accident conditions",df['Road_surface_conditions'].unique())
Light_conditions = st.selectbox("Select Light Condition", df['Light_conditions'].unique())
Weather_conditions=st.selectbox("Select the condition of weather",df['Weather_conditions'].unique())
Type_of_collision=st.selectbox("Select the type of collision",df['Type_of_collision'].unique())
Number_of_vehicles_involved = st.number_input("How many vehicles are involved in accident")
Number_of_casualties = st.number_input("How many casualties happened there")
Vehicle_movement=st.selectbox("Select the vehicle movement of that time",df['Vehicle_movement'].unique())
Casualty_class=st.selectbox("Select the casuality class",df['Casualty_class'].unique())
Sex_of_casualty=st.selectbox("Select the sex of casualty",df['Sex_of_casualty'].unique())
Age_band_of_casualty=st.selectbox("Select age band of casualty",df['Age_band_of_casualty'].unique())
Casualty_severity=st.selectbox("Select the casualty severity",df['Casualty_severity'].unique())
Work_of_casuality=st.selectbox("Select the work of casuality",df['Work_of_casuality'].unique())
Pedestrian_movement=st.selectbox("Select the movement of Pedestrian_movement",df['Pedestrian_movement'].unique())
Cause_of_accident = st.selectbox("Select a cause of accident", df['Cause_of_accident'].unique())
def predict():
    query = pd.DataFrame({
        'Time': [Time],
        'Day_of_week': [Day_of_week],
        'Age_band_of_driver': [Age_band_of_driver],
        'Sex_of_driver': [Sex_of_driver],
        'Educational_level':[Educational_level],
        'Vehicle_driver_relation':[Vehicle_driver_relation],
        'Driving_experience': [Driving_experience],
        'Type_of_vehicle':[Type_of_vehicle],
        'Owner_of_vehicle':[Owner_of_vehicle],
        'Service_year_of_vehicle':[Service_year_of_vehicle],
        'Area_accident_occured': [Area_accident_occured],
        'Lanes_or_Medians':[Lanes_or_Medians],
        'Road_allignment':[Road_allignment],
        'Types_of_Junction': [Types_of_Junction],
        'Road_surface_type':[Road_surface_type],
        'Road_surface_conditions':[Road_surface_conditions],
        'Light_conditions': [Light_conditions],
        'Weather_conditions':[Weather_conditions],
        'Type_of_collision':[Type_of_collision],
        'Number_of_vehicles_involved': [Number_of_vehicles_involved],
        'Number_of_casualties': [Number_of_casualties],
        'Vehicle_movement':[Vehicle_movement],
        'Casualty_class':[Casualty_class],
        'Sex_of_casualty':[Sex_of_casualty],
        'Age_band_of_casualty':[Age_band_of_casualty],
        'Casualty_severity':[Casualty_severity],
        'Work_of_casuality':[Work_of_casuality],
        'Pedestrian_movement':[Pedestrian_movement],
        'Cause_of_accident': [Cause_of_accident],
        
    })
    categorical_columns = query.select_dtypes('object')
    encoded_data = encoder.transform(categorical_columns)
    encoded_data = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
    numerical_columns = ['Number_of_vehicles_involved', 'Number_of_casualties']
    combined_data = pd.concat([encoded_data,query[numerical_columns]], axis=1)
    prediction_of_accident = model.predict(combined_data)
    predicted_class = np.argmax(prediction_of_accident)
    return predicted_class

if st.button("Predict"):
    predicted = predict()

    # Map predicted class to injury level
    injury_levels = {0: 'Fatal Injury', 1: 'Serious Injury', 2: 'Light Injury'}
    result = injury_levels.get(predicted, 'Unknown Injury Level')

    st.write("Unfortunately, but I think this will be:", result)
