# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 13:19:49 2022

@author: Mayank Rawat
"""
import numpy as np
import pickle 
import streamlit as st


#loading the saved model
loaded_model = pickle.load(open('trained_model.sav','rb'))


# creating a function for prediction

def heart_disease_prediction(input_data):
    
    # change the input data to numpy array
    input_data_numpy = np.asarray(input_data)
    
    data = np.array(input_data_numpy, dtype=int) #  convert using numpy

    # Reshape the numpy array as we are pridicting for a instance 
    input_data_reshaped = data.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    print(prediction)

    if(prediction[0] == 1):
      return 'The person is having the Heart Disease'
    else:
      return 'The person does not have any Heart Disease'



def main():
    
    # Giving a title
    st.title('Heart Disease prediction Web App using ML')
    
    st.markdown("---")
    
    # Getting the input data from the user
    #columns for input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        trestbps = st.text_input('Resting Blood Pressure')
        restecg = st.text_input('Resting electrocardiographic results (values 0,1,2)')
        oldpeak = st.text_input('ST depression induced by exercise relative to rest')
        thal = st.text_input('Thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
    with col2:
        sex = st.text_input('Sex (1 = Male, 0 = Female)')
        chol = st.text_input('Serum cholestoral in mg/dl')
        thalach = st.text_input('Maximum heart rate achieved')
        slope = st.text_input('The slope of the peak exercise ST segment')
        
        
    with col3:
        cp = st.text_input('Chest Pain Type (0,1,2,3)')
        fbs = st.text_input('Fasting blood sugar > 120 mg/dl')
        exang = st.text_input('Exercise induced angina')
        ca = st.text_input('Number of major vessels (0-3) colored by flourosopy')
    
    
    
    #code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        diagnosis = heart_disease_prediction([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
    
    
    st.success(diagnosis)
        



if __name__ == '__main__':
    main()

  
    
    
    
    
    

