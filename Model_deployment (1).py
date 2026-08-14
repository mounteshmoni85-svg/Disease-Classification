#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import pickle


# In[2]:


model=pickle.load(open('xgb10.pkl','rb'))


# In[3]:


sca=pickle.load(open('scaler10.pkl','rb'))


# In[4]:


### title
st.title('Alzheimer Disease Classification')


# In[10]:


def user_input_parameters():

    st.sidebar.header("Patient Details")

    # ---------------- Demographics ----------------
    Age = st.sidebar.slider("Age", 40, 100, 65)

    Gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    Gender = 0 if Gender == "Female" else 1

    Ethnicity = st.sidebar.selectbox(
    "Ethnicity",
    ["Caucasian", "African American", "Asian", "Other"]
    )

    Ethnicity = {
        "Caucasian": 0,
        "African American": 1,
        "Asian": 2,
        "Other": 3
        }[Ethnicity]

    EducationLevel = st.sidebar.selectbox(
    "Education Level",
    ["No Education", "High School", "Bachelor's", "Higher"]
    )

    EducationLevel = {
        "No Education": 0,
        "High School": 1,
        "Bachelor's": 2,
        "Higher": 3
        }[EducationLevel]

    BMI = st.sidebar.number_input("BMI", 10.0,50.0,25.0)

    # ---------------- Lifestyle ----------------

    Smoking = st.sidebar.selectbox("Smoking", ["No", "Yes"])
    Smoking = 0 if Smoking == "No" else 1

    AlcoholConsumption = st.sidebar.slider("Alcohol Consumption",0.0,20.0,5.0)

    PhysicalActivity = st.sidebar.slider("Physical Activity",0.0,10.0,5.0)

    DietQuality = st.sidebar.slider("Diet Quality",0.0,10.0,5.0)

    SleepQuality = st.sidebar.slider("Sleep Quality",0.0,10.0,5.0)

    # ---------------- Medical History ----------------

    FamilyHistoryAlzheimers = st.sidebar.selectbox("Family History of Alzheimer's", ["No", "Yes"])
    FamilyHistoryAlzheimers = 0 if FamilyHistoryAlzheimers == "No" else 1

    CardiovascularDisease = st.sidebar.selectbox("Cardiovascular Disease", ["No", "Yes"])
    CardiovascularDisease = 0 if CardiovascularDisease == "No" else 1

    Diabetes = st.sidebar.selectbox("Diabetes", ["No", "Yes"])
    Diabetes = 0 if Diabetes == "No" else 1

    Depression = st.sidebar.selectbox("Depression", ["No", "Yes"])
    Depression = 0 if Depression == "No" else 1

    HeadInjury = st.sidebar.selectbox("Head Injury", ["No", "Yes"])
    HeadInjury = 0 if HeadInjury == "No" else 1

    Hypertension = st.sidebar.selectbox("Hypertension", ["No", "Yes"])
    Hypertension = 0 if Hypertension == "No" else 1

    # ---------------- Clinical Measurements ----------------

    SystolicBP = st.sidebar.slider("Systolic BP",90,200,120)

    DiastolicBP = st.sidebar.slider("Diastolic BP",50,130,80)

    CholesterolTotal = st.sidebar.slider("Total Cholesterol",100,400,200)

    CholesterolLDL = st.sidebar.slider("LDL Cholesterol",30,250,100)

    CholesterolHDL = st.sidebar.slider("HDL Cholesterol",20,120,50)

    CholesterolTriglycerides = st.sidebar.slider("Triglycerides",50,400,150)

    # ---------------- Cognitive Assessment ----------------

    MMSE = st.sidebar.slider("MMSE Score",0,30,20)

    FunctionalAssessment = st.sidebar.slider("Functional Assessment",0.0,10.0,5.0)

    ADL = st.sidebar.slider("ADL Score",0.0,10.0,5.0)

    # ---------------- Symptoms ----------------

    MemoryComplaints = st.sidebar.selectbox("Memory Complaints", ["No", "Yes"])
    MemoryComplaints = 0 if MemoryComplaints == "No" else 1

    BehavioralProblems = st.sidebar.selectbox("Behavioral Problems", ["No", "Yes"])
    BehavioralProblems = 0 if BehavioralProblems == "No" else 1

    Confusion = st.sidebar.selectbox("Confusion", ["No", "Yes"])
    Confusion = 0 if Confusion == "No" else 1

    Disorientation = st.sidebar.selectbox("Disorientation", ["No", "Yes"])
    Disorientation = 0 if Disorientation == "No" else 1

    PersonalityChanges = st.sidebar.selectbox("Personality Changes", ["No", "Yes"])
    PersonalityChanges = 0 if PersonalityChanges == "No" else 1

    DifficultyCompletingTasks = st.sidebar.selectbox("Difficulty Completing Tasks", ["No", "Yes"])
    DifficultyCompletingTasks = 0 if DifficultyCompletingTasks == "No" else 1

    Forgetfulness = st.sidebar.selectbox("Forgetfulness", ["No", "Yes"])
    Forgetfulness = 0 if Forgetfulness == "No" else 1

    data = {
        'Age':Age,
        'Gender':Gender,
        'Ethnicity':Ethnicity,
        'EducationLevel':EducationLevel,
        'BMI':BMI,
        'Smoking':Smoking,
        'AlcoholConsumption':AlcoholConsumption,
        'PhysicalActivity':PhysicalActivity,
        'DietQuality':DietQuality,
        'SleepQuality':SleepQuality,
        'FamilyHistoryAlzheimers':FamilyHistoryAlzheimers,
        'CardiovascularDisease':CardiovascularDisease,
        'Diabetes':Diabetes,
        'Depression':Depression,
        'HeadInjury':HeadInjury,
        'Hypertension':Hypertension,
        'SystolicBP':SystolicBP,
        'DiastolicBP':DiastolicBP,
        'CholesterolTotal':CholesterolTotal,
        'CholesterolLDL':CholesterolLDL,
        'CholesterolHDL':CholesterolHDL,
        'CholesterolTriglycerides':CholesterolTriglycerides,
        'MMSE':MMSE,
        'FunctionalAssessment':FunctionalAssessment,
        'MemoryComplaints':MemoryComplaints,
        'BehavioralProblems':BehavioralProblems,
        'ADL':ADL,
        'Confusion':Confusion,
        'Disorientation':Disorientation,
        'PersonalityChanges':PersonalityChanges,
        'DifficultyCompletingTasks':DifficultyCompletingTasks,
        'Forgetfulness':Forgetfulness
    }

    features = pd.DataFrame(data,index=[0])
    return features


df = user_input_parameters()

prediction = model.predict(df)
prediction_prob = model.predict_proba(df)

if st.button("Predict"):

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("Positive for Alzheimer's Disease")

        st.write("""
        **Description:**
        The model predicts that the patient is likely to have Alzheimer's disease based on the entered clinical and cognitive features.

        **Note:** This is only a machine learning prediction and should not be considered a final medical diagnosis. Please consult a neurologist or healthcare professional for further evaluation.
        """)

    else:
        st.success("Negative for Alzheimer's Disease")

        st.write("""
        **Description:**
        The model predicts that the patient is unlikely to have Alzheimer's disease based on the entered clinical and cognitive features.

        **Note:** This result is only a machine learning prediction and should not replace professional medical advice.
        """)

    st.subheader("Prediction Probability")

    st.write(f"Probability of No Alzheimer's Disease: {prediction_prob[0][0]:.2%}")
    st.write(f"Probability of Alzheimer's Disease: {prediction_prob[0][1]:.2%}")


# In[ ]:




