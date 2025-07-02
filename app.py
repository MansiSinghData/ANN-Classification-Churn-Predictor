import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import numpy as np
import streamlit as st
import pickle


model=tf.keras.models.load_model('model.h5')

with open('label_encoder.pkl','rb') as file:
    label_encoder=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

with open('one_hot_encoder_geo.pkl','rb') as file:
    one_hot_encoder_geo=pickle.load(file)

#Streamlit App

st.title("Customer Churn Prediction")

#User Input

geography=st.selectbox('Geography',one_hot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])


input_data=pd.DataFrame([{
'CreditScore':credit_score,
'Gender':label_encoder.transform([gender])[0],
'Age':age,
'Tenure':tenure,
'Balance':balance,
'NumOfProducts':num_of_products,
'HasCrCard':has_cr_card,
'IsActiveMember':is_active_member,
'EstimatedSalary':estimated_salary

}])

#geo_encoded=one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded = one_hot_encoder_geo.transform(pd.DataFrame([[geography]], columns=['Geography'])
).toarray()

geo_encoded_df=pd.DataFrame(geo_encoded,columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

input_df=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_df=scaler.transform(input_df)


#Predict the Churn

prediction=model.predict(input_df)
st.write(f'Churn Probability: {prediction[0][0]:.2f}')
if prediction[0][0] >0.5:
    st.write('The cutomer is likely to Churn')
else:
    st.write('The customer is not likely to Churn')   
