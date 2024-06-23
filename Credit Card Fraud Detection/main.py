import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st 

# load data
credited_card_df = pd.read_csv('creditcard.csv')

# seperate lgitimate and fraudulent transactions
legit = credited_card_df[credited_card_df.Class==0]
fraud = credited_card_df[credited_card_df.Class==1]

# undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=492)
credited_card_df = pd.concat([legit_sample,fraud],axis=0)

#split data into training and testing sets
x = credited_card_df.drop('Class',axis=1)
y=credited_card_df['Class']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y, random_state=2)

# train logistic regression model
model=LogisticRegression()
model.fit(x_train,y_train)

# evaluate model performance
train_acc=accuracy_score(model.predict(x_train),y_train)
test_acc=accuracy_score(model.predict(x_test),y_test)

# web app
st.title("Credit Card Detection")
input_df=st.text_input("Enter All Required Features Values")
input_df_splited=input_df.split(',')

submit = st.button("Submit")

if submit:
    features = np.asarray(input_df_splited,dtype=np.float64)
    prediction = model.predict(features.reshape(1,-1))
    
    if prediction[0]==0:
        st.write("Legitimate Transaction")
    else:
        st.write("Fradulant Transaction")
