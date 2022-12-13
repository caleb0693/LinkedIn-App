import pandas as pd
import numpy as np
import altair as alt
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

s = pd.read_csv('/Users/calebginorio/Desktop/social_media_usage.csv')
s.head(5)
def clean_sm (x):
    x= np.where(x==1,1,0)
    return x


ss = pd.DataFrame({
    "sm_li": clean_sm(s['web1h']),
    "income":np.where(s["income"] > 9, np.nan, s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent":np.where(s["par"] ==1,1,0),
    "married":np.where(s["marital"] == 1,1,0),
    "female":np.where(s["gender"] == 2, 1, 0),
    "age": np.where (s["age"]>98, np.nan, s["age"])})

ss = ss.dropna()
ss = ss.astype(int)
ss.head()

y = ss["sm_li"]
X = ss[["income", "education", "parent", "married","female","age"]]


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,
                                                    test_size=0.2,  
                                                    random_state=987)

lr = LogisticRegression(class_weight = 'balanced')
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)




import streamlit as st

st.title('Are you a LinkedIn User?')
st.text('Caleb Ginorio Final Project')


income = st.selectbox("Select Your Income Level", 
             options = ["Less than $10,000",
                        "Between $10,000 and $20,000",
                        "Between $20,000 and $30,000",
                        "Between $30,000 and $40,000",
                        "Between $40,000 and $50,000",
                        "Between $50,000 and $75,000",
                        "Between $75,000 and $100,000",
                        "Between $100,000 and $150,000",
                        "$150,000 or more?"
                        ])

if income == "Less than $10,000":
    income = 1
elif income == "Between $10,000 and $20,000":
    income = 2
elif income == "Between $20,000 and $30,000":
    income = 3
elif income == "Between $30,000 and $40,000":
    income = 4
elif income == "Between $40,000 and $50,000":
    income = 5
elif income == "Between $50,000 and $75,000":
    income = 6
elif income == "Between $75,000 and $100,000":
    income = 7
elif income == "Between $100,000 and $150,000":
    income = 8
else: 
    income = 9




educ = st.selectbox("Education Level", 
             options = ["Less than High School Diploma",
                        "High School - Incomplete",
                        "High School - Graduate",
                        "Some College - No Degree",
                        "Two-Year Associate's Degree",
                        "Four Year Bachelors Degree",
                        "Some Post-Graduate or Professional Schooling  No Degree",
                        "Post-Graduate or Professional Degree"
                        ])


if educ == "Less than High School Diploma":
    educ = 1
elif educ == "High School - Incomplete":
    educ = 2
elif educ == "High School - Graduate":
    educ = 3
elif educ == "Some College - No Degree":
    educ = 4
elif educ == "Two-Year Associate's Degree":
    educ = 5
elif educ == "Four Year Bachelors Degree":
    educ = 6
elif educ == "Some Post-Graduate or Professional Schooling  No Degree":
    educ = 7
else: 
    educ = 8


parent = st.selectbox("Are you a parent?",
            options= ["Yes",
                      "No",
                        ])


if parent == "Yes":
    parent = 1
else:
    parent = 0 


married = st.selectbox("Are you married?",
            options= ["Yes",
                      "No",
                        ])

if married == "Yes":
    married = 1
else:
    married = 0 



gender = st.selectbox("Are you a male or a female?",
            options= ["Male",
                      "Female",
                        ])

if gender == "Female":
    gender = 1
else:
    gender = 0 



age = st.number_input('Please enter your age',
                min_value= 1,
                max_value= 99,
                value=30)


prediction = pd.DataFrame({
    "income":[income],
    "education":[educ],
    "parent":[parent], 
    "married": [married],
    "gender": [gender],
    "age":[age]
})


prob = round( lr.predict_proba(prediction)[0,1] * 100 , 2 )

outcome = lr.predict(prediction)


st.markdown(f"Probability {prob}%")

if prob >= 50:

    label = "A LinkedIn User"

else :

    label = "Not a LinkedIn User"

st.text(f"You are {label}" )
    
