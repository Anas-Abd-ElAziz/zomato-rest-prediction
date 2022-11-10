import streamlit as st
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv("zomato_cleaned.csv")
model = joblib.load('joblib/model.h5')
scaler = joblib.load('joblib/scaler.h5')
City_ENC = joblib.load('joblib/City_ENC.h5')
Location_ENC = joblib.load('joblib/Location_ENC.h5')
Type_ENC = joblib.load('joblib/Type_ENC.h5')
inp_data = []
result = ''

text = '### <center><p style="font-family:Verdana;color:DarkSalmon; font-size: 30px;">Choose from the following features To Predict if your resturant will succeed or not in city of💮Bangalore💮</p></center>'
st.markdown(text,unsafe_allow_html=True)
st.markdown('______________________________________')
price = st.text_input('- Enter an approximate price for two', '')
location = st.multiselect('- Location of your resturant',df['location'].unique(),max_selections=1)
city = st.multiselect('- the city that the resturant will be listed in', df['listed_in(city)'].unique(), max_selections=1)
Type = st.multiselect('- The type/s that the resturant will be listed in', df['listed_in(type)'].unique())
rest_type = st.multiselect('- The type of your resturant', list(df.columns)[7:33])
cuisines = st.multiselect('- Cuisines that your resturant will offer', list(df.columns)[33:])
table = 1 if st.checkbox(' Will your restaurant support table booking?', False) else 0
online = 1 if st.checkbox(' Will your restaurant support online ordering?', False) else 0
if st.button(' Predict '):
    inp_data.append(online)
    inp_data.append(table)
    inp_data.append(int(Location_ENC.transform([location[0]])[0]))
    inp_data.append(int(Type_ENC.transform([Type[0]])[0]))
    inp_data.append(int(City_ENC.transform([city[0]])[0]))
    inp_data.append(int(price))
    inp_data.append(len(cuisines))
    inp_data.extend([1 if x in rest_type else 0 for x in list(df.columns)[7:33]])
    inp_data.extend([1 if x in cuisines else 0 for x in list(df.columns)[33:]])
    result = np.array(inp_data).reshape(1, df.columns.shape[0])
    result = scaler.transform([inp_data])
    result = model.predict(result)[0]
if result == 1:
    st.success('   The resturant will succeed', icon="✅")
elif result == 0:
    st.error('   The resturant will not succeed', icon="❌")