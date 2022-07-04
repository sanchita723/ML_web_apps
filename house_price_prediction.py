import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image


pickle_in = open('model_pickle.pkl','rb')
classifier = pickle.load(pickle_in)

def welcome():
    return 'WELCOME ALL'


def predict_price(location, sqft, bath, bhk):

    X = np.zeros(244)
    X[0] = sqft
    X[1] = bath
    X[2] = bhk
    #if loc_index >= 0:
        #X[loc_index] = 1

    return np.round(classifier.predict([X])[0],3)

def main():
    home = pd.read_csv('C:/Users/Lenovo/Documents/bengaluru_house_clean_data.csv')
    loc = home['location'].unique()
    bath = home['bath'].unique()
    bhk = home['bhk'].unique()
    st.title('Bengaluru House Price Prediction')
    img = Image.open("C:/Users/Lenovo/AppData/Roaming/Microsoft/Windows/Network Shortcuts/maxresdefault.jpg")
    st.image(img)
    st.header("Streamit House Prediction ML App")
    st.subheader("Please enter the required details:")
    location = st.selectbox("Select location", loc)
    sqft = st.text_input("Write sq-ft area","")
    bath_no = st.selectbox("Select no of bathrooms", bath)
    bhk_no = st.selectbox("Select no of bedrooms", bhk)

    result = ""
    if st.button("House price in lakhs"):
        result = predict_price(location,sqft,bath_no, bhk_no)
    st.success("The final price in INR {}/- ".format(result))



if __name__ == '__main__':
    main()






