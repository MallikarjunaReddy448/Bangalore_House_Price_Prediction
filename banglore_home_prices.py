import pandas as pd 
import numpy as np 
from pickle import load
import streamlit as st
from sklearn.linear_model import LinearRegression


model = load(open("banglore_home_prices_model.pickle",'rb'))

df = pd.read_csv('columns.csv')

def predict_price(location,sqft,bath,bhk):
    loc_indeX = np.where(df.columns == location)[0][0]
    
    X = np.zeros(len(df.columns))
    X[0] = sqft
    X[1] = bath
    X[2] = bhk
    if loc_indeX >= 0:
        X[loc_indeX] = 1

    return model.predict([X])[0]

def main():

    st.set_page_config(
        page_title= "Bangalore House Price Predictor"
    )
    
    st.title('Bangalore House Price Prediction')
    
    menu = ['About','Check House Price','Contact']
    choice = st.sidebar.selectbox('Select option',menu)
    if choice == 'Check House Price':
        

        loc = st.selectbox(label='select location',options=df.columns[3:])

        sqft = st.number_input('Insert Area(Square Foot)')
    
        bath = st.number_input('Insert Number of Bathrooms')

        bhk = st.number_input('Insert Number of BHK')

        price = ''

        load = st.button('Get Price ')
        if "load_state" not in st.session_state :
         st.session_state.load_state = False

        if load or st.session_state.load_state:
            st.session_state.load_state= True

            price = predict_price(loc, sqft, bath, bhk)

            st.write(price,'Indian Lack Rupees')


    elif choice == 'About':

        st.warning('')
        st.write('''Welcome to Bangalore House price predictor.
                    This web API will predict the house price
                    based on the values(Area of the House (sqft),number of Baths and number of BHKs) and location(area in bangalore)
                    inserted by the user.''')
        st.warning('')
        st.write('''This API mainly runs based on the pre trained Machine Learning Model. The model was trained with the same parameters or values or options 
                    available for the user in the Check house price prediction section. 
                    And the output or price predicted in Indian Lakh Rupees.''')
        
        st.write('''Thank You.....!🙂''')

        # st.warning('Please select Check House price in the sidebar to get house price')

        st.warning('')

    elif choice == 'Contact':
        '''For your valuable suggestions please contact :'''
        '''Email ID : mallikarjunreddy448@gmail.com'''
        '''Linked In : https://www.linkedin.com/in/mallikarjuna-reddy-841190246/'''
        '''Git Hub : https://github.com/MallikarjunaReddy448'''

if __name__ == '__main__':
    main()
