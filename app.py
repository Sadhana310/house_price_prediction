import streamlit as st
import pickle
import json
import numpy as np
import os

# Check if model files exist
model_file = 'house_price_model.pkl'
columns_file = 'columns.json'

if not os.path.exists(model_file) or not os.path.exists(columns_file):
    st.error("Model files not found! Please run the model training script first.")
    st.stop()

try:
    # Load model and columns
    model = pickle.load(open(model_file, 'rb'))
    with open(columns_file, "r") as f:
        data_columns = json.load(f)['data_columns']

    locations = [col for col in data_columns if col not in ['total_sqft', 'bath', 'bhk']]

    # UI
    st.title("Welcome To House Price Predictor")

    location = st.selectbox("Select the Location:", locations)
    bhk = st.text_input("Enter BHK:", "2")
    bath = st.text_input("Enter Number of Bathrooms:", "2")
    sqft = st.text_input("Enter Square Feet:", "1000")

    if st.button("Predict Price"):
        try:
            bhk = int(bhk)
            bath = int(bath)
            sqft = float(sqft)

            # Input validation
            if bhk <= 0 or bath <= 0 or sqft <= 0:
                st.error("Please enter positive values for BHK, bathrooms, and square feet.")
            elif bhk > 20 or bath > 20 or sqft > 100000:
                st.error("Please enter reasonable values (BHK/bathrooms ≤ 20, square feet ≤ 100,000).")
            else:
                x = np.zeros(len(data_columns))
                x[0] = sqft
                x[1] = bath
                x[2] = bhk
                if location in data_columns:
                    loc_index = data_columns.index(location)
                    x[loc_index] = 1

                predicted_price = model.predict([x])[0]
                st.subheader(f"Prediction: ₹ {predicted_price:,.2f}")
        except ValueError:
            st.error("Please enter valid numeric values for BHK, bathrooms, and square feet.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

except Exception as e:
    st.error(f"Error loading model files: {str(e)}")
    st.stop()