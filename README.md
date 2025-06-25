# House Price Prediction App

A Streamlit web application for predicting house prices in Bangalore based on location, BHK, bathrooms, and square footage.

## Features

- Interactive web interface using Streamlit
- Machine learning model trained on Bangalore house data
- Input validation and error handling
- Real-time price predictions

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **First, train the model:**
```bash
python model_training.py
```
This will:
- Create a sample dataset if `Bangalore_House_Data.csv` is not found
- Train a linear regression model
- Save the model as `house_price_model.pkl`
- Save column information as `columns.json`

2. **Run the Streamlit app:**
```bash
streamlit run app.py
```

3. **Use the application:**
- Select a location from the dropdown
- Enter BHK (Bedroom, Hall, Kitchen) count
- Enter number of bathrooms
- Enter total square footage
- Click "Predict Price" to get the estimated price

## Input Validation

The app includes validation for:
- Positive values only
- Reasonable ranges (BHK/bathrooms ≤ 20, square feet ≤ 100,000)
- Numeric input validation

## Error Handling

The application handles various error scenarios:
- Missing model files
- Invalid input values
- File loading errors
- Prediction errors

## Model Information

- **Algorithm**: Linear Regression
- **Features**: Location (one-hot encoded), BHK, Bathrooms, Square Feet
- **Target**: House price in Indian Rupees 