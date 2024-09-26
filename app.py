from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from transformers import pipeline
import pickle
import joblib
app = Flask(__name__)

# Load pre-trained sentiment analysis pipeline
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)

# Load pre-trained regression model (Assuming you saved it previously using pickle)
with open("models_saved/mlp_regressor_model.pkl", "rb") as f:
    regression_model = joblib.load(f)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Preprocessing function for the input data
def preprocess_data(data):
    # Check if data is a list or a dictionary
    if isinstance(data, dict):
        df = pd.DataFrame([data])  # Single row
    elif isinstance(data, list):
        df = pd.DataFrame(data)  # Multiple rows
    else:
        raise ValueError("Input data must be a dictionary or a list of dictionaries.")

    # Sentiment analysis for 'Product_supervisor_review'
    sentiment_result = sentiment_pipeline(df['Product_supervisor_review'].values[0])[0]
    df['Product_supervisor_review'] = 1 if sentiment_result['label'] in ['5 stars', '4 stars'] else 0

    # Calculate competitor sales ratio
    df['Competitor_sales_ratio'] = df['Direct_competitor_sales'] / df['Last_month_sales']

    # One-hot encoding of categorical variables
    df_encoded = pd.get_dummies(df, columns=[
        'Product_supervisor_product_recommendation',
        'Product_seniority',
        'Product_competitor_seniority'
    ])

    # Ensure that required columns exist
    expected_cols = [
        'Product_supervisor_product_recommendation_more',
        'Product_supervisor_product_recommendation_same'
    ]
    
    for col in expected_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0  # Fill missing columns with zeros
    
    # Scale numerical columns
    num_columns = ['Last_month_sales', 'Direct_competitor_sales', 'Product_market_position', 'Competitor_sales_ratio']
    scaler = MinMaxScaler()
    df_encoded[num_columns] = scaler.fit_transform(df_encoded[num_columns])
    
    return df_encoded

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    data = {
        'Last_month_sales': int(request.form['last_month_sales']),
        'Product_supervisor_review': request.form['product_supervisor_review'],
        'Product_supervisor_product_recommendation': request.form['product_recommendation'],
        'Product_seniority': request.form['product_seniority'],
        'Direct_competitor_sales': int(request.form['direct_competitor_sales']),
        'Product_market_position': float(request.form['market_position']),
        'Product_competitor_seniority': request.form['competitor_seniority']
    }
    print(data.keys())
    # Preprocess the data
    
    processed_data = preprocess_data(data)
    print(processed_data.columns)
    # Make prediction using the regression model
    prediction = regression_model.predict(processed_data)+data['Last_month_sales']

    # Pass input data and prediction to the result page
    return render_template('result.html', data=data, prediction=int(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
