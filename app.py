# app.py (updated to handle unseen categories)
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model, scaler, label encoders, and numerical columns
model = joblib.load('models/heart_disease_model.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')
numerical_cols = joblib.load('models/numerical_cols.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = float(request.form['age'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = request.form['restecg']
        thalch = float(request.form['thalch'])
        exang = request.form['exang'] == 'True'
        oldpeak = float(request.form['oldpeak'])
        slope = request.form['slope']
        ca = float(request.form['ca'])
        sex = request.form['sex']
        cp = request.form['cp']
        
        # Convert categorical variables to numerical using the label encoders
        # Handle unseen categories by mapping to a default value
        try:
            restecg_encoded = label_encoders['restecg'].transform([restecg])[0]
        except ValueError:
            # If category not seen during training, use the most common class
            restecg_encoded = 0  # Default to first category
        
        # Convert slope to numerical (manual encoding based on dataset)
        slope_mapping = {'flat': 0, 'upsloping': 1, 'downsloping': 2}
        slope_encoded = slope_mapping.get(slope, 0)
        
        # Convert sex to numerical
        sex_Female = 1 if sex == 'Female' else 0
        sex_Male = 1 if sex == 'Male' else 0
        
        # Convert cp to numerical (one-hot encoding)
        cp_mapping = {
            'asymptomatic': [1, 0, 0, 0],
            'atypical angina': [0, 1, 0, 0],
            'non-anginal': [0, 0, 1, 0],
            'typical angina': [0, 0, 0, 1]
        }
        cp_encoded = cp_mapping.get(cp, [0, 0, 0, 0])
        
        # Create a DataFrame with all features
        input_df = pd.DataFrame([[
            age, trestbps, chol, fbs, restecg_encoded, thalch, 
            int(exang), oldpeak, slope_encoded, ca,
            sex_Female, sex_Male,
            cp_encoded[0], cp_encoded[1], cp_encoded[2], cp_encoded[3]
        ]], columns=[
            'age', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 
            'exang', 'oldpeak', 'slope', 'ca',
            'sex_Female', 'sex_Male',
            'cp_asymptomatic', 'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina'
        ])
        
        # Scale only the numerical features
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        
        # Make prediction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
        # Map prediction to readable result using the thal label encoder
        thal_classes = label_encoders['thal'].classes_
        result = thal_classes[prediction[0]]
        
        return render_template('result.html', 
                             prediction=result,
                             probability=max(probability[0]) * 100)
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return render_template('result.html', 
                             prediction=f"Error: {str(e)}",
                             probability=0)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        # Extract values from JSON
        age = float(data['age'])
        trestbps = float(data['trestbps'])
        chol = float(data['chol'])
        fbs = int(data['fbs'])
        restecg = data['restecg']
        thalch = float(data['thalch'])
        exang = data['exang']
        oldpeak = float(data['oldpeak'])
        slope = data['slope']
        ca = float(data['ca'])
        sex = data['sex']
        cp = data['cp']
        
        # Convert categorical variables to numerical using the label encoders
        # Handle unseen categories by mapping to a default value
        try:
            restecg_encoded = label_encoders['restecg'].transform([restecg])[0]
        except ValueError:
            # If category not seen during training, use the most common class
            restecg_encoded = 0  # Default to first category
        
        # Convert slope to numerical
        slope_mapping = {'flat': 0, 'upsloping': 1, 'downsloping': 2}
        slope_encoded = slope_mapping.get(slope, 0)
        
        # Convert sex to numerical
        sex_Female = 1 if sex == 'Female' else 0
        sex_Male = 1 if sex == 'Male' else 0
        
        # Convert cp to numerical (one-hot encoding)
        cp_mapping = {
            'asymptomatic': [1, 0, 0, 0],
            'atypical angina': [0, 1, 0, 0],
            'non-anginal': [0, 0, 1, 0],
            'typical angina': [0, 0, 0, 1]
        }
        cp_encoded = cp_mapping.get(cp, [0, 0, 0, 0])
        
        # Create a DataFrame with all features
        input_df = pd.DataFrame([[
            age, trestbps, chol, fbs, restecg_encoded, thalch, 
            int(exang), oldpeak, slope_encoded, ca,
            sex_Female, sex_Male,
            cp_encoded[0], cp_encoded[1], cp_encoded[2], cp_encoded[3]
        ]], columns=[
            'age', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 
            'exang', 'oldpeak', 'slope', 'ca',
            'sex_Female', 'sex_Male',
            'cp_asymptomatic', 'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina'
        ])
        
        # Scale only the numerical features
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        
        # Make prediction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
        # Map prediction to readable result
        thal_classes = label_encoders['thal'].classes_
        result = thal_classes[prediction[0]]
        
        return jsonify({
            'prediction': result,
            'probability': float(max(probability[0]) * 100),
            'status': 'success'
        })
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'status': 'error'
        })

if __name__ == '__main__':
    app.run(debug=True)