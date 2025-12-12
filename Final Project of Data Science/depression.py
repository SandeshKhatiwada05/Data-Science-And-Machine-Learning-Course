from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained pipeline
MODEL_PATH = r"D:\Github\Data-Science-And-Machine-Learning-Course\Final Project of Data Science\depression_pipeline. pkl"
pipeline = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return "Depression Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.json
        
        # Convert to DataFrame (single row)
        input_df = pd.DataFrame([data])
        
        # Make prediction
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]  # Probability of depression
        
        # Return result
        return jsonify({
            'depression':  int(prediction),
            'probability':  round(float(probability), 4),
            'message': 'Depression detected' if prediction == 1 else 'No depression detected'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/test', methods=['GET'])
def test():
    # Sample test data based on your dataset structure
    sample_data = {
        'Gender': 'Male',
        'Age':  25,
        'City': 'Kathmandu',
        'Profession': 'Student',
        'Academic Pressure': 3,
        'Work Pressure':  0,
        'CGPA':  7.5,
        'Study Satisfaction':  3,
        'Job Satisfaction':  0,
        'Sleep Duration': '5-6 hours',
        'Dietary Habits': 'Moderate',
        'Degree': 'BSc',
        'Have you ever had suicidal thoughts ? ': 'No',
        'Work/Study Hours': 6,
        'Financial Stress':  2,
        'Family History of Mental Illness': 'No'
    }
    
    input_df = pd.DataFrame([sample_data])
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]
    
    return jsonify({
        'test_input': sample_data,
        'depression':  int(prediction),
        'probability': round(float(probability), 4)
    })

if __name__ == '__main__':
    print("Loading model...")
    print(f"Model loaded from: {MODEL_PATH}")
    app.run(host='0.0.0.0', port=5000, debug=True)