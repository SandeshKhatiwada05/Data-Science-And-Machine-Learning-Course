from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained pipeline and encoders
MODEL_PATH = r"D:\Github\Data-Science-And-Machine-Learning-Course\Final Project of Data Science\depression_pipeline.pkl"
ENCODERS_PATH = r"D:\Github\Data-Science-And-Machine-Learning-Course\Final Project of Data Science\encoders.pkl"

pipeline = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)

def encode_input(data):
    """Apply label encoding to categorical features"""
    encoded_data = data.copy()
    
    categorical_columns = ['Gender', 'City', 'Profession', 'Sleep Duration', 
                          'Dietary Habits', 'Degree', 
                          'Have you ever had suicidal thoughts ? ',
                          'Family History of Mental Illness']
    
    for col in categorical_columns: 
        if col in encoded_data: 
            try:
                encoded_data[col] = encoders[col].transform([encoded_data[col]])[0]
            except ValueError: 
                encoded_data[col] = 0
    
    return encoded_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_form', methods=['POST'])
def predict_form():
    try:
        # Get form data
        data = {
            'Gender': request.form['gender'],
            'Age': int(request.form['age']),
            'City': request.form['city'],
            'Profession': request.form['profession'],
            'Academic Pressure': int(request.form['academic_pressure']),
            'Work Pressure':  int(request.form['work_pressure']),
            'CGPA': float(request.form['cgpa']),
            'Study Satisfaction': int(request.form['study_satisfaction']),
            'Job Satisfaction': int(request.form['job_satisfaction']),
            'Sleep Duration': request. form['sleep_duration'],
            'Dietary Habits': request. form['dietary_habits'],
            'Degree': request.form['degree'],
            'Have you ever had suicidal thoughts ? ':  request.form['suicidal_thoughts'],
            'Work/Study Hours': int(request. form['work_study_hours']),
            'Financial Stress': int(request.form['financial_stress']),
            'Family History of Mental Illness':  request.form['family_history']
        }
        
        # Encode and predict
        encoded_data = encode_input(data)
        input_df = pd.DataFrame([encoded_data])
        input_df.columns = input_df.columns.str.strip()
        
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]
        
        # Render results page
        return render_template('result.html',
                             prediction=int(prediction),
                             probability=round(probability * 100, 2),
                             user_data=data)
    
    except Exception as e:
        return f"Error: {str(e)}", 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        encoded_data = encode_input(data)
        input_df = pd. DataFrame([encoded_data])
        input_df.columns = input_df.columns.str.strip()
        
        prediction = pipeline. predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]
        
        return jsonify({
            'depression':  int(prediction),
            'probability': round(float(probability), 4),
            'message': 'Depression detected' if prediction == 1 else 'No depression detected'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/test', methods=['GET'])
def test():
    sample_data = {
        'Gender':  'Male',
        'Age': 25,
        'City': 'Pune',
        'Profession': 'Student',
        'Academic Pressure': 3,
        'Work Pressure': 0,
        'CGPA': 7.5,
        'Study Satisfaction': 3,
        'Job Satisfaction': 0,
        'Sleep Duration': '5-6 hours',
        'Dietary Habits': 'Moderate',
        'Degree': 'BSc',
        'Have you ever had suicidal thoughts ? ': 'No',
        'Work/Study Hours': 6,
        'Financial Stress': 2,
        'Family History of Mental Illness': 'No'
    }
    
    encoded_data = encode_input(sample_data)
    input_df = pd.DataFrame([encoded_data])
    input_df.columns = input_df.columns.str.strip()
    
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]
    
    return jsonify({
        'test_input': sample_data,
        'depression': int(prediction),
        'probability': round(float(probability), 4)
    })

@app.route('/debug', methods=['GET'])
def debug():
    return jsonify({
        'encoder_keys': list(encoders.keys())
    })

if __name__ == '__main__':
    print("Loading model and encoders...")
    print(f"Model loaded from:  {MODEL_PATH}")
    print(f"Encoders loaded from: {ENCODERS_PATH}")
    app.run(host='0.0.0.0', port=5000, debug=True)