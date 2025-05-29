from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    # Only show the top 20 features in the form
    return render_template('index.html', features=model.feature_names_)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Only process the 20 expected features
        input_features = [float(request.form[feature]) for feature in model.feature_names_]
        prediction = model.predict(np.array(input_features).reshape(1, -1))[0]
        return render_template('index.html',
                            prediction_text="Churn" if prediction == 1 else "No Churn",
                            features=model.feature_names_,
                            input_values=input_features)
    except Exception as e:
        return f"Error: {str(e)}", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050)