from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and encoders
model = joblib.load('model.joblib')
encoders = joblib.load('encoders.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']

        if file:
            df = pd.read_csv(file)

            # Apply encoders
            for col in df.columns:
                if col in encoders:
                    df[col] = encoders[col].transform(df[col])

            # Prediction
            predictions = model.predict(df)

            return render_template(
                'index.html',
                prediction_text=f"Predictions (first 5): {predictions[:5]}"
            )

        return render_template('index.html', prediction_text="No file uploaded")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)