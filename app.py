from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ================= LOAD DATA =================

# Make sure these CSV files are in same folder
fraud_df = pd.read_csv("fraud_values.csv")
valid_df = pd.read_csv("valid_values.csv")

# Combine datasets
data = pd.concat([fraud_df, valid_df], ignore_index=True)

# ================= PREPROCESS =================

# Target column must be 'class'
X = data.drop('class', axis=1)
y = data['class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= MODEL =================

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Model accuracy:", accuracy)

# ================= ROUTES =================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/inform')
def inform():
    return render_template('inform.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        me = request.form['message']

        # Convert input string to float list
        message = [float(x) for x in me.split()]
        input_data = np.array(message).reshape(1, -1)

        # Prediction
        prediction = model.predict(input_data)[0]

        result = "Fraud Transaction 🚨" if prediction == 1 else "Valid Transaction ✅"

        return render_template(
            'result.html',
            prediction=result,
            accuracy=round(accuracy * 100, 2)
        )

    except Exception as e:
        return f"Error: {str(e)}"

# ================= RUN =================

if __name__ == '__main__':
    app.run(debug=True)