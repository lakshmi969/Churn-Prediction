from flask import Flask, render_template, request
from sklearn.ensemble import GradientBoostingClassifier


import pickle

app = Flask(__name__)

# Load the trained model
with open('churn_prediction', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # If the form is submitted using POST, process the data
        features = [int(request.form['gender']),
         int(request.form['senior_citizen']),
        int(request.form['partner']),
         int(request.form['dependents']),
        int(request.form['tenure']),
        int(request.form['phone_service']),
        int(request.form['multiple_lines']),
        int(request.form['internet_service']),
        float(request.form['monthly_charges']),
        float(request.form['total_charges'])]

        # Make a prediction using the loaded model
        prediction = model.predict([features])[0]

        return render_template('result.html', prediction=prediction)

    # If the request is a GET request or the form hasn't been submitted yet
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)








