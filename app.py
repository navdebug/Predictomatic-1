from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('PCOS.pkl', 'rb')) 


@app.route('/')
def home():
    return render_template('PCOS.html')


@app.route('/predict', methods = ['GET', 'POST'])
def predict():

    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    if output == 0:
        return render_template('PCOS.html', prediction_text= 'PCOS : No')
    else:
        return render_template('PCOS.html', prediction_text= 'PCOS : Yes') 


if __name__ == "__main__":
    app.run(debug=True) 
