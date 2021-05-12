import numpy
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('flightprediction.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    print(int_features)
    final_features = numpy.array(int_features).reshape(1,3)
    prediction =int(model.predict(final_features))
    
    output = prediction

    return render_template('home.html', prediction_text='Flight Fare should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)