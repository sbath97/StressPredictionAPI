from flask import Flask, jsonify, request
import pickle
import numpy as np
import sklearn
import pandas as pd

import joblib



#model = pickle.load(open('svm_model.pkl','rb'))
model = joblib.load('svm_model.pkl')
#model = joblib.load('RandomForest_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():
    humidity = request.form.get ('humidity')
    temperature = request.form.get('temperature')
    step_count = request.form.get('step_count')
    #stress_level = request.form.get('stress_level')
    respiratery_rate = request.form.get('respiratery_rate')
    heart_rate = request.form.get('heart_rate')
    input_query = np.array([[humidity, temperature, step_count, respiratery_rate, heart_rate]])
    print(input_query)
    #input_query = [[humidity, temperature, step_count, respiratery_rate, heart_rate]]
    #df1 = pd.DataFrame(input_query)
    result = model.predict(input_query)

  
    #result = {'humidity':humidity, 'temperature':temperature, 'step_count':step_count, 'stress_level':stress_level, 'respiratery_rate':respiratery_rate, 'heart_rate':heart_rate}
    print(result[0])
    #return jsonify(str(result))
    return jsonify({'stress_level':str(result)})



if __name__ == '__main__':
    app.run(debug=True)


