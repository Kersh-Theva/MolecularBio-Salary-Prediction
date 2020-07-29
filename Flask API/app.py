import pickle 
from flask import Flask, jsonify, request
import json
import numpy as np 
from dataInput import *

#load the pickled model 
def load_models():
    file_name = "models/model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model

app = Flask(__name__)
@app.route('/predict', methods=['GET'])

#get the expected prediction
def predict():

    # parse input features from request
    request_json = request.get_json()
    
    #Make jobDescriptor object from json
    rawData = jobDescriptor(request_json['input'])

    #Convert it into a proper array
    data_in = np.array(rawData.transformDict())[0].reshape(1,-1)

    #Assuming userData coming from dataInput.py    
    #headers = {'address': "application/json"}
    #r = requests.get(URL, headers=headers, json=data)


	#load model
    model = load_models()
    prediction = model.predict(data_in)[0]

    response = json.dumps({'response': prediction})
    return response, 200

if __name__ == '__main__':
    application.run(debug=True)

#kill -9 2526
#lsof -i TCP:8080 | grep LISTEN