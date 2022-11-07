from flask import Flask
import pandas as pd
import numpy as np
from pathlib import Path
import os

from app.handlers.routes import configure_routes
import json

def test_base_route():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/'

    response = client.get(url)

    assert response.status_code == 200
    assert response.get_data() == b'try the predict route it is great!'

def validate_inputs(url):
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    response = client.get(url, query_string = {"G1": 18, "G2": 15} )
    assert response.status_code == 200
    response_data = json.loads(response.get_data())
    assert (response_data['prediction'] == 0 or response_data['prediction'] == 1)


    failed_response1 = client.get(url, query_string = {"G1": 4, "health": 3})
    assert failed_response1.status_code == 400
    failed_response1_data = json.loads(failed_response1.get_data())
    assert failed_response1_data['message'] == "Invalid query: G1 and G2 arguments are invalid"

    failed_response2 = client.get(url, query_string = {"GOne": 4, "G2": 3})
    assert failed_response2.status_code == 400
    failed_response2_data = json.loads(failed_response2.get_data())
    assert failed_response2_data['message'] == "Invalid query: G1 and G2 arguments are invalid"

    failed_response3 = client.get(url, query_string = {"G1": -1, "G2": 15})
    assert failed_response3.status_code == 400
    failed_response3_data = json.loads(failed_response3.get_data())
    assert failed_response3_data['message'] == "Invalid query: G1 and G2 arguments are invalid"

    failed_response4 = client.get(url, query_string = {"G1": 21, "G2": 15})
    assert failed_response4.status_code == 400
    failed_response4_data = json.loads(failed_response4.get_data())
    assert failed_response4_data['message'] == "Invalid query: G1 and G2 arguments are invalid"

    failed_response5 = client.get(url, query_string = {"G1": 15, "G2": -1})
    assert failed_response5.status_code == 400
    failed_response5_data = json.loads(failed_response5.get_data())
    assert failed_response5_data['message'] == "Invalid query: G1 and G2 arguments are invalid"

    failed_response6= client.get(url, query_string = {"G1": 19, "G2": 21})
    assert failed_response6.status_code == 400
    failed_response6_data = json.loads(failed_response6.get_data())
    assert failed_response6_data['message'] == "Invalid query: G1 and G2 arguments are invalid"

    response7 = client.get(url, query_string = {"G1": 0, "G2": 0, "age": 22})
    assert response7.status_code == 200
    response7_data = json.loads(response.get_data())
    assert (response7_data['prediction'] == 0 or response7_data['prediction'] == 1)


    response8 = client.get(url, query_string = {"G1": 20, "G2": 20})
    assert response8.status_code == 200
    response8_data = json.loads(response.get_data())
    assert (response8_data['prediction'] == 0 or response8_data['prediction'] == 1)


def test_predict_route():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    
    validate_inputs('/predict')
    super_dir = Path.cwd().parent
    data_path = os.path.join(super_dir, 'data', 'student-mat.csv')
    df = pd.read_csv(data_path, sep=';')
    count = 0
    size = df.shape[0]
    for _,row in df.iterrows():
        response = client.get('/predict', query_string = {"G1": row["G1"], 
                                                          "G2": row["G2"] })
        actual_pred = (row["G3"] > 15)
        pred = json.loads(response.get_data())
        pred = pred['prediction']
        if (pred == 1 and actual_pred) or (pred == 0 and not actual_pred):
            count+=1

    model_accuracy = count/size

    assert model_accuracy > .80

def test_predict_more_route():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/predict/more'

    response = client.get(url, query_string = {"G1":15, "G2": 14})
    response_data = json.loads(response.get_data())
    assert response.status_code == 200

    #check that keys are in dictionary
    assert 'prediction' in response_data
    assert 'confidence' in response_data

    #check data types
    assert type(response_data['prediction']) is int
    assert type(response_data['confidence']) is float

    #check that model accuracy falls between 0 and 1
    assert response_data['prediction'] == 0 or response_data['prediction'] == 1
    assert response_data['confidence'] >= 0 and response_data['confidence'] <= 1

    #check that all inputs are present
    validate_inputs('/predict')

def test_accuracy_route():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/about/accuracy'

    response = client.get(url)
    response_data = response.get_data()

    assert response.status_code == 200
    assert type(response_data) is int 
    assert response_data <= 100 and response_data >= 0
    assert response_data > 50 # our goal for now, may change the threshold later
    

def test_weight_route():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/about/weight'

    response = client.get(url)
    response_data = json.loads(response.get_data())

    attributes = ["school", "age", "address", "famsize", "Pstatus", "Medu", 
                        "Fedu", "Mjob", "Fjob", "reason", "guardian", "traveltime", 
                        "studytime", "failures", "schoolsup", "famsup", "paid", 
                        "activities", "nursery", "higher", "internet", "romantic", 
                        "famrel", "freetime", "goout", "Dalc", "Walc", "health", 
                        "absences", "G1", "G2", "G3"]

    assert response.status_code == 200
    assert type(response_data) is dict
    
    sumOfWeights = 0.0
    for attribute in attributes:
        assert attribute in response_data.keys()
        assert float(response_data[attribute]) >= 0.0
        sumOfWeights += float(response_data[attribute])

    assert sumOfWeights == 1.0
