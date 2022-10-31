from flask import Flask

import pandas as pd
import numpy as np

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
    #Can change the arguments later
    response = client.get(url, json = {"age": 18, "absences": 5, "health": 4} )
    assert response.status_code == 200
    assert (response.get_data() == 0 or response.get_data() == 1)

    failed_response1 = client.get(url, json = {"absences": 4, "health": 3})
    assert failed_response1.status_code == 200
    assert failed_response1.get_data() == "Invalid query: Arguments are missing"

    failed_response2 = client.get(url, json = {"age": 4, "health": 3})
    assert failed_response2.status_code == 200
    assert failed_response2.get_data() == "Invalid query: Arguments are missing"

    failed_response3 = client.get(url, json = {"age": "Eighteen", "absences": 2, "health": 3})
    assert failed_response3.status_code == 200
    assert failed_response3.get_data() == "Invalid query: Arguments have invalid types or ranges"

    failed_response4 = client.get(url, json = {"age": -5, "absences": 2, "health": 3})
    assert failed_response4.status_code == 200
    assert failed_response4.get_data() == "Invalid query: Arguments have invalid types or ranges"

    failed_response5 = client.get(url, json = {"age": 17, "absences": -1, "health": 3})
    assert failed_response5.status_code == 200
    assert failed_response5.get_data() == "Invalid query: Arguments have invalid types or ranges"

    failed_response6 = client.get(url, json = {"age": 17, "absences": 24, "health": 0})
    assert failed_response6.status_code == 200
    assert failed_response6.get_data() == "Invalid query: Arguments have invalid types or ranges"

def test_predict_route():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    
    validate_inputs('/predict')
    
    df = pd.read_csv('data/student-mat.csv', sep=';')
    count = 0
    size = df.shape[0]
    for row in df.rows:
        response = client.get('/predict', json = {"age": row["age"], 
                                           "absences": row["absences"],
                                           "health": row["health"] })
        actual_pred = (row["G3"] > 15)
        pred = response.get_data()
        if (pred == 1 and actual_pred) or (pred == 0 and not actual_pred):
            count+=1

    model_accuracy = count/size

    assert model_accuracy > .50 ##can change threshold late


def test_predict_more_route():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/predict/more'

    response = client.get(url)
    response_data = json.loads(response.get_data())
    assert response.status_code == 200

    #check that keys are in dictionary
    assert 'prediction' in response_data
    assert 'confidence' in response_data

    #check data types
    assert type(response_data['prediction']) is float
    assert type(response_data['confidence']) is float

    #check that model accuracy falls between 0 and 1
    assert response.get_data()['prediction'] >= 0 and response.get_data()['prediction'] <= 1
    assert response.get_data()['confidence'] >= 0 and response.get_data()['confidence'] <= 1

    #check that all inputs are present
    validate_inputs(url)

def test_accuracy_route():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/about/accuracy'

    response = client.get(url)
    response_data = int(response.get_data())

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
