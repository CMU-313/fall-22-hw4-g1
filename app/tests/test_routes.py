from flask import Flask
import pandas as pd
import numpy as np
from app.handlers.routes import configure_routes


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
