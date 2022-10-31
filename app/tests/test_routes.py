from flask import Flask
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