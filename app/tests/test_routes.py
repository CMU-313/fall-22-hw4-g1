from flask import Flask

from app.handlers.routes import configure_routes


def test_base_route():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/'

    response = client.get(url)

    assert response.status_code == 200
    assert response.get_data() == b'try the predict route it is great!'


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
