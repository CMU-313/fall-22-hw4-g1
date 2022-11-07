import this
from flask import Flask, jsonify, request, abort
import joblib
import pandas as pd
import numpy as np
import os

def configure_routes(app):

    this_dir = os.path.dirname(__file__)
    model_path = os.path.join(this_dir, "model.pkl")
    clf = joblib.load(model_path)


    @app.route('/')
    def hello():
        return "try the predict route it is great!"


    @app.route('/about/accuracy')
    def accuracy():
        return "75"


    @app.route('/about/weight')
    def weight():
        attributes = ["school", "age", "address", "famsize", "Pstatus", "Medu", 
                        "Fedu", "Mjob", "Fjob", "reason", "guardian", "traveltime", 
                        "studytime", "failures", "schoolsup", "famsup", "paid", 
                        "activities", "nursery", "higher", "internet", "romantic", 
                        "famrel", "freetime", "goout", "Dalc", "Walc", "health", 
                        "absences", "G1", "G2", "G3"]
        weights = dict()
        for attribute in attributes:
            weights[attribute] = "0.0"
        weights["age"] = "1.0"

        return jsonify(weights)

    def get_query(G1, G2):
        #Convert types
        G1 = int(G1)
        G2 = int(G2)
        #Check boundaries
        if G1<0 or G1>20 or G2<0 or G2>20:
            abort(400, description = "Invalid query: Arguments are missing")
        query_df = pd.DataFrame({
            'G1': pd.Series(G1),
            'G2': pd.Series(G2)
        })
        query = pd.get_dummies(query_df)
        return query


    @app.route('/predict', methods =['GET'])
    def predict():
       
        G1 = request.args['G1']
        G2 = request.args['G2']
        query = get_query(G1, G2)
        prediction = clf.predict(query)
        output = dict()
        output["prediction"] = np.ndarray.item(prediction)
        return jsonify(output)


    @app.route('/predict/more')
    def predict_more():
        
        G1 = request.args['G1']
        G2 = request.args['G2']
        
        query = get_query(G1, G2)
        
        prediction = clf.predict(query)
        confidence = clf.predict_proba(query)
        
        output = dict()
        output["prediction"] = np.ndarray.item(prediction)
        output["confidence"] = confidence[0][int(output["prediction"])]
        return jsonify(output)
