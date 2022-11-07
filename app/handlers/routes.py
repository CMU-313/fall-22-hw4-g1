import this
from flask import Flask, jsonify, request
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


    @app.route('/predict', methods =['GET'])
    def predict():
        
        #use entries from the query string here but could also use json
        #age = request.args.get('age', dtype='float64')
        #absences = request.args.get('absences', dtype='float64')
        #health = request.args.get('health', dtype = 'float64')
        G1 = request.args.get('G1')
        G2 = request.args.get('G2')
        print(G1)
        print(G2)
        #data = [[age], [health], [absences], [G1], [G2]]
        query_df = pd.DataFrame({
         #   'age': pd.Series(age),
          #  'health': pd.Series(health),
           # 'absences': pd.Series(absences),
            'G1_15': pd.Series(G1),
            'G2_15': pd.Series(G2)
        })
        query = pd.get_dummies(query_df)
        print(query)
        prediction = clf.predict(query)
        output = dict()
        output["prediction"] = np.ndarray.item(prediction)
        return jsonify(output)


    @app.route('/predict/more')
    def predict_more():
        #use entries from the query string here but could also use json
       # age = request.args.get('age', dtype='float64')
        #absences = request.args.get('absences', dtype='float64')
        #health = request.args.get('health', dtype = 'float64')
        G1 = request.args.get('G1')
        print(G1)
        G2 = request.args.get('G2')
        print(G2)
        #data = [[age], [health], [absences], [G1], [G2]]
        query_df = pd.DataFrame({
            #'age': pd.Series(age),
            #'health': pd.Series(health),
           # 'absences': pd.Series(absences),
            'G1': pd.Series(G1),
            'G2': pd.Series(G2)
        })
        #predictions, accuracy
        query = pd.get_dummies(query_df)
        print(query)
        prediction = clf.predict(query)
        confidence = clf.predict_proba(query)
        print(confidence)
        output = dict()
        output["prediction"] = np.ndarray.item(prediction)
        print(confidence[0][output["prediction"]])
        output["confidence"] = confidence[0][int(output["prediction"])]
        return jsonify(output)
