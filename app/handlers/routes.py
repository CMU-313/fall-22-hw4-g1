import this
from flask import Flask, jsonify, request, abort, make_response
import joblib
import pandas as pd
import numpy as np
import os
from pathlib import Path

def configure_routes(app):

    this_dir = os.path.dirname(__file__)
    model_path = os.path.join(this_dir, "model.pkl")
    clf = joblib.load(model_path)


    @app.route('/')
    def hello():
        return "try the predict route it is great!"


    @app.route('/about/accuracy')
    def accuracy():
        app = Flask(__name__)
        configure_routes(app)
        client = app.test_client()
    
        super_dir = Path.cwd().parent
        data_path = os.path.join(super_dir, 'data', 'student-mat.csv')
        df = pd.read_csv(data_path, sep=';')
        count = 0
        size = df.shape[0]
        for row in df.iterrows():
            response = client.get('/predict', data = {"G1": row["G1"], 
                                                      "G2": row["G2"] })
        actual_pred = (row["G3"] > 15)
        pred = response.get_data()
        if (pred == 1 and actual_pred) or (pred == 0 and not actual_pred):
            count+=1

        model_accuracy = count/size
        return({"accuracy": model_accuracy})
        # app = Flask(__name__)
        # configure_routes(app)
        # client = app.test_client()
        # app_dir = os.path.dirname(this_dir)
        # super_dir = os.path.dirname(app_dir)
        # data_path = os.path.join(super_dir, "data", "student.mat.csv")
        # print(data_path, repr(data_path))
        # data_path = data_path.replace("\\\\", "@")
        # print(data_path, repr(data_path))
        # df = pd.read_csv(data_path, sep=';')
        # count = 0
        # size = df.shape[0]
        # for row in df.rows:
        #     response = client.get('/predict', query_string = {'G1': row["G1"], 'G2': row["G2"]})
        #     actual_pred = (row["G3"] > 15)
        #     pred = response.get_data()
        #     if (pred == 1 and actual_pred) or (pred == 0 and not actual_pred):
        #         count+=1

        # return (count/size)
        

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
        try:
            G1 = int(G1)
            G2 = int(G2)
        except:
            abort(make_response(
                    jsonify({"message" : "Invalid query: G1 and G2 arguments are invalid"}), 400)
                  )
        #Check boundaries
        if G1<0 or G1>20 or G2<0 or G2>20:
            abort(make_response(
                    jsonify({"message" : "Invalid query: G1 and G2 arguments are invalid"}), 400)
                  )
        query_df = pd.DataFrame({
            'G1': pd.Series(G1),
            'G2': pd.Series(G2)
        })
        query = pd.get_dummies(query_df)
        return query


    @app.route('/predict', methods =['GET'])
    def predict():
       
        G1 = request.args.get('G1')
        G2 = request.args.get('G2')
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
