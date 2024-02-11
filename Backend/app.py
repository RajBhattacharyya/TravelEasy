from flask import Flask,request, jsonify
from flask_restful import Resource, Api
import pickle
import pandas as pd 
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

@app.route("/")
def hap():
    return "maow"



@app.route("/predict/<int:hp>/<int:sa>/<int:fd>/<int:tc>/<int:loc>/<int:loc2>")
def predict(hp,sa,fd,tc,loc,loc2):
    try:    
        t_model=joblib.load("model.joblib")
        res=t_model.predict([[hp,sa,fd,tc,loc,loc2]])
        return (jsonify(res[0]))
    except ValueError:
        return jsonify({"error": "Invalid input. Please provide valid float values for parameters."})



if __name__ == '__main__':
    app.run(debug=True)
