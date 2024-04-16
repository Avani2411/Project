import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

ridge=pickle.load(open('Pickle_files/ridge_project.pkl','rb'))
scaler=pickle.load(open('Pickle_files/scaler_project.pkl','rb'))


##route for home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))


        new_data_sc=scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])

        Result=ridge.predict(new_data_sc)

        return render_template('index.html',Result=Result[0])
    
    
    else:
        return render_template('index.html')
    
if __name__=="__main__":
    app.run(host="0.0.0.0", port=5001)