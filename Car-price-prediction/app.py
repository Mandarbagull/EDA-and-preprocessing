from flask import Flask,render_template,request,redirect, send_from_directory,json
import sklearn
import pandas as pd
import pickle
import numpy as np
import os 

app = Flask(__name__)
car = pd.read_csv('cleaned_car.csv')

@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')



@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique())
    fuel_types = car['fuel_type'].unique()

    return render_template('index.html', companies= companies, car_models= car_models, years = years, fuel_types= fuel_types)

@app.route('/', methods = ['POST', 'GET'])
def predict():
    
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique())
    fuel_types = car['fuel_type'].unique()

    
    if request.method == 'POST':
        print('hi')
        company = request.form['company']
        car_model = request.form['car_model']
        year = request.form['year']
        fuel_type = request.form['fuel_type']
        kms_driven = request.form['kilo_driven']
        
        file = open("RandomForestRegressor.pkl",'rb')
        lr = pickle.load(file)
        prediction = lr.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array([car_model, company, year, kms_driven, fuel_type]).reshape(1,5)))
        print(prediction)
        return render_template('index.html', prediction = int(prediction[0]),companies= companies, car_models= car_models, years = years, fuel_types= fuel_types )
if __name__ == '__main__':
    app.run(debug=True)