# flask, scikit-learn, pandas pickle-mixin

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
from flask import Flask, render_template, request

app = Flask(__name__)
data = pd.read_csv('templates\cleaned_data.csv')
pipe = pickle.load(open('templates\RidgeModel.pkl','rb'))

@app.route('/')
def index():
    # print('hi2')
    locations = sorted(data['location'].unique())
    area_types = sorted(data['area_type'].unique())
    
    return render_template('index.html', locations=locations,area_types=area_types)


@app.route('/predict', methods=['POST'])
def predict():
    # print('hi3')
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')
    area_type = request.form.get('area_type')
    balcony = request.form.get('balcony')
    print('hi2')
    print(location, bhk, bath, sqft, area_type, balcony)
    input = pd.DataFrame([[location, sqft, bath, bhk, area_type, balcony]],columns=['location', 'total_sqft', 'bath', 'bhk', 'area_type', 'balcony'])
    prediction = pipe.predict(input)[0] *100000 
    return str(np.round(prediction,2))

if __name__ == '__main__':
    app.run(debug=True)