import pickle

from flask import request,Flask,url_for,jsonify,render_template
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt


app = Flask(__name__)


#Load model
model = pickle.load(open('reg_model.pkl','rb'))

#Load preprocessor
preprocess = pickle.load(open('scale.pkl','rb'))



#Function to convert data to a pd Dataframe
#This is because the preprocessor would require column conversion
def convert_to_df(data):

    df={}
    for k in data:
        df[k] = data[k]
    return pd.DataFrame(df)







def predict_price(df):
    lst = ['Year','Engine','FuelConsumption','Milleage']
    df.loc[:,lst] = df[lst].astype(np.float64)
    df.loc[:,'FuelConsumption'] = df.FuelConsumption / 100
    
    print(df.FuelConsumption)

    
    #preprocess data
    new_data = preprocess.transform(df).toarray()
    
    #Make prediction
    prediction = model.predict(new_data)
    
    return  str(prediction[0])
    
    
 
 
 
 



@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict', methods=['POST'])
def predict():
    car_data = {
        'Brand': [request.form['Brand'].lower().capitalize()],
        'Year': [request.form['Year']],
        'Car_Type': [request.form['Car_Type'].lower().upper()],
        'Transmission': [request.form['Transmission'].lower().capitalize()],
        'Engine': [request.form['Engine']],
        'FuelConsumption': [request.form['FuelConsumption']],
        'Mileage': [request.form['Mileage']]
    }
    
    car_data = convert_to_df(car_data)

    predicted_price = predict_price(car_data)
    
    

    return render_template('prediction.html', predicted_price=predicted_price)


   
    
    

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=4000,debug=True)

