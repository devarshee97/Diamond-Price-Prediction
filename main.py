import os

from flask import Flask, request, render_template
from flask_cors import cross_origin
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('lasso_regression_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


sc = StandardScaler()
df = pd.read_csv('train_data.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
df_sc = sc.fit_transform(df)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        x = int(request.form['x'])
        y = int(request.form['y'])
        z = int(request.form['z'])
        Volume = x * y * z
        carat = int(request.form["carat"])
        depth = int(request.form['depth'])
        table = int(request.form['table'])
        cut_order = request.form['cut_order']
        if cut_order == 'Ideal':
            cut_order =1
        elif cut_order == 'Premium':
            cut_order =2
        elif cut_order == 'Very Good':
            cut_order = 3
        elif cut_order == 'Good':
            cut_order = 4
        else:
            cut_order = 5

        clarity_order = request.form['clarity_order']
        if clarity_order == 'IF':
            clarity_order = 1
        elif clarity_order == 'VVS1':
            clarity_order = 2
        elif (clarity_order == 'VVS2'):
            clarity_order = 3
        elif (clarity_order == 'VS1'):
            clarity_order = 4
        elif (clarity_order == 'VS2'):
            clarity_order = 5
        elif (clarity_order == 'SI1'):
            clarity_order = 6
        elif (clarity_order == 'SI2'):
            clarity_order = 7
        else:
            clarity_order = 8

        color_order = request.form['color_order']
        if (color_order == 'D'):
            color_order = 1
        elif (color_order == 'E'):
            color_order = 2
        elif (color_order == 'F'):
            color_order = 3
        elif (color_order == 'G'):
            color_order = 4
        elif (color_order == 'H'):
            color_order = 5
        elif (color_order == 'I'):
            color_order = 6
        else:
            color_order = 7

        input_data = [[Volume, carat, depth, table, cut_order, clarity_order, color_order]]
        input_data_scaled = sc.transform(input_data)

        diamond_price = model.predict(input_data_scaled)

        return render_template("index.html",
                               prediction_text="The selling price of your diamond piece will be close to {} US Dollar".format(
                                   diamond_price[0]))


if __name__ == "__main__":
    app.run(debug=True)
