import os
import sys
path = os.path.expanduser('~/cat_price_predict')
sys.path.insert(0, path)
import pandas as pd
import dill
import json


def predict():
    with open(os.path.expanduser('~/cat_price_predict/data/models/model.pkl'), 'rb') as file:
        model = dill.load(file)
    l = os.listdir((os.path.expanduser('~/cat_price_predict/data/test')))
    result = []
    id = []
    price = []
    for item in l:
        path = os.path.expanduser('~/cat_price_predict/data/test') + '/' + item
        with open(path, 'rb') as file:
            form = json.load(file)
        df = pd.DataFrame.from_dict([form])
        y = model['model'].predict(df)
        result.append(y[0])
        id.append(form['id'])
        price.append(form['price'])

    df = pd.DataFrame({'id': id, 'price': price, 'result': result})
    df.to_csv(os.path.expanduser('~/cat_price_predict/data/predictions/predict.csv'))


if __name__ == '__main__':
    predict()
