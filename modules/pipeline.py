import pandas as pd
from dill import dump
import datetime
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_selector
import os


def delete_columns(df):
    df1 = df.copy()
    df1 = df1.drop(columns=['id',
                            'url',
                            'region',
                            'region_url',
                            'price',
                            'manufacturer',
                            'image_url',
                            'description',
                            'posting_date',
                            'lat',
                            'long'
                            ])
    return df1


def delete_outlier_year(df):
    df1 = df.copy()
    q75 = df1['year'].quantile(0.75)
    q25 = df1['year'].quantile(0.25)
    iter_q = (q75 - q25) * 1.5
    b = (round(q25 - iter_q), round(q75 + iter_q))
    df1['year'] = df1['year'].apply(lambda x: b[0] if x < b[0] else b[1] if x > b[1] else x)
    return df1


def short_model(df):
    import pandas

    df1 = df.copy()
    df1['short_model'] = \
        df1['model'].apply(lambda x: x.lower().split(' ')[0] if not pandas.isna(x) else x)
    df1 = df1.drop(columns=['model'])
    return df1


def age_category(df):
    df1 = df.copy()
    df1['age_category'] = df1['year'].apply(lambda x: 'new'
    if x > 2013 else ('old' if x < 2006 else 'average'))
    return df1


def make_model():
    df = pd.read_csv(os.path.expanduser('~/cat_price_predict/data/train/train_set.csv'))
    data = df.copy()
    x = data.drop(columns=['price_category'])
    y = data['price_category']
    os.path.abspath('modules/model.pkl')
    pipe = Pipeline(steps=[
        ('delete_columns', FunctionTransformer(delete_columns)),
        ('delete_outlier_year', FunctionTransformer(delete_outlier_year)),
        ('age_category', FunctionTransformer(age_category)),
        ('short_model', FunctionTransformer(short_model))
    ])

    pipe_num = Pipeline(steps=[
        ('delete_na', SimpleImputer(strategy='median')),
        ('std', StandardScaler())
    ])

    pipe_cat = Pipeline(steps=[
        ('delete', SimpleImputer(strategy='most_frequent')),
         ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))
         ])

    column_trans = ColumnTransformer([
        ('transform_cat', pipe_cat,\
         make_column_selector(dtype_include=object)),
        ('transform_num', pipe_num, make_column_selector(dtype_include=['int64', 'float64']))
    ])
    # в задании он решил только одометер стандартизировать, год оставил прежним
    preprocessing = Pipeline(steps=[
        ('pipe', pipe),
        ('column_trans', column_trans)
    ])

    models = [
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        SVC()
    ]
    best_score = 0
    best_model = None
    best_pipe = None
    for m in models:
        head_pipe = Pipeline(steps=[
            ('preprocessing', preprocessing),
            ('model', m)
        ])
        score = cross_val_score(head_pipe, x, y, cv=4, scoring='accuracy')
        print(score)
        if best_score < score.mean():
            best_score = score.mean()
            best_pipe = head_pipe
            best_model = m
    best_pipe.fit(x,y)
    print(f'Лучшая модель - {best_model}, оценка - {best_score}')

    model_to_pkl = {'model': best_pipe,
                    'metadata': {
                      'name': 'loan prediction pipeline',
                      'author': 'Alexandr Strigo',
                      'version': 1,
                      'date': datetime.datetime.now(),
                      'type': str(best_model)[:-2],
                      'accuracy': best_score}
                    }

    with open(os.path.expanduser('~/cat_price_predict/data/models/model.pkl'), 'wb')as file:
        dump(model_to_pkl, file)


if __name__ == '__main__':
    make_model()
