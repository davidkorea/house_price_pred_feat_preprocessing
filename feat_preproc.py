import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
import numpy as np

DATA_FILE = './data_ai/house_data.csv'
# print(os.path.exists(DATA_FILE))
NUMERIC_FEAT_COLS = ['sqft_living', 'sqft_above', 'sqft_basement', 'long', 'lat']
CATEGORY_FEAT_COLS = ['waterfront']

def feat_preprocessing(X_train,X_test):
    encoder = OneHotEncoder(sparse=False)
    encoded_tr_feat = encoder.fit_transform(X_train[CATEGORY_FEAT_COLS])
    encoded_te_feat = encoder.transform(X_test[CATEGORY_FEAT_COLS])

    scaler = MinMaxScaler()
    scaled_tr_feat = scaler.fit_transform(X_train[NUMERIC_FEAT_COLS])
    scaled_te_feat = scaler.transform(X_test[NUMERIC_FEAT_COLS])

    X_train_proc = np.hstack((encoded_tr_feat,scaled_tr_feat))
    X_test_proc = np.hstack(((encoded_te_feat,scaled_te_feat)))
    return X_train_proc,X_test_proc


def main():
    house_data = pd.read_csv(DATA_FILE, usecols=NUMERIC_FEAT_COLS+CATEGORY_FEAT_COLS+['price'])

    X = house_data[NUMERIC_FEAT_COLS+CATEGORY_FEAT_COLS]
    y = house_data['price']

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=10)

    linear_reg = LinearRegression()
    linear_reg.fit(X_train,y_train)
    r2_score = linear_reg.score(X_test,y_test)
    print(r2_score)

    X_train_proc, X_test_proc = feat_preprocessing(X_train,X_test)
    model = LinearRegression()
    model.fit(X_train_proc,y_train)
    r2_score2 = model.score(X_test_proc,y_test)
    print(r2_score2)



main()