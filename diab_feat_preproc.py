import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
import numpy as np


DATA_FILE = './data_ai/diabetes.csv'
# print(os.path.exists(DATA_FILE))
NUMERIC_FEAT_COLS = ['AGE', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
CATEGORY_FEAT_COLS = ['SEX']

def feat_preprocessing(X_train,X_test):
    encoder = OneHotEncoder(sparse=False)
    encoded_tr_feat = encoder.fit_transform(X_train[CATEGORY_FEAT_COLS])
    encoded_te_feat = encoder.transform(X_test[CATEGORY_FEAT_COLS])

    scaler = MinMaxScaler()
    scaled_tr_feat = scaler.fit_transform(X_train[NUMERIC_FEAT_COLS])
    scaled_te_feat =scaler.transform(X_test[NUMERIC_FEAT_COLS])

    X_train_proc = np.hstack((encoded_tr_feat,scaled_tr_feat))
    X_test_proc = np.hstack((encoded_te_feat,scaled_te_feat))

    return X_train_proc,X_test_proc

def main():
    diabetes_data = pd.read_csv(DATA_FILE)
    X = diabetes_data[NUMERIC_FEAT_COLS+CATEGORY_FEAT_COLS]
    y = diabetes_data['Y']

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/5,random_state=10)
    model_1 = LinearRegression()
    model_1.fit(X_train,y_train)
    r2_1 = model_1.score(X_test,y_test)
    print(r2_1)

    X_train_proc, X_test_proc = feat_preprocessing(X_train,X_test)
    model_2 = LinearRegression()
    model_2.fit(X_train_proc,y_train)
    r2_2 = model_2.score(X_test_proc,y_test)
    print(r2_2)

main()