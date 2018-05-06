# house_price_pred_feat_preprocessing

# 1. Basic

![](https://github.com/davidkorea/house_price_pred_feat_preprocessing/blob/master/images/feat1.png)

![](https://github.com/davidkorea/house_price_pred_feat_preprocessing/blob/master/images/feat2.png)

![](https://github.com/davidkorea/house_price_pred_feat_preprocessing/blob/master/images/feat3.png)

![](https://github.com/davidkorea/house_price_pred_feat_preprocessing/blob/master/images/feat4.png)

# 2. Code

```python
encoder = OneHotEncoder(sparse=False)
encoded_tr_feat = encoder.fit_transform(X_train[CATEGORY_FEAT_COLS])
encoded_te_feat = encoder.transform(X_test[CATEGORY_FEAT_COLS])

scaler = MinMaxScaler()
scaled_tr_feat = scaler.fit_transform(X_train[NUMERIC_FEAT_COLS])
scaled_te_feat = scaler.transform(X_test[NUMERIC_FEAT_COLS])

X_train_proc = np.hstack((encoded_tr_feat,scaled_tr_feat))
X_test_proc = np.hstack(((encoded_te_feat,scaled_te_feat)))

model = LinearRegression()
model.fit(X_train_proc,y_train)
r2_score2 = model.score(X_test_proc,y_test)
```
