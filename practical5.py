import pandas as pd
df = pd.read_csv('res/MELBOURNE_HOUSE_PRICES_LESS.csv')
print(df.head().to_string())
print(df.shape) #(200, 13)
print(df.nunique())
print(df.isna().sum())
df.dropna(inplace=True)
print(df.shape) #(159,13) Price Drop
print(df.isna().sum()) #none of them are 0
df=pd.get_dummies(df,drop_first=True)
print(df.head().to_string())
X = df.drop('Price',axis=1)
y = df['Price']
print('Shape of X = ', X.shape) #(159, 312)
print('Shape of y = ', y.shape) #(159,)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
random_state=51)
print('Shape of X_train = ', X_train.shape) #(111, 312)
print('Shape of y_train = ', y_train.shape) #(111,)
print('Shape of X_test = ', X_test.shape) #(48,312)
print('Shape of y_test = ', y_test.shape) #(48,)

from sklearn import linear_model
lr=linear_model.LinearRegression()
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))
print(lr.score(X_train,y_train))

from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import r2_score
lasso_regression = Lasso()
lasso_regression.fit(X_train,y_train)
x_test_predict_lasso = lasso_regression.predict(X_test)
print(r2_score(y_test, x_test_predict_lasso))
ridge_regression = Ridge()
ridge_regression.fit(X_train,y_train)
x_test_predict_ridge = ridge_regression.predict(X_test)
print(r2_score(y_test, x_test_predict_ridge)) #not preferable
