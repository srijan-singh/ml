import pandas as pd
df = pd.read_csv('res/Advertising.csv')
print(df.head().to_string())
print(df.shape) #(200,5)
X = df.iloc[:, 0:3] #X = df['TV','Radio','Newspaper'] required 2D so
y = df['Sales']
print('Shape of X = ', X.shape) #(200,3)
print('Shape of y = ', y.shape) #(200,)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
random_state=51)
print('Shape of X_train = ', X_train.shape) #(140,3)
print('Shape of y_train = ', y_train.shape) #(140,)
print('Shape of X_test = ', X_test.shape) #(60,3)
print('Shape of y_test = ', y_test.shape) #(60,)
from sklearn import linear_model
lr=linear_model.LinearRegression()
lr.fit(X_train, y_train)
ansfor1=lr.predict([[50,37,60]]) #We can pass age 2D manually
ansfor1strec=lr.predict(X_test[0:1]) #only first record
print(ansfor1)
print(ansfor1strec)
print("PRedict Advertisment:",lr.predict(X_test)) #All the record
print("Let's Compare with in % Accuracy")
print(lr.score(X_test, y_test))
