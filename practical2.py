import pandas as pd
df = pd.read_csv("res/insurance.csv")
print(df.head().to_string())
print(df.shape) #(5,2)
X = df.iloc[:, 0:1] #X = df['age'] required 2D so
y = df['premium']
print('Shape of X = ', X.shape) #5
print('Shape of y = ', y.shape) #5


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
random_state=51)
print('Shape of X_train = ', X_train.shape) #(3,)
print('Shape of y_train = ', y_train.shape) #(3,)
print('Shape of X_test = ', X_test.shape) #(2,)
print('Shape of y_test = ', y_test.shape) #(2,)
#print(X_train)


from sklearn import linear_model
lr=linear_model.LinearRegression()
lr.fit(X_train, y_train)
ansfor21=lr.predict([[21]]) #We can pass age 2D manually
ansfor50=lr.predict([[50]])
ansfor1strec=lr.predict(X_test[0:1]) #only first record
print(ansfor21)
print(ansfor50)
print(ansfor1strec)
print("PRedict Premium:",lr.predict(X_test)) #All the record
print("Let's Compare with in % Accuracy")
print(lr.score(X_test, y_test))
