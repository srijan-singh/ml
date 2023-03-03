import pandas as pd
df=pd.read_csv("res/insurance2.csv")
print(df)
m=df.height.mean()
print("Mean value = ",m) #165.125
df.height=df.height.fillna(m) #After
print("After fillna")
print(df)
X = df.iloc[:,0:3] #X = ['Age','height','weight']
y = df['premium']
print('Shape of X = ', X.shape) #5
print('Shape of y = ', y.shape) #5


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
random_state=51)
print('Shape of X_train = ', X_train.shape) #(3,3)
print('Shape of y_train = ', y_train.shape) #(3,)
print('Shape of X_test = ', X_test.shape) #(2,3)
print('Shape of y_test = ', y_test.shape) #(2,)


from sklearn import linear_model
lr=linear_model.LinearRegression()
lr.fit(df[['Age','height','weight']],df['premium'])
ans1list=lr.predict([[27,167.56,60]]) # Let’s predict for this
ans2list=lr.predict([[60,165.10,80]]) # Let’s predict for this
print(ans1list)
print(ans2list)
ansfor1strec=lr.predict(X_test[0:1]) #only first record
print("For1st rec ",ansfor1strec)
print("PRedict Premium:",lr.predict(X_test)) #All the record
print("Let's Compare with in % Accuracy")
print(lr.score(X_test, y_test))
