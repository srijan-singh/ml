import pandas as pd
df = pd.read_csv('res/insurance3.csv')
print(df.head().to_string())
print(df.shape) #(8,2)
df['boughtinsurance'].replace({'no':'0','yes':'1'},inplace=True)
print(df) #it returns cateogrial data , so it's changed
X = df[['age']]
y = df[['boughtinsurance']]
print('Shape of X = ', X.shape) #(8,)
print('Shape of y = ', y.shape) #(8,)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
random_state=51)
print('Shape of X_train = ', X_train.shape) #(5,)
print('Shape of y_train = ', y_train.shape) #(5,)
print('Shape of X_test = ', X_test.shape) #(3,)
print('Shape of y_test = ', y_test.shape) #(3,)


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
print(lr.predict([[60]])) #Interested
print(lr.predict([[21]]))
print("X_test ",X_test)
print("PRedict Yes or No:",lr.predict(X_test)) #All the record
print("Let's Compare with in % Accuracy")
print(lr.score(X_test, y_test))