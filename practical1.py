import pandas as pd

df = pd.read_csv("res/BagnHousePrice.csv")
print(df.head().to_string())
print(df.shape) #total records (7120, 108)
x=df.drop('price',axis=1) #Other than Price are X
y=df['price'] #dependent variable Y
print("Shape of X = ",x.shape) #Shape of X = (7120, 107)
print("Shape of Y = ",y.shape) #Shape of Y = (7120)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2,
random_state=51)
print('Shape of X_train = ', X_train.shape) #(5696, 107)
print('Shape of y_train = ', y_train.shape) #(5696,)
print('Shape of X_test = ', X_test.shape) #(1424, 107)
print('Shape of y_test = ', y_test.shape) #(1424,)


#FeatureScalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


from sklearn.linear_model import LinearRegression
lr = LinearRegression() #object of modal lr
lr.fit(X_train, y_train) #we are passing
#print("X_test : ",X_test[0, :]) #we are passing test data set 1st house
print("PRedict PRice:",lr.predict([X_test[0, :]])) #we are passing test data set 1st house
#Predict PRice: [76.90661876]
#Print for all X_test
#print("X_test : ",X_test) #we are passing test data set 1st house
#76.90.....
print("PRedict PRice of All houses:",lr.predict(X_test)) 
#we are passing test data set
#80..
print("Let's Compare with original one")
print(y_test)
print("Let's Compare with in % Accuracy")
print(lr.score(X_test, y_test))