import pandas as pd
df = pd.read_csv('res/empdataLogical.csv')
print(df.to_string()) #15 records so not using print(df.head().to_string())
print(df.shape) #(15,4)
#it contains categorial data in city so let's change
print(df['City'].unique)
df['City'].replace({'NewYork':'2','Mumbai':'3','Tokyo':'4','Singapore':'5'},inplace
=True)
print(df.to_string()) #15 records so not using print(df.head().to_string())
X = df.drop('Eligibleforbonus',axis=1)
y = df['Eligibleforbonus']
print('Shape of X = ', X.shape) #(8,)
print('Shape of y = ', y.shape) #(8,)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
random_state=51)
print('Shape of X_train = ', X_train.shape) #(5,)
print('Shape of y_train = ', y_train.shape) #(5,)
print('Shape of X_test = ', X_test.shape) #(3,)
print('Shape of y_test = ', y_test.shape) #(3,)
#In logistic regression,
#we will do feature scaling because we want accurate result of predictions
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
X_train= st_x.fit_transform(X_train)
X_test= st_x.transform(X_test)
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
#need to convert
temp_Singa_test= st_x.transform([[5,30,52000]])
temp_Tokyo_test= st_x.transform([[4,50,52000]])

ansforSinga=classifier.predict(temp_Singa_test) #We can pass age 2D manually
ansforTokyo=classifier.predict(temp_Tokyo_test) #We can pass age 2D manually
y_pred= classifier.predict(X_test)

print("Singapore 30 52000 =>",temp_Singa_test)
print("Tokyo 50 52000 =>",temp_Tokyo_test)
print("All test records but converted",X_test)
print(ansforSinga)
print(ansforTokyo)
print(y_pred)
print("Score of matching")
print(classifier.score(X_test, y_test))
