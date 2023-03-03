import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
print(data.keys()) #will print all the keys
#31 features
print(data.feature_names)
#print(data.data) no meaning to print it
cancer_Data=np.array(data.data)
#print(cancer_Data.shape) #(569,30)
#print(data.target)
#print(data.target_names)
df_cancer=pd.DataFrame(cancer_Data,columns=data.feature_names)
df=pd.DataFrame(np.c_[data.data,data.target],columns=[list(data.feature_names
) + ['target']])
print(df.head())
print(df.shape)


print(df.tail())
X=df.iloc[:,0:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=2020)
print('Shape of X_train = ', X_train.shape) #(455, 30)
print('Shape of y_train = ', y_train.shape) #(455)
print('Shape of X_test = ', X_test.shape) #(114,30)
print('Shape of y_test = ', y_test.shape) #(114,)
from sklearn.neighbors import KNeighborsClassifier
Classifier= KNeighborsClassifier(n_neighbors=5)
Classifier.fit(X_train,y_train)
print(Classifier.score(X_test,y_test))
#Let's predict Cancer , so we need patient’s data
#one row's example , one patient’s 30 columns value
patient1 = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373,
0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656,
0.7119, 0.2654, 0.4601, 0.1189]


patient1 = np.array([patient1])
print(patient1)
Classifier.predict(patient1)
pred = Classifier.predict(patient1)
if pred[0] == 0:
    print('Patient has Cancer (malignant tumor)')
else:
    print('Patient has no Cancer (malignant benign)')
