import pandas as pd
df=pd.read_csv("res/tennis.csv")
print(df.head())
print(df.shape)
#now let's convert this text data

from sklearn.preprocessing import LabelEncoder
Outlook=LabelEncoder()
Temperature =LabelEncoder()
Humidity =LabelEncoder()
Wind =LabelEncoder()
PlayTennis=LabelEncoder()
df['Outlook']=Outlook.fit_transform(df['Outlook'])
df['Temperature']=Temperature.fit_transform(df['Temperature'])
df['Humidity']=Humidity.fit_transform(df['Humidity'])
df['Wind']=Wind.fit_transform(df['Wind'])
df['PlayTennis']=Outlook.fit_transform(df['PlayTennis'])
print(df)

X=df.drop('PlayTennis',axis=1)
y=df['PlayTennis']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,
random_state=51)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='gini')
classifier.fit(X_train,y_train)
print(classifier.predict(X_test)) #0 no 1 yes
print(classifier.score(X_test,y_test))
from sklearn import tree
tree.plot_tree(classifier)
