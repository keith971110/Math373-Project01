import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data = pd.read_csv('train.csv')
data.head()

data.shape
data.isnull().sum()
data = data.drop(labels=['PassengerId','Name','Ticket','Cabin'],axis=1)
data.head()

data = data.dropna()
data_dummy = pd.get_dummies(data[['Sex','Embarked']])
data_dummy.head()

data_conti = pd.DataFrame(data,columns=['Survived','Pclass','Age','SibSp','Parch','Fare'],index=data.index)
data = data_conti.join(data_dummy)
data.head()

X = data.iloc[:,1:] 
y = data.iloc[:,0] 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0) 

stdsc = StandardScaler()
X_train_conti_std = stdsc.fit_transform(X_train[['Age','SibSp','Parch','Fare']]) 
X_test_conti_std = stdsc.fit_transform(X_test[['Age','SibSp','Parch','Fare']]) 
#CHange ndarray to DataFrame
X_train_conti_std = pd.DataFrame(data = X_train_conti_std,columns=['Age','SibSp','Parch','Fare'],index=X_train.index)
X_test_conti_std = pd.DataFrame(data = X_test_conti_std,columns=['Age','SibSp','Parch','Fare'],index=X_test.index)

classifier = LogisticRegression(random_state=0) 
classifier.fit(X_train,y_train) #model train

y_pred = classifier.predict(X_test)
confusion_matrix = confusion_matrix(y_pred,y_test)
print(confusion_matrix)
# Test accuracy
print(classifier.score(X_test,y_test))