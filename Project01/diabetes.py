from sklearn import datasets
diabetes = datasets.load_diabetes()

print(diabetes.data)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
linear=linear_model.LinearRegression()

# Training set
x_train=diabetes.data[:-20]
y_trian=diabetes.target[:-20]

# Test
x_test=diabetes.data[-20:]
y_test=diabetes.target[-20:]

x0_test=x_test[:,0]
#print(x0_test)
x0_train=x_train[:,0]

x0_test=x0_test[:,np.newaxis]
x0_train=x0_train[:,np.newaxis]
linear.fit(x0_train,y_trian)
y=linear.predict(x0_test)
plt.scatter(x0_test,y_test,color='k')
plt.plot(x0_test,y,color='b')
plt.show()