
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

# print(diabetes.DESCR)
diabetes_X = diabetes.data

# print(diabetes_X)

#Creating Train and Test set
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

diabetes_Y_train = diabetes_y = diabetes.target[:-30]
diabetes_Y_test = diabetes_y = diabetes.target[-30:]

#Creation of a model
model = linear_model.LinearRegression()

#Model Training
model.fit(diabetes_X_train,diabetes_Y_train)

#Model Prediction
diabetes_Y_predict = model.predict(diabetes_X_test)

print('Mean squared error is :', mean_squared_error(diabetes_Y_test,diabetes_Y_predict))
print('Weights: ', model.coef_)
print('Intercepts: ', model.intercept_)

#Graph Ploting

# plt.scatter(diabetes_X_test,diabetes_Y_test)
# plt.plot(diabetes_X_test,diabetes_Y_predict)
# plt.show()

#Previous_Result
# Mean squared error is : 3035.060115291269
# Weights:  [941.43097333]
# Intercepts:  153.39713623331644