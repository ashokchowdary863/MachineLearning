# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 21:14:31 2019

@author: ashok
Simple Liner Regression
    y = b0 + b1 * x

No need of feature scaling for SLR but there are suprises
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
#Differentiate the independent variables and depenent variables(dependent variables are the one the model will predict)
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values


#Split the data set into training set and test set
#Good to use 20-25% of dataset as test set and remaning as training set
#randomState : Good to use 42 for better results
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)


#Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the test set results
Y_pred = regressor.predict(X_test)

#Visualising the training set results
plt.scatter(X_train,Y_train,color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'green')
plt.title('Salary vs Experience [Training Set]')
plt.xlabel('Yrs of Exp')
plt.ylabel('Salary')
plt.show()

#Visualising the test set results
plt.scatter(X_test,Y_test,color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'green')
plt.title('Salary vs Experience [Test Set]')
plt.xlabel('Yrs of Exp')
plt.ylabel('Salary')
plt.show()

