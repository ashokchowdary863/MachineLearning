# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 19:14:32 2019

@author: ashok
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
#Differentiate the independent variables and depenent variables(dependent variables are the one the model will predict)
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

#Fill missing values by Mean/Avergae technique
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Encode the categorical data (Encode the strings to numbers and normalize the priority)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


#Split the data set into training set and test set
#Good to use 20-25% of dataset as test set and remaning as training set
#randomState : Good to use 42 for better results


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


#Feature scaling (Based on Euclidian distance)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
