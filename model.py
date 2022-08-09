"""
Churn Modelling using ANN python

"""

# importing 
import pandas as pd
import numpy as np



# loading the dataset
data= pd.read_csv("data/Churn_Modelling.csv")

# Preprocessing of Data

# Label Encoding the Categorical data
from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
data['Geography']= le.fit_transform(data['Geography'])
data ['Gender'] = le.fit_transform(data['Gender'])

x_data = data.iloc[:,3:13].values
y_data = data.iloc[:,13].values


# Train test split
from sklearn.model_selection import train_test_split
x_train ,x_test, y_train , y_test = train_test_split(x_data, y_data , test_size= 0.2, random_state= 0)

# Standard  Scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# ANN - Defining 

import keras 
from keras.models import Sequential
from keras.layers import Dense

machine = Sequential()

machine.add(Dense(13,activation='relu',use_bias=True))
machine.add(Dense(6,activation='relu',use_bias=True))
machine.add(Dense(1,activation='relu',use_bias=True))

# compiling the model
machine.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

# fitting the model
machine.fit(x_train , y_train , batch_size=10 , epochs=100)

# prediting the model
y_pred= machine.predict(x_test)
y_pred = (y_pred>0.5)


# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix :")
print(cm)

# accuracy score
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print("Accuracy Score :",ac)

# done