# the glob module is used to retrieve files/pathnames matching a specified pattern.
import glob
from keras.models import Sequential, load_model
import numpy as np
import pandas as pd
import keras as k
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt

#load the data
df= pd.read_csv('kidney_disease.csv')

#print data
print(df.head())
#get the shape of the data(the numbers of rows and cols)
print(df.shape)

#create a list of column names to keep
df=df.filter(['sg','al','sc','hemo','pcv','wc','rbc','htn','classification'])
print(df.head())
print(df.shape)

#Drop the rows with na or missing value
df=df.dropna(axis=0)
print(df.head())
print(df.shape)

#Transform the non-numeric data in the columns
for column in df.columns:
    if df[column].dtype== np.number:
        continue
    df[column]=LabelEncoder().fit_transform(df[column])
#print the first 5 rows of the new cleaned data set
print(df.head())

#Split the data into independent(X) dataset (the features) and dependent (Y) dataset (the target)
X= df.drop(['classification'],axis=1)
Y=df['classification']

#Feature Scaling
#min max scaler method scales the data set so that all the input features lie between 0 and 1
x_scaler= MinMaxScaler()
x_scaler.fit(X)
column_names= X.columns
X[column_names]= x_scaler.transform(X)

#Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, shuffle= True)

#Building the model
model= Sequential()
model.add(Dense(256,input_dim=len(X.columns),kernel_initializer=k.initializers.random_normal(seed=13), activation= 'relu' ))
model.add=(Dense(1,activation= 'hard_sigmoid'))

#compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam',metrics='accuracy')

#training the model
history=model.fit(X_train,Y_train, epochs=3000, batch_size= X_train.shape[0])

#saving the model
model.save('ckd.model')

#visualising loss and accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('model accuracy and loss')
plt.ylabel('accuracy and loss')
plt.xlabel('epoch')
plt.show()