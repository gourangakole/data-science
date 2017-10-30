# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 20:45:13 2017

@author: Jose
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
save_result = []
save_result_one = []
save_result_two = []
save_result_three = []
save_result_four = []

iris_data = pd.read_csv('https://raw.githubusercontent.com/mpourhoma/CS4661/master/iris.csv')
df = pd.DataFrame(iris_data)
print(df)
feature_col = ['sepal_length',  'sepal_width',  'petal_length',  'petal_width'  ]
X = df[feature_col]
print(X.shape)
Y = df['species']
print(Y.shape)
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y, test_size=0.4, random_state=1)
#print(X_Train.shape)
#print(Y_Train.shape)
#print(X_Test.shape)
#print(Y_Test.shape)
i = 60
k = 1
while k < i:
    train_knn_one = KNeighborsClassifier(n_neighbors = k)
    train_knn_one.fit(X_Train, Y_Train)
    X_Predi = train_knn_one.predict(X_Test)
    #print(X_Predi)
    pre = [0,2,1,1]
    actual = [0,2,2,1]
    score = accuracy_score(actual, pre)
    #print(score)
    accuracy = accuracy_score(Y_Test, X_Predi)
    #print(accuracy)
    res = pd.DataFrame()
    res['actual'] = Y_Test
    res['prediction'] = X_Predi
    #print(res)
    save_result.append(accuracy)
    k = k + 1

print("Whole data set complete" + str(save_result))
print("-----------------------------------------------------------------------------------------------------------")
feature_col_one = ['sepal_length']  
feature_col_two = ['sepal_width']  
feature_col_three = ['petal_length']  
feature_col_four = ['petal_width']
X_ONE = df[feature_col_one]
X_TWO = df[feature_col_two]
X_THREE = df[feature_col_three] 
X_FOUR = df[feature_col_four]
all_Fea =[X_ONE, X_TWO, X_THREE, X_FOUR]
Y = df['species']
print("-----------------------------------------------------------------------------------------------------------")

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_ONE,Y, test_size=0.4, random_state=1)
#print(X_Train.shape)
#print(Y_Train.shape)
#print(X_Test.shape)
#print(Y_Test.shape)
i = 60
k = 1
while k < i:
    train_knn_one = KNeighborsClassifier(n_neighbors = k)
    train_knn_one.fit(X_Train, Y_Train)
    X_Predi = train_knn_one.predict(X_Test)
    #print(X_Predi)
    pre = [0,2,1,1]
    actual = [0,2,2,1]
    score = accuracy_score(actual, pre)
    #print(score)
    accuracy = accuracy_score(Y_Test, X_Predi)
    #print(accuracy)
    res = pd.DataFrame()
    res['actual'] = Y_Test
    res['prediction'] = X_Predi
    #print(res)
    save_result_one.append(accuracy)
    k = k + 1

print("Set one complete" + str(save_result_one))

print("-----------------------------------------------------------------------------------------------------------")

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_TWO,Y, test_size=0.4, random_state=1)
#print(X_Train.shape)
#print(Y_Train.shape)
#print(X_Test.shape)
#print(Y_Test.shape)
i = 60
k = 1
while k < i:
    train_knn_one = KNeighborsClassifier(n_neighbors = k)
    train_knn_one.fit(X_Train, Y_Train)
    X_Predi = train_knn_one.predict(X_Test)
    #print(X_Predi)
    pre = [0,2,1,1]
    actual = [0,2,2,1]
    score = accuracy_score(actual, pre)
    #print(score)
    accuracy = accuracy_score(Y_Test, X_Predi)
    #print(accuracy)
    res = pd.DataFrame()
    res['actual'] = Y_Test
    res['prediction'] = X_Predi
    #print(res)
    save_result_two.append(accuracy)
    k = k + 1

print("Set two complete" + str(save_result_two))
print("-----------------------------------------------------------------------------------------------------------")

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_THREE,Y, test_size=0.4, random_state=1)
#print(X_Train.shape)
#print(Y_Train.shape)
#print(X_Test.shape)
#print(Y_Test.shape)
i = 60
k = 1
while k < i:
    train_knn_one = KNeighborsClassifier(n_neighbors = k)
    train_knn_one.fit(X_Train, Y_Train)
    X_Predi = train_knn_one.predict(X_Test)
    #print(X_Predi)
    pre = [0,2,1,1]
    actual = [0,2,2,1]
    score = accuracy_score(actual, pre)
    #print(score)
    accuracy = accuracy_score(Y_Test, X_Predi)
    #print(accuracy)
    res = pd.DataFrame()
    res['actual'] = Y_Test
    res['prediction'] = X_Predi
    #print(res)
    save_result_three.append(accuracy)
    k = k + 1

print("Set three complete" + str(save_result_three))

print("-----------------------------------------------------------------------------------------------------------")

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_THREE,Y, test_size=0.4, random_state=1)
#print(X_Train.shape)
#print(Y_Train.shape)
#print(X_Test.shape)
#print(Y_Test.shape)
i = 60
k = 1
while k < i:
    train_knn_one = KNeighborsClassifier(n_neighbors = k)
    train_knn_one.fit(X_Train, Y_Train)
    X_Predi = train_knn_one.predict(X_Test)
    #print(X_Predi)
    pre = [0,2,1,1]
    actual = [0,2,2,1]
    score = accuracy_score(actual, pre)
    #print(score)
    accuracy = accuracy_score(Y_Test, X_Predi)
    #print(accuracy)
    res = pd.DataFrame()
    res['actual'] = Y_Test
    res['prediction'] = X_Predi
    #print(res)
    save_result_four.append(accuracy)
    k = k + 1

print("Set four complete" + str(save_result_four))
print("-----------------------------------------------------------------------------------------------------------")
