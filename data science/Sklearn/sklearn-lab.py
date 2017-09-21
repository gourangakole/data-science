# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#libraries to import
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pandas as pd

#sklearn function to load iris data set can be any kind of data sets
#a few are available inside the package
print("loading the data set")
iris = load_iris()

print("")
#load data from librabry and printing the data
print("load data from librabry and printing the data")
X = iris.data
print(X)

print("")
#grabing the attribute names
print("grabing the attribute names")
print(iris.feature_names)

print("")
print("printing the size of X the data row x colums")
#printing the size of the data
print(X.shape)

print("")
print("Y label vector")
Y = iris.target
print(Y)

print("")
print("printing the size of Y the data row")
print(Y.shape)

print("")
print("printing the name of attirbute using target_names setosa=0, versicolor=1, virginica=2")
print("")
print(iris.target_names)

print("")
print("now using knn we instantiated as an object from the KNeighborsClassifier")
k = 1
knn = KNeighborsClassifier(n_neighbors=k)

print("")
print("now by using the method fit on the object we can train our model based on the labels and datasets")
print("")
print(knn.fit(X, Y))

print("")
print("now we can use the method predict to start the training process with the knn object base on the testing of sample data to perform prediction")
test_x =[[3, 4, 5.3, 5.1],[2, 3.1, 7, 9.1]]
predict_x = knn.predict(test_x)
print(predict_x)
print("second data sample to train")
test_x =[[2, 4.1, 2.1, 9.1]]
predict_x = knn.predict(test_x)
print(predict_x)

#now using a external data set to read it as a csv file
iris_df = pd.read_csv('https://raw.githubusercontent.com/mpourhoma/CS4661/master/iris.csv')

#now we check for the data by doing so we check in sets of 10
print("")
print("checking chuncks of data by 10")
print(iris_df[0::10])

print("")
print("now we can conver the label for the type of flower to a numeric value")
print("**Running Function**")

def categ_to_num(a):
    if a == "setosa":
        return 0
    elif a == "versicolor":
        return 1
    elif a == "virginica":
        return 2

print("")
print("now we will apply the function to add another label that has a numeric value")
iris_df['categ_type'] = iris_df['species'].apply(categ_to_num)

print("")
print("we apply a new label giving it a numeric value to destigues the type of category")
print(iris_df[0::10])

print("")
#creating a feature matrix for the iris dataset
print("creating the feature col for iris data")
feature_cols = ['sepal_length','sepal_width','petal_length','petal_width']

X = iris_df[feature_cols]
print("")
print(X)

print("")
print("check that the data is consistent and we have the same amount as whit what we started")
print("")
print(X.shape)

print("")
print("selecting a series of label (the last col) from the dataframe we can print the numeri type or the string type of the category")
Y = iris_df['categ_type']
print("first we print the numeric category")
print(Y[0::10])
print("")
Y = iris_df['species']
print("second we print the type category")
print(Y[0::10])

print("")
print("now we create another object from the knn we instantiated")
print("")
k = 5
knn_v1 = KNeighborsClassifier(n_neighbors=k)
print(knn_v1.fit(X, Y))

print("")
print("Now were going train our model again with different values ")
test_x = [[4, 1, 2.1, 3.1],[1.2, 1, 4, 5]]
pred_x = knn_v1.predict(test_x)
print("")
print(pred_x)

print("")
print("now we will evaluate our accuracy of our classifiers")
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.3, random_state=1)
print("")
#dimension 
print("size of training set")
print(X_Train.shape)
print(Y_Train.shape)
print("")
print("size of testing set")
print(X_Test.shape)
print(Y_Test.shape)
print("")

print(X_Test)
print('\n')
print(Y_Test)

print("")
print("now we train ONLY our training sets")
print("")
print(knn_v1.fit(X_Train, Y_Train))
print("")
print("testing on the testing set")
predi_y = knn_v1.predict(X_Test)
print(predi_y)
print("")

#testing example
y_pred = [0, 2, 1, 1]
y_actual = [0, 1, 2, 1]

score = accuracy_score(y_actual, y_pred)
print("")
print(score)
print("")
print("testing the accuracy")
accuray = accuracy_score(Y_Test, predi_y)
print(accuray)

print("")
results = pd.DataFrame()
results['actual'] = Y_Test
results['prediction'] = predi_y
print("results")
print(results)

print("")
dc_tree = DecisionTreeClassifier()
print("")
print("Decision Tree Classifier")
print("")
print(dc_tree.fit(X_Train, Y_Train))
print("")
predic_y = dc_tree.predict(X_Test)
print("")
print(predic_y)

accu = accuracy_score(Y_Test, predic_y)

print(accu)
