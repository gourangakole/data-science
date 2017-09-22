# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 20:42:23 2017

@author: Jose

"""
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

x_min, x_max =X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max =X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

plt.figure(2, figsize=(10,10))
plt.clf()

plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel("Sepal Lenght")
plt.ylabel("Sepal Width")

plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.xticks(())
plt.yticks(())

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])


plt.show()