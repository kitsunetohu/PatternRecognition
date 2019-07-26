import  numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

obj = KNeighborsClassifier()

iris = datasets.load_iris()
x = iris.data
y = iris.target

print(x[:4, :])
print(y)
obj.fit(x, y)
y_pred = obj.predict(x[:4, :])

print(y_pred)
