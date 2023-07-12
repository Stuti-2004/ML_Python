import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

data = pd.read_csv("student_mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
#print(data.head())

predict = "G3" #Label is what you are trying to get

x = np.array(data.drop([predict], axis=1)) #drops G3 the thing we are trying to predict
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

#split into 4 variables

'''
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

    
    linear = linear_model.LinearRegression()

    linear.fit(x_train,y_train)
    accuracy = linear.score(x_train,y_train)
    print(accuracy)

    if accuracy > best:
        best = accuracy
        with open("Studentmodule.pickle", "wb") as f:
            pickle.dump(linear, f)
'''

pickle_in = open("Studentmodule.pickle", "rb")
linear = pickle.load(pickle_in)


print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = "studytime"
style.use("ggplot")
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()

