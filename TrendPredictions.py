import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import sklearn
from sklearn import linear_model

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures"]]
#print(data)
predict = "G3" #predicting the final grade
y = np.array(data[predict])
x = np.array(data.drop([predict],1)) #makes an array of everything except what we're predicting
#split our data into 4 arrays, test the accuracy of the model we make
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.1)
best=0
"""for _ in range(10):


    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test) #accuracy of the model
    #print(accuracy)
    if(accuracy > best):
        best = accuracy
        with open("studentGrades.pickle" , "wb") as f:
            pickle.dump(linear, f)"""


pickle_in = open("studentGrades.pickle", "rb")
linear = pickle.load(pickle_in)


predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = "G2"
style.use("ggplot")
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()