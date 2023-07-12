import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


# This is a data set describing information about stars, and based on the data given we want to predict
# the classification as g = gamme or h = hadron

# features -- pass into model to predict the class (label)
cols = [
    "fLength",
    "fWidth",
    "fSize",
    "fConc",
    "fConc1",
    "fAsym",
    "fM3Long",
    "fM3Trans",
    "fAlpha",
    "fDist",
    "class"
]

data = pd.read_csv("magic04.data", names=cols)
#print(data.head())

data['class'] = (data['class'] == "g").astype(int)
#print(data.head())

for label in cols[:-1]:
    plt.hist(data[data["class"] == 1][label], color = "blue", label = "gamma", alpha = 0.7, density=True) 
    # getting all the values where class = g
    
    plt.hist(data[data["class"] == 0][label], color = "red", label = "hadrons", alpha = 0.7, density=True) 
    # getting all the values where class = g
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    #plt.show()

    # SET DATASETS

    train, valid, test = np.split(data.sample(frac=1), [int(0.6*len(data)), int(0.8*len(data))])
    # setting training data   = 0 - 60%
    # setting validation data = 60 - 80%
    # setting validation data = 80 - 100%

print(len(train[train["class"] == 1]))
print(len(train[train["class"] == 0]))

def scale_data (dataframe, oversample = False):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[:-1]].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)

    data = np.hstack((x,np.reshape(y, (-1,1))))

    return data, x, y

train, x_train, y_train = scale_data(train, oversample = True)
valid, x_train, y_train = scale_data(valid, oversample = False)
test, x_train, y_train = scale_data(test, oversample = False)

