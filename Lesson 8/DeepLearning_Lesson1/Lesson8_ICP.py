import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


dataset = pd.read_csv("Breas cancer.csv")
# Convert Malignant and  Benign to numerical data
dataset["diagnosis"] = dataset["diagnosis"].replace('M',1)
dataset["diagnosis"] = dataset["diagnosis"].replace('B',0)
dataset = dataset.values
# Normalize data

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(dataset)

# "id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst",
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,2:31], dataset[:,1],
                                                    test_size=0.25, random_state=87)
np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(20, input_dim=29, activation='relu')) # hidden layer
my_first_nn.add(Dense(20, input_dim=29, activation='relu')) # hidden layer
my_first_nn.add(Dense(20, input_dim=29, activation='relu')) # hidden layer

my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100,
                                     initial_epoch=0)
my_first_nn.summary()
loss, accuracy = my_first_nn.evaluate(X_test, Y_test)
print("LOSS: {}".format(loss))
print("ACCURACY: {}".format(accuracy))

