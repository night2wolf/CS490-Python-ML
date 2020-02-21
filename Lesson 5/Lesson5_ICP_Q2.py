import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('winequality-red.csv')
##handling missing value
data = train.select_dtypes(include=[np.number]).interpolate().dropna()

numeric_features = train.select_dtypes(include=[np.number])

corr = numeric_features.corr()
# fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality
print (corr['quality'].sort_values(ascending=False)[:5], '\n')
print (corr['quality'].sort_values(ascending=False)[-5:])

#quality_pivot = train.pivot_table(index='alcohol', values='quality', aggfunc=np.median)
#print(quality_pivot)


##Build a linear model
y = np.log(train.quality)
X = data[['alcohol','sulphates','volatile acidity']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)
from sklearn import linear_model
train_copy =pd.get_dummies(X)
train_copy=train_copy.fillna(0)
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
##Evaluate the performance and visualize results
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error

print ('RMSE is: \n', mean_squared_error(y_test, predictions))


##visualize
fig, axes = plt.subplots(1,len(train_copy.columns.values),sharey=True,constrained_layout=True,figsize=(30,15))
for i,e in enumerate(train_copy.columns):
  lr.fit(train_copy[e].values[:,np.newaxis], y.values)
  axes[i].set_title("Best fit line")
  axes[i].set_xlabel('Quality')
  axes[i].set_ylabel(str(e))
  axes[i].scatter(train_copy[e].values[:,np.newaxis], y,color='g')
  axes[i].plot(train_copy[e].values[:,np.newaxis], 
  lr.predict(train_copy[e].values[:,np.newaxis]),color='k')
plt.show()
# credit to stack overflow for plotting :
# https://stackoverflow.com/questions/52404857/how-do-i-plot-for-multiple-linear-regression-model-using-matplotlib