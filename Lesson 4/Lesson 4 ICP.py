import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
"""
1. find the correlation between ‘survived’ (target column) and ‘sex’ column for the Titanic use case in class.
 Do you think we should keep this feature?
"""
train_df = pd.read_csv('./train_preprocessed.csv')
test_df = pd.read_csv('./test_preprocessed.csv')

TitanicX_train = train_df.drop("Survived",axis=1)
TitanicY_train = train_df["Survived"]
TitanicX_test = test_df.drop("PassengerId",axis=1).copy()
print(train_df[train_df.isnull().any(axis=1)])
print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False) )
##KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(TitanicX_train, TitanicY_train)
TitanicY_pred = knn.predict(TitanicX_test)
acc_knn = round(knn.score(TitanicX_train, TitanicY_train) * 100, 2)
print("KNN accuracy is:",acc_knn)

"""
2. Implement Naïve Bayes method using scikit-learn library
Use dataset available in https://umkc.box.com/s/ea6wn1cidukan67t02j60nmp1ljln3kd 
Use train_test_split to create training and testing part
Evaluate the model on testing part using score and 
classification_report(y_true, y_pred)
"""

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

Glass_data = pd.read_csv("./glass.csv")
Glass_X = Glass_data.drop("Type",axis=1)
Glass_Y = Glass_data["Type"]
Glass_Y_true = Glass_data["Type"].values
# Glass_X, Glass_Y = load_iris(return_X_y=True)
Glass_X_train, Glass_X_test, Glass_Y_train, Glass_Y_test = train_test_split(Glass_X, Glass_Y, test_size=0.3, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(Glass_X_train, Glass_Y_train).predict(Glass_X_test)

acc_svc = round(gnb.score(Glass_X_test, Glass_Y_test) * 100, 2)
print(classification_report(Glass_Y_test,y_pred))
print("NB accuracy is:", acc_svc)

"""
3. Implement linear SVM method using scikit library
Use the same dataset above
Use train_test_split to create training and testing part
Evaluate the model on testing part using score and 
classification_report(y_true, y_pred)
Which algorithm you got better accuracy? Can you justify why?
"""


from sklearn import svm
Glass_data = pd.read_csv("./glass.csv")
svm_X = Glass_data.drop("Type",axis=1)
svm_Y = Glass_data["Type"]
svm_Y_true = Glass_data["Type"].values
svm_X_train, svm_X_test, svm_Y_train, svm_Y_test = train_test_split(Glass_X, Glass_Y, test_size=0.3, random_state=0)
svc = SVC()
svc.fit(svm_X_train, svm_Y_train)
SVC_Y_pred = svc.predict(svm_X_test)
acc_svc = round(svc.score(svm_X_test, svm_Y_test) * 100, 2)

print(classification_report(svm_Y_test,y_pred))
print("svm accuracy is:", acc_svc)
