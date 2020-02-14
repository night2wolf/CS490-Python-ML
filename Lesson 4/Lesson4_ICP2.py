import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
# visualization
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

Glass_data = pd.read_csv("./glass.csv")

Glass_data = Glass_data.drop("Si",axis=1)
#Glass_data = Glass_data.drop("Ca",axis=1)
Glass_data = Glass_data.drop("RI",axis=1)
Glass_X = Glass_data.drop("Type",axis=1)
Glass_Y = Glass_data["Type"]
# RI,Na,Mg,Al,Si,K,Ca,Ba,Fe,Type
#print(Glass_data[['RI', 'Type']].groupby(['Type'], as_index=False).mean().sort_values(by='Type', ascending=True) )
print(Glass_data[['Na', 'Type']].groupby(['Type'], as_index=False).mean().sort_values(by='Type', ascending=True) )
print(Glass_data[['Mg', 'Type']].groupby(['Type'], as_index=False).mean().sort_values(by='Type', ascending=True) )
#print(Glass_data[['Al', 'Type']].groupby(['Type'], as_index=False).mean().sort_values(by='Type', ascending=True) )
#print(Glass_data[['Si', 'Type']].groupby(['Type'], as_index=False).mean().sort_values(by='Type', ascending=True) )
print(Glass_data[['K', 'Type']].groupby(['Type'], as_index=False).mean().sort_values(by='Type', ascending=True) )
#print(Glass_data[['Ca', 'Type']].groupby(['Type'], as_index=False).mean().sort_values(by='Type', ascending=True) )
print(Glass_data[['Ba', 'Type']].groupby(['Type'], as_index=False).mean().sort_values(by='Type', ascending=True) )
print(Glass_data[['Fe', 'Type']].groupby(['Type'], as_index=False).mean().sort_values(by='Type', ascending=True) )
Glass_X_train, Glass_X_test, Glass_Y_train, Glass_Y_test = train_test_split(Glass_X, Glass_Y, test_size=0.3, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(Glass_X_train, Glass_Y_train).predict(Glass_X_test)

acc_svc = round(gnb.score(Glass_X_test, Glass_Y_test) * 100, 2)
# print(classification_report(Glass_Y_test,y_pred))
print("NB accuracy is:", acc_svc)
svc = SVC()
svc.fit(Glass_X_train, Glass_Y_train)
Y_pred = svc.predict(Glass_X_test)
acc_svc = round(svc.score(Glass_X_test, Glass_Y_test) * 100, 2)
print("svm accuracy is:", acc_svc)