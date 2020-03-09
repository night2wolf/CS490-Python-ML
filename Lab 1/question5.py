# @author Landon
# imports
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# NOTE: some code and ideas borrowed from previous ICPs completed

# LOAD DATA SET

train_df = pd.read_csv('./titanic.csv')

# DATA SET PRE-PROCESSING

# drop unrelated data
train_df = train_df.drop('PassengerId', axis=1)
train_df = train_df.drop('Name', axis=1)
train_df = train_df.drop('Ticket', axis=1)
train_df = train_df.drop('Cabin', axis=1)

# fill nulls
train_df['Embarked'] = train_df['Embarked'].fillna(train_df.Embarked.dropna().mode()[0])
train_df['Age'] = train_df['Age'].fillna(int(train_df.Age.dropna().mean()))

# eliminate low correlation categorical features
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
plt.show()

grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
plt.show()

print("Removing 'Embarked' column...")
train_df.drop('Embarked', axis=1, inplace=True)

# map categorical features to numeric
# train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
train_df['Sex'] = train_df['Sex'].map({'female': 1, 'male': 0}).astype(int)

# eliminate low correlation numeric features
train_df_features = train_df.drop('Survived', axis=1)

for column_name in train_df_features.columns:
    print("Correlation for: ", column_name)
    print(train_df[['Survived', column_name]].
          groupby(['Survived']).
          mean().
          sort_values(by=column_name, ascending=False))

# eliminate outliers in numeric features

fare_df = train_df_features.get("Fare")
z = np.abs(stats.zscore(fare_df))
outliers = []
pos_to_drop = []
for i, score in enumerate(z):
    if score > 3:
        outliers.append((i, score))
        pos_to_drop.append(i)
        # print(i, score)

print("The following outliers will be removed:")
print(outliers)

train_df_features.drop(train_df_features.index[pos_to_drop], inplace=True)
train_df.drop(train_df.index[pos_to_drop], inplace=True)

# PARTITION DATA SET
X = train_df_features
y = train_df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# RUN CLASSIFICATION ALGORITHMS

# Naive Bayes
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
acc_gnb = round(gnb.score(X_test, y_test) * 100, 2)

print("nb accuracy is:", acc_gnb)
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))
print(classification_report(y_test, y_pred))

# SVM
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_test, y_test) * 100, 2)

print("svm accuracy is:", acc_svc)
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))
print(classification_report(y_test, y_pred))

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)

print("KNN accuracy is:", acc_knn)
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))
print(classification_report(y_test, y_pred))
