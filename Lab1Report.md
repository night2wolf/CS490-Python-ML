# CS490-Python-ML
Lab 01 Report 

Team:

	Trevor Klinkenberg (24)
	Landon Vollkmann (12)
Video Link: https://youtu.be/DMUXmS61z9A

# Introduction
	This project was implemented to fulfill the lab requirement for UMKC’s CS 490 - Introduction to Deep Learning with Python course. The scope of this project covers basic functionality in Python, simple problem solving abilities, and introductory machine learning concepts such as Classification, Regression, and Natural Language Processing.
  
# Objectives
	The objectives of this project were educational in nature. The sole purpose of this project is to demonstrate the participant’s competence in Python, problem solving, and some of the introductory machine learning techniques mentioned above.
  
# Approaches/Methods
	This project invoked a selection of machine learning techniques. These techniques include:
Natural Language Processing via the NLTK library
Classifiers
SVM
Naive Bayes
K Neighbors
Clustering
K Means
Multiple Regression with R^2 and RMSE

# Workflow
	The team tackled this project with a divide and conquer approach. Given how small and discrete each problem was, the tasks were split evenly and tackled alone. The group members then conducted a light code review before submission.
  
# Datasets 
Web Scraping: https://catalog.umkc.edu/course-offerings/graduate/comp-sci/
NLP: https://umkc.app.box.com/s/7by0f4540cdbdp3pm60h5fxxffefsvrw
Classification: https://umkc.app.box.com/s/lanhyin9ysil5r0lywitqylytxfxtzp6
Clustering/MultipleRegression: https://archive.ics.uci.edu/ml/machine-learning-databases/housing/ 
 
# Parameters
 
Classification:
	Feature points with a z-score > 3 were deemed as outliers
	Test/Train Split: 0.3/0.7
Elbow Curve: 3
Dictionaries: {'a':1,'b':2,'c':9,'d':4} , {'foo':7,'bar':5,'contoso':6}
MR Test/Train Split:  33Test 67Train
Subset Input: [1, 2, 2]
Flights : 
KC to Chicago, 2020/02/22 9:00, 10:00
	KC to NYC, 2020/02/22 9:00, 12:00
Employees:
	Tina GoodLady, Attendant
	Kevin Twinkle, Attendant
	Sully Something, Pilot
Passengers:
	Trevor Klinkenberg,10B
	Brady Volkmann, 12C
	Landon Volkman, 22A
  
# Evaluation & Discussion
Clustering Findings:
	K- Means Silhouette Score: 0.7206527282357368
Regression Findings:
	R^2 is: 0.9473926938136905
RMSE is: 0.008440395970796327
Classification Findings:
	With some light preprocessing such as filling nulls with averages, removing outliers, and dropping low correlated features, the following was found:
  
Naive bayes accuracy is: 75.95

Number of mislabeled points out of a total 262 points : 63

              precision    recall  f1-score   support
           0       0.82      0.77      0.80       159
           1       0.68      0.74      0.71       103
    accuracy                           0.76       262
   macro avg       0.75      0.76      0.75       262
weighted avg       0.76      0.76      0.76       262
 
svm accuracy is: 67.56

Number of mislabeled points out of a total 262 points : 85

              precision    recall  f1-score   support
 
           0       0.67      0.92      0.77       159
           1       0.70      0.30      0.42       103
    accuracy                           0.68       262
   macro avg       0.69      0.61      0.60       262
weighted avg       0.68      0.68      0.64       262
 
KNN accuracy is: 85.06

Number of mislabeled points out of a total 262 points : 82

              precision    recall  f1-score   support
           0       0.73      0.77      0.75       159
           1       0.61      0.56      0.59       103
    accuracy                           0.69       262
   macro avg       0.67      0.67      0.67       262
weighted avg       0.68      0.69      0.68       262
 
# Conclusion

If anything can be concluded from this series of disjoint problems, it’s that the world of machine learning is incredibly vast and continually expanding. We’ve only touched the tip of the iceberg in regard to the sheer number of approaches there are to these problems. Whatsmore, different preprocessing and sanitization techniques may yield different results even with our tested approaches.


