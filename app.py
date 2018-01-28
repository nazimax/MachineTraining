import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('diabetes.csv')
column = [ "Pregnancies",  "Glucose",  "BloodPressure",  "SkinThickness" , "Insulin",   "BMI", "DiabetesPedigreeFunction",  "Age"]

labels = df["Outcome"].values
features =df[list(column)].values
# print features


x_train, x_test,y_train,y_test=train_test_split(features,labels,test_size=0.3)

#print x_test
clf=RandomForestClassifier(n_estimators=1)
clf = clf.fit(x_train,y_train)
accuraacy=clf.score(x_train,y_train)

print "Training  model accurancy :\n ",accuraacy*100
accuraacy=clf.score(x_test,y_test)
print "Test model accurancy :\n",accuraacy*100


ypredict = clf.predict(x_train)
print '\n Training classification report \n', classification_report(y_train,ypredict)

print '\n Confusion matrix of training \n',confusion_matrix(y_train,ypredict)

ypredict = clf.predict(x_test)
print '\n Testing classification report \n', classification_report(y_test,ypredict)

print '\n Confusion matrix of training \n',confusion_matrix(y_test,ypredict)



