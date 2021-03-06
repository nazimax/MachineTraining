import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model

df = pd.read_csv('kc_house_data.csv')

# to get missing values
# print df.info()
# print df.isnull().sum()


# labels the thing that we want to predict

df = df.drop(["id", 'date', "lat", "long"], axis=1)
# print df.head()

plt.figure(figsize=(48, 6))
sb.stripplot(x="bedrooms", y="price", data=df)

labels = df['price'].values
column = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
          "waterfront", "view", "condition", "grade",
          "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
          "zipcode", "sqft_living15", "sqft_lot15"]

features = df[list(column)].values
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)


regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
accuracy = regr.score(x_train, y_train)
print 'training model accuracy :\n', accuracy
accuracy = regr.score(x_test, y_test)
print 'testing model accuracy :\n', accuracy
