# imports
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#from sklearn.cross_validation import train_test_split
#import numpy as np

# read data into a DataFrame
data = pd.read_csv('auto.csv',names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name'], header=None,na_values=["?"])
data = data.dropna()
feature_cols = ['displacement']
X = data[feature_cols]
y = data.mpg
# instantiate and fit
lm = LinearRegression()
lm.fit(X, y)

# print the coefficients
print lm.intercept_
print lm.coef_
lm.predict(data.displacement.min(), data.displacement.max())
print lm.predict(data.displacement.min(), data.displacement.max())
data.plot(kind='scatter', x='displacement', y='mpg')
plt.plot(X_new, preds, c='red', linewidth=2)
plt.savefig('test.png')



