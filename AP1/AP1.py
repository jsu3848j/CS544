# imports
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# read data into a DataFrame
data = pd.read_csv('auto.csv',names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name'], header=None,na_values=["?"])
data = data.dropna()

#Displacement 
feature_cols = ['displacement']
X = data[feature_cols]
y = data.mpg
# instantiate and fit
lm1 = LinearRegression()
lm1.fit(X, y)

# print the coefficients
print lm1.intercept_
print lm1.coef_
print lm1.score(X, y)

X_new = pd.DataFrame({'displacement': [data.displacement.min(), data.displacement.max()]})
preds = lm1.predict(X_new)
data.plot(kind='scatter', x='displacement', y='mpg')
plt.plot(X_new, preds, c='red', linewidth=2)
plt.savefig('displacement.png')

#horsepower
feature_cols = ['horsepower']
X = data[feature_cols]
y = data.mpg
# instantiate and fit
lm2 = LinearRegression()
lm2.fit(X, y)

# print the coefficients
print lm2.intercept_
print lm2.coef_
print lm2.score(X, y)

X_new = pd.DataFrame({'horsepower': [data.horsepower.min(), data.horsepower.max()]})
preds = lm2.predict(X_new)
data.plot(kind='scatter', x='horsepower', y='mpg')
plt.plot(X_new, preds, c='red', linewidth=2)
plt.savefig('horsepower.png')

#Weight
feature_cols = ['weight']
X = data[feature_cols]
y = data.mpg
# instantiate and fit
lm3 = LinearRegression()
lm3.fit(X, y)

# print the coefficients
print lm3.intercept_
print lm3.coef_
print lm3.score(X, y)

X_new = pd.DataFrame({'weight': [data.weight.min(), data.weight.max()]})
preds = lm3.predict(X_new)
data.plot(kind='scatter', x='weight', y='mpg')
plt.plot(X_new, preds, c='red', linewidth=2)
plt.savefig('weight.png')

#Acceleration
feature_cols = ['acceleration']
X = data[feature_cols]
y = data.mpg
# instantiate and fit
lm4 = LinearRegression()
lm4.fit(X, y)

# print the coefficients
print lm4.intercept_
print lm4.coef_
print lm4.score(X, y)

X_new = pd.DataFrame({'acceleration': [data.acceleration.min(), data.acceleration.max()]})
preds = lm4.predict(X_new)
data.plot(kind='scatter', x='acceleration', y='mpg')
plt.plot(X_new, preds, c='red', linewidth=2)
plt.savefig('acceleration.png')

#Combined
feature_cols = ['displacement','horsepower','weight','acceleration']
X = data[feature_cols]
y = data.mpg
# instantiate and fit
lm5 = LinearRegression()
lm5.fit(X, y)

# print the coefficients
print lm5.intercept_
print lm5.coef_
print lm5.score(X, y)

grid = sns.pairplot(data, x_vars=['displacement','horsepower','weight','acceleration'], y_vars='mpg', kind='reg')
grid.savefig('combined.png')

