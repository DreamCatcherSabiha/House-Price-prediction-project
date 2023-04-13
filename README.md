# House-Price-prediction-project
Machine Learning Project
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 19:38:49 2022

@author: SABIHA
       
"""

import pandas as pd
housing = pd.read_csv("F:\datasets\housing_boston.csv")
housing.head()
housing.info ()
housing['chas'].value_counts()
housing.describe()

#%matplotlib inline
# # For plotting histogram
import matplotlib.pyplot as plt
 housing.hist(bins=50, figsize=(20, 15))
 
 #TEST TRAIN SPLITTING
  # For learning purpose
 import numpy as np
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:] 
    return data.iloc[train_indices], data.iloc[test_indices]
from sklearn.model_selection import train_test_split
train_set, test_set  = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")

#for actual split with equal no. of test pattern and train pattern ,so that like in chas we get proper 0 & 1 to both test and train set
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['chas']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_test_set['chas'].value_counts()
strat_train_set['chas'].value_counts()

#Check both train and test ratio
95/7
376/28
# 95/7
#Output 13.571428571428571

#376/28
#Output: 13.428571428571429

#LOOKING FOR CORRELATION 
corr_matrix = housing.corr()
corr_matrix

# GRAPHS ANALYSIS

from pandas.plotting import scatter_matrix
attributes=["medv","rm","zn","lstat"]
scatter_matrix(housing[attributes],figsize=(10,6))

corr_matrix
housing.plot(kind="scatter", x="rm", y="medv", alpha=0.8)#most suitable relation we get
housing.plot(kind="scatter", x="zn", y="medv", alpha=0.8)
housing.plot(kind="scatter",x="chas", y="medv", alpha=0.8)
housing.plot(kind="scatter",x="crim",y="medv", alpha= 0.8)
housing.plot(kind="scatter",x="indus",y="medv", alpha= 0.8)
housing.plot(kind="scatter",x="lstat",y="medv", alpha= 0.8)

#histogram plot
housing.hist(figsize=(20,15))

import seaborn as sns
sns.distplot(housing['chas'])
print("The skewness of chas is {}".format(housing['chas'].skew()))

sns.distplot(housing['rm'])
print("The skewness of rm is {}".format(housing['rm'].skew()))

sns.distplot(housing['crim'])
print("The skewness of scatter is {}".format(housing['crim'].skew()))
#heat maplot
import matplotlib.pyplot as plt #plotting,  visualizing
from sklearn import model_selection
import seaborn as sns
corr_matrix = housing.corr()
fig=plt.figure(figsize=(12,9))
sns.heatmap(corr_matrix,vmax=.8,square=True)

#TRYING OUT ATTRIBUTE COMBINATION

housing.head()

corr_matrix = housing.corr()
corr_matrix['medv'].sort_values(ascending=False)

housing = strat_train_set.drop("medv", axis=1)
housing_labels = strat_train_set["medv"].copy()

housing_labels

#MISSING ATTRIBUTES

# To take care of missing attributes, you have three options:
#   1. Get rid of the missing data points
#    2. Get rid of the whole attribute
#     3. Set the value to some value(0, mean or median)
a = housing.dropna(subset=["rm"]) #Option 1
a.shape
# Note that the original housing dataframe will remain unchanged

housing.drop("rm", axis=1).shape # Option 2
# Note that there is no RM column and also note that the original housing dataframe will remain unchanged

median = housing["rm"].median() # Compute median for Option 3
median

housing["rm"].fillna(median) # Option 3
# Note that the original housing dataframe will remain unchanged

housing.shape
housing.describe()# before we started filling missing attributes

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)

imputer.statistics_



X = imputer.transform(housing)

housing_tr = pd.DataFrame(X, columns=housing.columns)
housing_tr.describe

#Scikit-learn Design¶
#Primarily, three types of objects

#1)Estimators - It estimates some parameter based on a dataset. Eg. imputer. It has a fit method and transform method. Fit method - Fits the dataset and calculates internal parameters

#2)Transformers - transform method takes input and returns output based on the learnings from fit(). It also has a convenience function called fit_transform() which fits and then transforms.

#3)Predictors - LinearRegression model is an example of predictor. fit() and predict() are two common functions. It also gives score() function which will evaluate the predictions.


#Feature Scaling
#Primarily, two types of feature scaling methods:

#1)Min-max scaling (Normalization) (value - min)/(max - min) Sklearn provides a class called MinMaxScaler for this

#2)Standardization (value - mean)/std Sklearn provides a class called StandardScaler for this

#CREATING PIPELINE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])


Code
Dragon Real Estate - Price Predictor
import pandas as pd
housing = pd.read_csv("F:\datasets\housing_boston.csv")
housing.head()
housing['chas'].value_counts()
#0    471
#1     35
housing.describe()

%matplotlib inline
# # For plotting histogram
# import matplotlib.pyplot as plt
# housing.hist(bins=50, figsize=(20, 15))
Train-Test Splitting
# For learning purpose
import numpy as np
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:] 
    return data.iloc[train_indices], data.iloc[test_indices]

# train_set, test_set = split_train_test(housing, 0.2)
# print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")
from sklearn.model_selection import train_test_split
train_set, test_set  = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")

#for actual split with equal no. of test pattern and train pattern ,so that like in chas we get proper 0 #& 1 to both test and train set
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['chas']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
strat_test_set['chas'].value_counts()
strat_train_set['chas'].value_counts()


# 95/7
# 376/28
housing = strat_train_set.copy()

#LOOKING FOR CORRELATION
corr_matrix = housing.corr()

from pandas.plotting import scatter_matrix
attributes=["medv","rm","zn","lstat"]
scatter_matrix(housing[attributes],figsize=(10,6))

# from pandas.plotting import scatter_matrix
# attributes = ["MEDV", "RM", "ZN", "LSTAT"]
# scatter_matrix(housing[attributes], figsize = (12,8))
corr_matrix

#GRAPH ANALYSIS
housing.plot(kind="scatter", x="rm", y="medv", alpha=0.8)

housing.hist(figsize=(20,15))
import seaborn as sns
sns.distplot(housing['chas'])
print("The skewness of chas is {}".format(housing['chas'].skew()))


The skewness of chas is 3.404265772962613

sns.distplot(housing['rm'])
print("The skewness of rm is {}".format(housing['rm'].skew()))


The skewness of rm is 0.326670699060074

sns.distplot(housing['crim'])
print("The skewness of scatter is {}".format(housing['crim'].skew()))


The skewness of scatter is 4.649931610435271

#heat maplot
import matplotlib.pyplot as plt#plotting, visvualizing
from sklearn import model_selection


import seaborn as sns
corr_matrix = housing.corr()
fig=plt.figure(figsize=(12,9))
sns.heatmap(corr_matrix,vmax=.8,square=True)

#Trying out Attribute combinations
housing["tax"] = housing['tax']/housing['rm']
housing.head()
corr_matrix = housing.corr()
corr_matrix['medv'].sort_values(ascending=False)
housing.plot(kind="scatter", x="tax", y="medv", alpha=0.8)

housing = strat_train_set.drop("medv", axis=1)
housing_labels = strat_train_set["medv"].copy()
Missing Attributes
# To take care of missing attributes, you have three options:
#     1. Get rid of the missing data points
#     2. Get rid of the whole attribute
#     3. Set the value to some value(0, mean or median)
a = housing.dropna(subset=["rm"]) #Option 1
a.shape
# Note that the original housing dataframe will remain unchanged
(404, 13)
housing.drop("rm", axis=1).shape # Option 2
# Note that there is no RM column and also note that the original housing dataframe will remain unchanged
(404, 12)
median = housing["rm"].median() # Compute median for Option 3
housing["rm"].fillna(median) # Option 3
# Note that the original housing dataframe will remain unchanged
housing["rm"].fillna(median) # Option 3
housing.shape
housing.shape
(404, 13)
# before we started filling missing attributes
housing.describe() # before we started filling missing attribute
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)
SimpleImputer(copy=True, fill_value=None, missing_values=nan,
       strategy='median', verbose=0)
imputer.statistics_
imputer.statistics_
X = imputer.transform(housing)
X = imputer.transform(housing)
housing_tr = pd.DataFrame(X, columns=housing.columns)
housing_tr = pd.DataFrame(X, columns=housing.columns)
housing_tr.describe()
Scikit-learn Design
Primarily, three types of objects

Estimators - It estimates some parameter based on a dataset. Eg. imputer. It has a fit method and transform method. Fit method - Fits the dataset and calculates internal parameters

Transformers - transform method takes input and returns output based on the learnings from fit(). It also has a convenience function called fit_transform() which fits and then transforms.

Predictors - LinearRegression model is an example of predictor. fit() and predict() are two common functions. It also gives score() function which will evaluate the predictions.

Feature Scaling
Primarily, two types of feature scaling methods:

Min-max scaling (Normalization) (value - min)/(max - min) Sklearn provides a class called MinMaxScaler for this

Standardization (value - mean)/std Sklearn provides a class called StandardScaler for this

Creating a Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])

housing_num_tr = my_pipeline.fit_transform(housing)
housing_num_tr.shape

#Selecting a desired model for Dragon Real Estates
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]

prepared_data = my_pipeline.transform(some_data)
model.predict(prepared_data)

list(some_labels)

#Evaluating the model
from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)
rmse

#Using better evaluation technique - Cross Validation¶

# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
rmse_scores

def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())

print_scores(rmse_scores)

#Testing the model on test data
X_test = strat_test_set.drop("medv", axis=1)
Y_test = strat_test_set["medv"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse
# print(final_predictions,list(Y_test))

prepared_data[0]	

