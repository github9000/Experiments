
'''
   Perform regression on house-prices for Boston Housing dataset 
   Uses k-fold cross-validation.
   Calculates R-squared regression value
   Can compare output results with neural-net boston_*_nn.py programs 

   Usage:  python  boston_regression_rsq.py
   Output is R-squared and std.deviation values.
   
'''


# Cross Validation Regression R^2
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Init random seed to fixed value
seed = 3 


# Load dataset
filename = '../datasets/boston-housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values


# Split into X and Y variables
X = array[:,0:13]
Y = array[:,13]

# K-fold cross-validation
kfold = KFold(n_splits=10, random_state=seed)
model = LinearRegression()
scoring = 'r2'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

# Output r-squared value (explains variability)
print("R^2: {0:.3f}  {1:.3f}".format(results.mean(), results.std()))




