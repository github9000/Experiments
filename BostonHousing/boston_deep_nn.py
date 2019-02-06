
'''
   Perform regression on house-prices for Boston Housing dataset 
   Defines a standard 3-layer net using Keras. 
   Can compare output results with wider boston_wide_nn.py  

   Uses ADAM optimizer and k-fold cross-validation.
   Using the mean-squared error (MSE) loss function
   gives us the error in US Dollars.

   Usage:  python  boston_deep_nn.py
   Output is MSE and std.deviation values.
   
'''

import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Init random seed to fixed value
seed = 3
numpy.random.seed(seed)

# Load the dataset
dataframe = read_csv("../datasets/boston-housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values


# Split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]

# Define the model
def larger_model():
	# Create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, activation="relu", kernel_initializer="normal"))
	model.add(Dense(6, activation="relu", kernel_initializer="normal"))
	model.add(Dense(1, kernel_initializer="normal"))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


# Evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, nb_epoch=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Deeper 3-Layer NN: %.2f (%.2f) MSE" % (results.mean(), results.std()))



