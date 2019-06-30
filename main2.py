# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split

from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import pandas

def getModel(input_shape, output_shape, nb_node_p_layer, nb_layer):
	model = Sequential()

	model.add(Dense(input_shape))
	for _ in range(0, nb_layer):
		model.add(Dense(nb_node_p_layer))
	model.add(Dense(output_shape))
	model.compile(loss='mse', optimizer='rmsprop', metrics=['mse'])
	#model.summary()
	return model

def getSettingedModel():
	return getModel(input_shape, output_shape, nb_node_p_layer, nb_layer)

# define the model
def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# On charge le dataset
house_data = pd.read_csv('house.csv')

# On affiche le nuage de points dont on dispose
#plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)
#plt.show()


dataframe = pandas.read_csv("house2.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
xl = dataset[:,0:13]
yl = dataset[:,13]

print(xl[0])
print(yl[0])

#x, y = house_data['loyer'], house_data['surface']
#x, y = x.tolist(), y.tolist()


#xl = [[i] for i in x]
yl = [[i] for i in yl]

# On affiche le nuage de points dont on dispose
#plt.plot(x, y, 'ro', markersize=4)
#plt.show()

sc = StandardScaler()

x = sc.fit_transform(xl)
y = sc.fit_transform(yl)

print(xl[0])
print(sc.transform([xl[0]]))
print(sc.inverse_transform(sc.transform([xl[0]])))

# On affiche le nuage de points dont on dispose
#plt.plot(x, y, 'ro', markersize=4)
#plt.show()

n_features = len(x)

input_shape = len(xl[0])
output_shape = len(yl[0])
nb_node_p_layer = 30
nb_layer = 2

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=larger_model, epochs=100, batch_size=5, verbose=1)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, x_test, y_test, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))