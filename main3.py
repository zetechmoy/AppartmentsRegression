
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
	model.compile(loss='mse', optimizer='adam', metrics=['mse'])
	#model.summary()
	return model

dataframe = pandas.read_csv("house2.csv", delim_whitespace=True, header=None)
dataset = dataframe.values

xl = dataset[:,0:13]
yl = dataset[:,13]
yl = [[i] for i in yl]

sc = StandardScaler()

x = sc.fit_transform(xl)
y = sc.fit_transform(yl)

n_features = len(x)

input_shape = len(x[0])
output_shape = len(y[0])
nb_node_p_layer = 16
nb_layer = 1

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = getModel(input_shape, output_shape, nb_node_p_layer, nb_layer)

print("Fitting ...")
for learning_year in range(1, 10):
	print("#"+str(learning_year))
	model.fit(x_train, y_train, epochs=10, batch_size=4, callbacks=[])
	# evaluate the model
	scores = model.evaluate(x_test, y_test)
	print("\n%s: %.2f" % (model.metrics_names[1], scores[1]*100))

	for i in range(0, 10):
		index = random.randint(0, len(x_test)-1)
		data = x_test[index]
		correct = y_test[index]
		res = model.predict(np.array([data]))
		#print('D', sc.inverse_transform(data), end=' ')
		print('C', sc.inverse_transform(correct)[0], end=' ')
		print('P', sc.inverse_transform(res[0])[0])