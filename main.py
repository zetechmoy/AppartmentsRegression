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


def getModel(input_shape, output_shape, nb_node_p_layer, nb_layer):
	model = Sequential()

	model.add(Dense(input_shape))
	for _ in range(0, nb_layer):
		model.add(Dense(nb_node_p_layer))
	model.add(Dense(output_shape))
	model.compile(loss='mse', optimizer='rmsprop', metrics=['mse'])
	#model.summary()
	return model

# On charge le dataset
house_data = pd.read_csv('output.csv')

# On affiche le nuage de points dont on dispose
#plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)
#plt.show()

x, y = house_data['loyer'], house_data['surface']
x, y = x.tolist(), y.tolist()

x = [[i] for i in x]
y = [[i] for i in y]

n_features = len(x)

input_shape = len(x[0])
output_shape = len(y[0])
nb_node_p_layer = 3
nb_layer = 50

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

model = getModel(input_shape, output_shape, nb_node_p_layer, nb_layer)

print("Fitting ...")
for learning_year in range(1, 10):
	print("#"+str(learning_year))
	model.fit(x_train, y_train, epochs=10, batch_size=4, callbacks=[])
	# evaluate the model
	scores = model.evaluate(x_test, y_test)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	for i in range(0, 10):
		index = random.randint(0, len(x_test)-1)
		data = x_test[index]
		correct = y_test[index]
		res = model.predict(np.array([data]))
		print('D', data[0], end=' ')
		print('C', correct[0], end=' ')
		print('P', res[0][0])

scores = model.evaluate(x_test, y_test)
model.save("apparts"+str(n_features)+"_"+str(nb_node_p_layer)+"_"+str(int(scores[1]*100))+"_v1.h5")
