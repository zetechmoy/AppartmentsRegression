import random
import matplotlib.pyplot as plt
import numpy as np

#create data like y = ax + b
#with some noise

nb_to_create = 2000

a_range = [23, 110]
b = 0

xs = []
ys = []

#generate x following normal law to create outliers
#N(moy sigma²) => np.random.randn(1) * sigma + moy
xs = np.random.randn(nb_to_create) * 400 + 10
xs = xs.tolist()

for i in range(0, len(xs)):
	if xs[i] < 0:
		xs[i] = -xs[i]

print(type(xs))

#generate y in function of x, with noise
for i in range(0, nb_to_create):
	a = abs(random.uniform(a_range[0], a_range[1]))
	y = a*xs[i] + b

	ys.append(y)
	print(str(a)+" * "+str(xs[i])+" + "+str(b)+" = "+str(y))

plt.plot(xs, ys, 'ro', markersize=4)
plt.show()

file = open("output.csv", "a")
file.seek(0)
file.truncate(0)
file.write("loyer,surface\n")

for i in range(0, len(xs)):
	file.write(str(ys[i])+","+str(xs[i])+"\n")
