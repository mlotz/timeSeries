import numpy
import scipy
import matplotlib
import pandas
import sklearn
import tensorflow
import keras
import matplotlib.pyplot as plt

print("Hello, World!")

dataset = pandas.read_csv('dataset_PSE (1).csv', delimiter = ',')
#print(dataset.iloc[:,3])


Vds= dataset.iloc[:,3] # ZAP, raw, unstamped.
#visualize ZAP
plt.figure()
Vds.plot()
#plt.show()
plt.savefig('Dataset_ZAP.pdf')


#print(dataset)
