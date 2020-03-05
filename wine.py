import pandas as pd
import numpy as np
import nn

data = np.array(pd.read_csv("./winequality-red.csv", sep=';'));
X = data[:,0:-1]
Y = data[:,-1]




print("X and Y shapes =", X.shape, Y.shape)

model == nn.train(X, Y)
_, accuracy = model.evaluate(X, Y)

print('Accuracy: %.2f' % (accuracy*100))
