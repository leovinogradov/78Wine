import pandas as pd
import numpy as np
import nn
import utils

data = np.array(pd.read_csv("./winequality-red.csv", sep=';'));
X = data[:,0:-1]
Y = data[:,-1]


Xtr, Xte, Ytr, Yte = utils.splitData(X, Y, 0.9);

print("X and Y shapes =", X.shape, Y.shape)

results, estimator = nn.train(Xtr, Ytr)

print(results)

estimator.fit(Xtr, Ytr)
Yhat = estimator.predict(Xte)

mse = utils.mse(Yte, Yhat)

print("mse on testing data is", mse)

"""
BEST RESULTS SO FAR:

neg_root_mean_squared_error: Mean = -0.67 Std = 0.02

mse on testing data is 0.524910799519418

"""

from multiprocessing import Pool
import yourFunctionDefs

numProcesses = 3 # or however many you need

p = Pool(processes=numProcesses)
results = []

for i in range(numProcesses):
    res = p.apply_async(yourFunctionDefs.yourFunc, (arg_a, arg_b, etc, i)) # call your training function with your arguments
    results.append(res)
    print("started getting result for", i)

for j, res in enumerate(results):
    result = res.get() # get results asynchronously
    print("got result", result)

p.close()

"""
