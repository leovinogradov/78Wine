import pandas as pd
import numpy as np
import nn

data = np.array(pd.read_csv("./winequality-red.csv", sep=';'));
X = data[:,0:-1]
Y = data[:,-1]




print("X and Y shapes =", X.shape, Y.shape)

results, estimator = nn.train(X, Y)

print(results)


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
