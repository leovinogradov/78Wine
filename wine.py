import pandas as pd
import numpy as np
import nn
import utils

def runTests(X, Y):
    Xtr, Xte, Ytr, Yte = utils.splitData(X, Y, 0.9);

    print("X and Y shapes =", X.shape, Y.shape)

    results, estimator = nn.train(Xtr, Ytr)

    print(results)

    estimator.fit(Xtr, Ytr)
    Yhat = estimator.predict(Xte)

    mse = utils.mse(Yte, Yhat)

    print("mse on testing data is", mse)

    return (results, mse,)



data_red = np.array(pd.read_csv("./winequality-red.csv", sep=';'));
X_red = data_red[:,0:-1]
Y_red = data_red[:,-1]

cross_val_red, mse_red = runTests(X_red, Y_red)



data_white = np.array(pd.read_csv("./winequality-white.csv", sep=';'));
X_white = data_white[:,0:-1]
Y_white = data_white[:,-1]

cross_val_white, mse_white = runTests(X_white, Y_white)


print("\n\nResults:\n\n")

print("Red Wine:")
print(cross_val_red)
print(mse_red)
print()
print("White Wine:")
print(cross_val_white)
print(mse_white)

"""
BEST RESULTS SO FAR:

neg_root_mean_squared_error: Mean = -0.67 Std = 0.02

mse on testing data is 0.524910799519418

"""
