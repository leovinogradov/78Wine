import pandas as pd
import numpy as np

data = np.array(pd.read_csv("./winequality-red.csv", sep=';'));

print("shape =", data.shape)
