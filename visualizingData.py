import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils


# x = [1, 2, 3]
# y = [1, 10, 1]
#
# m = 0
#
# for i in range(3):
#     m += (x[i] * y[i])
#
# m /= 30
#
# print(m)

data_red = np.array(pd.read_csv("./winequality-red.csv", sep=';'));
X_red = data_red[:,0:-1]
Y_red = data_red[:,-1]

n, m = X_red.shape

# PLOT 1 figure
fig, ax = plt.subplots(2, 1)
fig.tight_layout(pad=2.0)

# PLOT 1.1: get frequency of ratings given
categories = [[] for i in range(10)]
for row in data_red:
    y = row[-1]
    i = round(y.astype(np.int64) - 1)
    categories[i].append(row)
ratings = [i+1 for i in range(10)]
num_ratings = [len(categories[i]) for i in range(10)]

plt.sca(ax[0])
plt.bar(ratings, num_ratings, width=0.5)
ax[0].set_title("Bar graph of ratings given")

# PLOT 1.2: just a sample histogram of the data
plt.sca(ax[1])
plt.hist(X_red[:,0])
ax[1].set_title("Histogram of the first input")

plt.show()

# PLOT 2 figure
fig, ax = plt.subplots(1, 1)

# PLOT 2.1: mean vs mean scaled by Y values
def getScaledMean(i):
    total = np.float64();
    num_elements = np.float64();
    for c in categories:
        for row in c:
            y = row[-1]
            total += (row[i] * y)
            num_elements += y
    return total / num_elements;

inputs = [i for i in range(11)]
unscaled_mean = np.array([np.mean(X_red[:,i]) for i in range(11)])
scaled_mean = np.array([getScaledMean(i) for i in range(11)])

plt.sca(ax)
plt.plot(inputs, unscaled_mean)
plt.plot(inputs, scaled_mean, color="orange")
ax.set_title("Mean vs Scaled Mean for each input")
plt.show()

# PLOT 3 figure
fig, ax = plt.subplots(1, 1)

# PLOT 3.1: graph the difference in percentage for each input
plt.sca(ax)
plt.plot(inputs, (scaled_mean / unscaled_mean) - 1)
ax.set_title("Percentage difference of Scaled Mean / Unscaled Mean for each input")
plt.show()
