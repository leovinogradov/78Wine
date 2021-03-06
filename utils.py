import numpy as np
from numpy import asarray as arr
from numpy import atleast_2d as twod
import matplotlib.pyplot as plt

def splitData(X, Y=None, train_fraction=0.80):
    """
    Split data into training and test data.

    Parameters
    ----------
    X : MxN numpy array of data to split
    Y : Mx1 numpy array of associated target values
    train_fraction : float, fraction of data used for training (default 80%)

    Returns
    -------
    to_return : (Xtr,Xte,Ytr,Yte) or (Xtr,Xte)
        A tuple containing the following arrays (in order): training
        data from X, testing data from X, training labels from Y
        (if Y contains data), and testing labels from Y (if Y
        contains data).
    """
    nx,dx = twod(X).shape
    ne = int(round(train_fraction * nx))

    Xtr,Xte = X[:ne,:], X[ne:,:]
    to_return = (Xtr,Xte)

    if Y is not None:
        Y = arr(Y).flatten()
        ny = len(Y)
        if ny > 0:
            assert ny == nx, 'splitData: X and Y must have the same length'
            Ytr,Yte = Y[:ne], Y[ne:]
            to_return += (Ytr,Yte)

    return to_return


def mse(A, B, ax=None):
    return (np.square(A - B)).mean(axis=ax)

# --- Just a test, does not work ---
curr_pos = 0
def showMultipagePlot(figs, axs):
    def key_event(e):
        global curr_pos

        if e.key == "right":
            curr_pos = curr_pos + 1
        elif e.key == "left":
            curr_pos = curr_pos - 1
        else:
            return
        curr_pos = curr_pos % 2

        axs[curr_pos].cla()
        # ax.plot(plots[curr_pos][0], plots[curr_pos][1])
        figs[curr_pos].canvas.draw()

    figs[curr_pos].canvas.mpl_connect('key_press_event', key_event)
    # ax = fig.add_subplot(111)
    # ax.plot(t,y1)
    plt.show()
