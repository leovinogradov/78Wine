from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def baseline_model():
    # create the model
    model = Sequential()
    model.add(Dense(12, input_dim=11, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(8, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(1, kernel_initializer='normal'))

    # compile the keras model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model

def train(X, Y, epochs=100, nSplits=5):
    print("\nStarting training...")
    estimator = KerasRegressor(build_fn=baseline_model, epochs=epochs, batch_size=5, verbose=0)
    kfold = KFold(n_splits=nSplits)
    results = cross_val_score(estimator, X, Y, cv=kfold)
    print("\nEnded training.")
    print("Baseline: Mean = %.2f Std = %.2f MSE" % (results.mean(), results.std()))
    return (results, estimator,)
