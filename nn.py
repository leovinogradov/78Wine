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

    # cv: int, cross-validation generator or an iterable
    # n_jobs: int or None, optional (default=None)
    #   The number of CPUs to use to do the computation.
    #   None means 1 unless in a joblib.parallel_backend context.
    #   -1 means using all processors. See Glossary for more details.
    results = cross_val_score(estimator, X, Y, cv=kfold, scoring='neg_root_mean_squared_error', n_jobs=-1)

    print("\nEnded training.")
    print("neg_root_mean_squared_error: Mean = %.2f Std = %.2f" % (results.mean(), results.std()))
    return (results, estimator,)
