import augur
import numpy as np

X = np.array([[1, 2, 3, 3], [1, 3, 4, 5], [2, 4, 3, 2]], np.double)
Y = np.array([1, -1, -1], np.double)

X_test = np.array([[1, 2, 3, 4]], np.double)

augur.run(X, Y, X_test)
