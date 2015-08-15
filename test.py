import augur
import numpy as np

X = np.array([.001, .01, .002, .0001, .01, .001, .01, .002, .0001, .01, 1.001, .01, .002, .0001, .01, .001, .01, .002, .0001, .01], np.double)
Y = np.array([1.0], np.double)
# Y = np.array([1, -1, -1], np.double)

# X_test = np.array([[1, 2, 3, 4]], np.double)

augur.run(X, Y, 50)
