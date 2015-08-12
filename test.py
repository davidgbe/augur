import augur
import numpy as np

X = np.array([1, 2, 3, 6, 2, 1, 7, 10, 11, 230, 11, 3, 55, 66, 5, 4, 2, 8, 9, 1], np.double)
# Y = np.array([1, -1, -1], np.double)

# X_test = np.array([[1, 2, 3, 4]], np.double)

augur.run(X)
