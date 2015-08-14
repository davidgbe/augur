from libcpp cimport bool
from cython.operator cimport dereference as deref
import numpy as np
cimport numpy as np
import ctypes

cdef extern from "neural_net.h" namespace "augur":
    cdef cppclass NeuralNet:
        NeuralNet() except +
        NeuralNet(int* structure, int num_levels) except +
        void backpropagate_train(double* X, double* Y, int iterations)
        double predict(double* X)
        void generate_errors(double y)

def run(np.ndarray[np.double_t, ndim=1] X, np.ndarray[np.double_t, ndim=1] Y):
    cdef int iterations = 200
    X = np.ascontiguousarray(X)
    Y = np.ascontiguousarray(Y)
    # Y = np.ascontiguousarray(Y)
    # X_test = np.ascontiguousarray(X_test)
    # if X.shape[0] != Y.shape[0]:
    #     raise StandardError('Training label matrix must have same number of rows as feature matrix')
    cdef np.ndarray[int, ndim=1, mode="c"] structure = np.ascontiguousarray( np.array([20, 30, 20, 1], dtype=ctypes.c_int) )
    cdef NeuralNet nn =  NeuralNet(&structure[0], 3)
    print nn.backpropagate_train(&X[0], &Y[0], 1)


    # print 'predictions:'
    # cdef np.ndarray[np.double_t, ndim=1] predictions = np.zeros(X.shape[1], dtype=np.double)
    # predictions = np.ascontiguousarray(predictions)
    # p.predict(&X[0,0], X.shape[1], &predictions[0])
    # for i in range(X.shape[0]):
    #     print 'predict:'
    #     print predictions[i]
    #     print 'actual:'
    #     print Y[i]

