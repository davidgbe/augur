cimport numpy as np
from libcpp cimport bool
from cython.operator cimport dereference as deref
import numpy as np

cdef extern from "perceptron.h" namespace "augur":
    cdef cppclass Perceptron:
        Perceptron() except +
        void train(double* X, double* Y, int num_features, int num_examples, int num_iterations)
        void predict(double* feature_set, int num_sets, double* predictions)
        double* get_weights()
        int get_num_weights()
        double get_bias()

def run(np.ndarray[np.double_t, ndim=2] X, np.ndarray[np.double_t, ndim=1] Y, np.ndarray[np.double_t, ndim=2] X_test):
    cdef int iterations = 200
    X = np.ascontiguousarray(X)
    Y = np.ascontiguousarray(Y)
    X_test = np.ascontiguousarray(X_test)
    if X.shape[0] != Y.shape[0]:
        raise StandardError('Training label matrix must have same number of rows as feature matrix')
    cdef Perceptron p =  Perceptron()
    p.train(&X[0,0], &Y[0], X.shape[1], X.shape[0], iterations)

    cdef double* weights = p.get_weights()
    cdef int length = p.get_num_weights()
    for i in range(length):
        print weights[i]
    print p.get_bias()

    print 'predictions:'
    cdef np.ndarray[np.double_t, ndim=1] predictions = np.zeros(X.shape[1], dtype=np.double)
    predictions = np.ascontiguousarray(predictions)
    p.predict(&X[0,0], X.shape[1], &predictions[0])
    for i in range(X.shape[0]):
        print 'predict:'
        print predictions[i]
        print 'actual:'
        print Y[i]

