#include <stddef.h>
#include <cstdlib>
#include <iostream>

#include "perceptron.h"

namespace augur {

  Perceptron::Perceptron() {
    weights = NULL;
    error = 0;
  }

  Perceptron::~Perceptron() {
    if(weights != NULL) {
      delete[] weights;
    }

    // if(gradients != NULL) {
    //   delete[] gradients;
    // }
  }

  void Perceptron::train(double* X, double* Y, int num_features, int num_examples, int num_iterations) {
    initialize_weights(num_features);
    for(int i = 0; i < num_iterations; ++i) {
      for(int ex_num = 0; ex_num < num_examples; ++ex_num) {
        double* features_ptr = X + ex_num * num_features;
        double prediction = compute_activation(features_ptr);
        double y = Y[ex_num];
        if(0 >= y * prediction) {
          for(int f_num = 0; f_num < num_features; ++f_num) {
            weights[f_num] = weights[f_num] + (y * features_ptr[f_num]);
          }
          bias += y;
        }
      }
    }
  }

  void Perceptron::initialize_weights(int num_features) {
    num_weights = num_features;
    weights = new double[num_weights];
    for(int idx = 0; idx < num_weights; ++idx) {
      weights[idx] = std::rand() % 10;
      std::cout << weights[idx] << std::endl;
    }
    bias = 0;
  }

  void Perceptron::predict(double* feature_set, int num_sets, double* predictions) {
    if(!num_weights) {
      //throw error
    }
    for(int idx = 0; idx < num_sets; ++idx) {
      predictions[idx] = compute_activation(feature_set + idx * num_weights);
    }
  }

  double Perceptron::compute_activation(double* features) {
    double total = 0;
    for(int f_num = 0; f_num < num_weights; ++f_num) {
      total += ( features[f_num] * weights[f_num] );
    }
    total += bias;
    if(total >= 0) {
      return 1.0;
    } else {
      return -1.0;
    }
  }

  double* Perceptron::get_weights() {
    return weights;
  }

  int Perceptron::get_num_weights() {
    return num_weights;
  }

  double Perceptron::get_bias() {
    return bias;
  }
}
