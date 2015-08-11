#include "perceptron.h"

#include <iostream>

namespace augur {

  Perceptron::Perceptron(int pos, int num_features) {
    weights = NULL;
    gradients = NULL;
    error = 0;
    position = pos;
    num_weights = num_features;
    initialize_weights();
    std::cout << "Perceptron initialized at position: ";
    std::cout << pos << std::endl;
  }

  Perceptron::~Perceptron() {
    if(weights != NULL) {
      delete[] weights;
    }
    // if(gradients != NULL) {
    //   delete[] gradients;
    // }
  }

  void Perceptron::train(double* X, int num_features, double* Y) {
    if(num_features != num_weights) {
      //throw an error
    }
    double prediction = compute_activation(X);
    double y = *Y;
    if(0 >= y * prediction) {
      for(int f_num = 0; f_num < num_weights; ++f_num) {
        weights[f_num] = weights[f_num] + (y * X[f_num]);
      }
      bias += y;
    }
  }

  void Perceptron::initialize_weights() {
    weights = new double[num_weights];
    for(int idx = 0; idx < num_weights; ++idx) {
      weights[idx] = std::rand() % 10;
      std::cout << weights[idx] << std::endl;
    }
    bias = 0;
  }

  void Perceptron::predict(double* activations, double* prediction) {
    *prediction = transform(compute_activation(activations));
  }

  double Perceptron::transform(double activation) {
    return activation;
  }

  double Perceptron::compute_activation(double* features) {
    double total = 0;
    for(int f_num = 0; f_num < num_weights; ++f_num) {
      total += ( features[f_num] * weights[f_num] );
    }
    return total + bias;
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
