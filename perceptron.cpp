#include "perceptron.h"

#include <iostream>
#include <math.h>

namespace augur {

  Perceptron::Perceptron(int pos, int num_features) {
    weights = NULL;
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
  }

  void Perceptron::initialize_weights() {
    weights = new double[num_weights];
    for(int idx = 0; idx < num_weights; ++idx) {
      weights[idx] = (std::rand() % 1000) / 1000.0;
    }
    bias = 0;
  }

  void Perceptron::predict(double* activations, double* prediction) {
    double p = transform(compute_activation(activations));
    *prediction = p;
  }

  double Perceptron::transform(double activation) {
    return tanh(activation);
  }

  double Perceptron::transform_derivative(double activation) {
    return 1 - pow(tanh(activation), 2.0);
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

  void Perceptron::generate_error_as_root(double* activations, double y) {
    error = y - compute_activation(activations);
    std::cout << "root error: " << error << std::endl;
  }

  void Perceptron::generate_error_as_parent(double* activations, std::vector<Perceptron*>* children) {
    double activation = compute_activation(activations);
    error = 0;
    for(int i = 0; i < children->size(); ++i) {
      Perceptron* child = children->at(i);
      double diff = (child->error * child->weights[position] * transform_derivative(activation));
      error += diff;
    }
    std::cout << "error: " << error << std::endl;
  }

  void Perceptron::update_weights(double* activations, double learning_rate) {
    for(int i = 0; i < num_weights; ++i) {
      weights[i] += (learning_rate * activations[i] * error);
    }
  }
}
