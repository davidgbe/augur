#include <stddef.h>
#include <cstdlib>
#include <iostream>

#include "neural_net.h"

namespace augur {

  NeuralNet::NeuralNet() {

  }

  NeuralNet::NeuralNet(int* structure, int num_lvls) {
    num_levels = num_lvls;
    for(int i = 0; i < num_levels; ++i) {
      std::cout << i << std::endl;
      net.push_back( new Layer(structure[i + 1], structure[i], i) );
    }
    std::cout << "Finished initializing NeuralNet" << std::endl;
  }

  NeuralNet::~NeuralNet() {
    // for(int i = 0; i < num_levels; ++i) {
    //   delete net.at(i);
    // }
  }

  double NeuralNet::forward_propagate(double* X, bool store_inputs) {
    double* activations = X;
    for(int i = 0; i < num_levels; ++i) {
      if(store_inputs) {
        net.at(i)->set_inputs(activations);
        activations = net.at(i)->feed_forward(activations, false);
      } else {
        activations = net.at(i)->feed_forward(activations, true);
      }
    }
    return activations[0];
  }

  void NeuralNet::backpropagate_train(double* X, double* Y, int iterations) {
    for(int i = 0; i < iterations; ++i) {
      std::cout << "guess: " << forward_propagate(X, true) << std::endl;
      std::cout << "answer " << Y[0] << std::endl;
      generate_errors(Y[0]);
      update_all_weights(.01);
    }
  }

  double NeuralNet::predict(double* X) {
    return forward_propagate(X, false);
  }

  void NeuralNet::generate_errors(double y) {
    net.at(num_levels - 1)->calculate_root_error(y);
    if(num_levels > 1) {
      for(int i = num_levels - 2; i >= 0; --i) {
        net.at(i)->calculate_perceptron_errors( net.at(i + 1) );
      }
    }
  }

  void NeuralNet::update_all_weights(double learning_rate) {
    for(int i = 0; i < net.size(); ++i) {
      net.at(i)->update_perceptron_weights(learning_rate);
    }
  }
}
