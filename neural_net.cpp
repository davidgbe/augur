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
      int num_nodes = (i != num_levels - 1) ? structure[i + 1] : 1;
      net.push_back( new Layer(num_nodes, structure[i], i) );
    }
  }

  NeuralNet::~NeuralNet() {
    for(std::vector<Layer*>::iterator it = net.begin() ; it != net.end(); ++it) {
      it = net.erase(it);
    }
  }

  double NeuralNet::forward_propagate(double* X, bool store_inputs) {
    double* activations = X;
    for(int i = 0; i < num_levels; i++) {
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

  }

  double NeuralNet::predict(double* X) {
    return forward_propagate(X, false);
  }

  void NeuralNet::generate_errors(double y) {
    net.at(num_levels - 1)->calculate_root_error(y);
    for(int i = num_levels - 1; i >= 0; ++i) {
      net.at(i)->calculate_perceptron_errors( net.at(i + 1) );
    }
  }


}
