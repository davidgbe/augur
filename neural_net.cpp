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
    std::cout << "here" << std::endl;
  }

  NeuralNet::~NeuralNet() {
    for(std::vector<Layer*>::iterator it = net.begin() ; it != net.end(); ++it) {
      it = net.erase(it);
    }
  }

  void NeuralNet::backpropagate_train(double* X, double* Y, int iterations) {

  }

  double NeuralNet::predict(double* X) {
    double* activations = X;
    for(int i = 0; i < num_levels; i++) {
      activations = net.at(i)->feed_forward(activations);
    }
    return activations[0];
  }


}
