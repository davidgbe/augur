#include <stddef.h>
#include <cstdlib>
#include <iostream>

#include "neural_net.h"
#include "layer.h"

namespace augur {

  NeuralNet::NeuralNet(int* struc, int num_lvls) {
    num_levels = num_lvls;
    int i = 0;
    for(i = 0; i < num_levels; ++i) {
      int num_nodes = (i != num_levels - 1) ? structure[i + 1] : 1;
      net.push_back( new Layer(num_nodes, structure[i], i) );
    }
  }

  NeuralNet::~NeuralNet() {
    for(std::vector<Layer*>::iterator it = net.begin() ; it != net.end(); ++it) {
      delete *it;
    }
  }

  void NeuralNet::backpropagate_train(double* X, double* Y, int iterations) {

  }

  double NeuralNet::predict(double* X) {
    return 1.0;
  }


}
