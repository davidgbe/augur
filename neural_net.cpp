#include <stddef.h>
#include <cstdlib>
#include <iostream>

#include "neural_net.h"

namespace augur {

  NeuralNet::NeuralNet(int* structure, int num_levels) {
    for(int i = 0; i < num_levels; ++i) {
      
    }
  }

  NeuralNet::~NeuralNet() {

  }

  void NeuralNet::backpropagate_train(double* X, double* Y, int iterations) {

  }

  double NeuralNet::predict(double* X) {
    return 1.0;
  }


}
