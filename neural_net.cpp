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
    double* predictions = NULL;
    net.at(0)->feed_forward(X, predictions);
    std::cout << predictions[0] << std::endl;
    // for(int i = 1; i < num_levels; i++) {

    // }
    return 1.0;
  }


}
