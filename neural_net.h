#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "layer.h"

#include <vector>
#include <string>

namespace augur {
  class NeuralNet {
    public:
      NeuralNet();
      NeuralNet(int* structure, int num_levels);
      ~NeuralNet();
      void backpropagate_train(double* X, double* Y, int iterations);
      double predict(double* X);
      void generate_errors(double y);
    private:
      void forward_propagate(double* X, bool store_inputs);

      std::string name;
      int num_levels;
      std::vector<Layer*> net;
  };
}

#endif
