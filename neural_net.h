#ifndef NEURAL_NET_H
#define NUERAL_NET_H

#include "layer.h"

namespace augur {
  class NeuralNet {
    public:
      NeuralNet(int* structure, int num_levels);
      ~NeuralNet();
      void backpropagate_train(double* X, double* Y, int iterations);
      double predict(double* X);

    private:
      std::string name;
      int* stucture;
      int num_levels;
      std::vector<Layer*> net;
  };
}

#endif
