#ifndef LAYER_H
#define LAYER_H

#include "perceptron.h"

#include <vector>

namespace augur {

  class Layer {
    public:
      Layer(int number_of_nodes, int num_features, int level);
      ~Layer();
      double* feed_forward(double* activations);
      static void* perceptron_predict(void* info);

    private:
      int level;
      int num_nodes;
      int num_features;
      std::vector<Perceptron*> perceptrons;
  };
}

#endif
