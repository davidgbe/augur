#ifndef LAYER_H
#define LAYER_H

#include "perceptron.h"

namespace augur {
  class Layer {
    public:
      Layer(int number_of_nodes, int num_features, int level);
      ~Layer();
      double* feed_forward(double* activations, int num_activations);

    private:
      void perceptron_predict(Perceptron* target, double* activations, int num_activations, double* prediction);
      int level;
      int num_nodes;
      int num_features;
      std::vector<Perceptron*> perceptrons;
  };
}

#endif
