#ifndef LAYER_H
#define LAYER_H

#include "perceptron.h"

#include <vector>

namespace augur {

  class Layer {
    public:
      Layer(int number_of_nodes, int num_features, int level);
      ~Layer();
      double* feed_forward(double* activations, bool del_activations);
      static void* perceptron_predict(void* info);
      void set_inputs(double* in);
      static void* calculate_single_perceptron_error(void* thread_info);
      void calculate_perceptron_errors(Layer* child);
      void calculate_root_error(double y);

    private:
      int level;
      int num_nodes;
      int num_features;
      std::vector<Perceptron*> perceptrons;
      double* inputs;
  };
}

#endif
