#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>

namespace augur {
  class Perceptron {
    public:
      Perceptron(int pos, int num_features);
      ~Perceptron();
      void predict(double* activations, double* prediction);
      double* get_weights();
      double get_bias();
      int get_num_weights();
      void generate_error_as_root(double* activations, double y);
      void generate_error_as_parent(double* activations, std::vector<Perceptron*>* children);
    private:
      void initialize_weights();
      double compute_activation(double* features);
      double transform(double activation);

      double* weights;
      int num_weights;
      double bias;
      double error;
      int position;
  };
}

#endif
