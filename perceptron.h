#ifndef PERCEPTRON_H
#define PERCEPTRON_H

namespace augur {
  class Perceptron {
    public:
      Perceptron();
      ~Perceptron();
      void train(double* X, double* Y, int num_features, int num_examples, int num_iterations);
      void predict(double* feature_set, int num_sets, double* predictions);
      double* get_weights();
      double get_bias();
      int get_num_weights();
    private:
      void initialize_weights(int num_features);
      double compute_activation(double* features);
      double* weights;
      int num_weights;
      double bias;
  };
}

#endif
