#ifndef PERCEPTRON_H
#define PERCEPTRON_H

namespace augur {
  class Perceptron {
    public:
      Perceptron(int pos, int num_features);
      ~Perceptron();
      void train(double* X, int num_features, double* Y);
      void predict(double* activations, int num_activations, double* prediction);
      double* get_weights();
      double get_bias();
      int get_num_weights();
    private:
      void initialize_weights();
      double compute_activation(double* features);
      double transform(double activation);

      double* weights;
      int num_weights;
      double bias;

      double error;
      double* gradients;
      int position;
      double prediction;
  };
}

#endif
