#ifndef PERCEPTRON_H
#define PERCEPTRON_H

namespace augur {
  class Perceptron {
    public:
      Perceptron(int pos);
      ~Perceptron();
      void train(double* X, double* Y, int num_features, int num_examples, int num_iterations);
      void predict(double* activations, double* prediction);
      double* get_weights();
      double get_bias();
      int get_num_weights();
    private:
      void initialize_weights(int num_features);
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
