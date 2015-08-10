#include <thread>

namespace augur {

  Layer::Layer(int number_of_nodes) {
    num_nodes = number_of_nodes;
    for(int i = 0; i < number_of_nodes; ++i) {
      perceptrons[i] = new Perceptron(number_of_nodes);
    }
  }

  Layer::~Layer() {
    for(int i = 0; i < num_nodes; ++i) {
      delete perceptrons[i];
    }
  }

  void Layer::perceptron_predict(Perceptron* target, double* activations, int num_activations, double* prediction) {
    target->predict(activations, num_activations, prediction);
  }

  double* Layer::feed_forward(double* activations, int num_activations) {
    std::thread perceptron_jobs[num_nodes];
    predictions = new double[num_nodes];

    for(int i = 0; i < num_nodes; ++i) {
      perceptron_jobs[i] = std::thread(perceptron_predict, perceptrons[i], activations, num_activations, predictions + i);
    }
    for(int i = 0; i < num_nodes; ++i) {
      perceptron_jobs[i].join();
    }
    return predictions;
  }

}
