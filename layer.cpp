#include <pthread.h>
#include <iostream>

#include "layer.h"

namespace augur {

  struct thread_info {
    Perceptron* perceptron;
    double* activations;
    double* prediction;
  };

  Layer::Layer(int number_of_nodes, int num_feats, int lvl) {
    num_nodes = number_of_nodes;
    num_features = num_feats;
    level = lvl;
    std::cout << "Layer at level " << lvl << " initialized" << std::endl;
    for(int i = 0; i < number_of_nodes; ++i) {
      perceptrons.push_back( new Perceptron(i, num_features) );
    }
  }

  Layer::~Layer() {
    for(int i = 0; i < num_nodes; ++i) {
      delete perceptrons.at(i);
    }
  }

  void* Layer::perceptron_predict(void* info) {
    struct thread_info* ptr = (struct thread_info*) info;
    ptr->perceptron->predict(ptr->activations, ptr->prediction);
    return info;
  }

  double* Layer::feed_forward(double* activations, double* predictions) {
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    void* status;

    pthread_t threads[num_nodes];
    predictions = new double[num_nodes];
    thread_info infos[num_nodes];

    for(int i = 0; i < num_nodes; ++i) {
      infos[i].perceptron = perceptrons.at(i);
      infos[i].activations = activations;
      infos[i].prediction = predictions + i;

      int rc = pthread_create(&threads[i], NULL, perceptron_predict, (infos + i));
    }

    for(int i = 0; i < num_nodes; ++i) {
      int rc = pthread_join(threads[i], &status);
    }

    return predictions;
  }

}
