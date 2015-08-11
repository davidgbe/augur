#include <pthread.h>

#include "layer.h"

namespace augur {

  Layer::Layer(int number_of_nodes, int num_feats, int lvl) {
    num_nodes = number_of_nodes;
    num_features = num_feats;
    level = lvl;
    for(int i = 0; i < number_of_nodes; ++i) {
      perceptrons.push_back( new Perceptron(i, num_features) );
    }
  }

  Layer::~Layer() {
    for(int i = 0; i < num_nodes; ++i) {
      delete perceptrons.at(i);
    }
  }

  void Layer::perceptron_predict(void* info) {
    info = (ThreadInfo*) info;
    info->perceptron->predict(info->activations, info->num_activations, info->prediction);
  }

  double* Layer::feed_forward(double* activations, int num_activations) {
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    void* status;

    pthread_t threads[num_nodes];
    double* predictions = new double[num_nodes];
    ThreadInfo infos[num_nodes];

    for(int i = 0; i < num_nodes; ++i) {
      infos[i].perceptron = perceptrons.at(i);
      infos[i].activations = activations;
      infos[i].num_activations = num_activations;
      infos[i].prediction = predictions + i;

      int rc = pthread_create(&threads[i], NULL, &perceptron_predict, (infos + i));
    }
    for(int i = 0; i < num_nodes; ++i) {
      int rc = pthread_join(threads[i], &status);

    }
    return predictions;
  }

}
