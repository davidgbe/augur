#include <pthread.h>
#include <iostream>

#include "layer.h"

namespace augur {

  struct feed_forward_thread_info {
    int thread_num;
    Perceptron* perceptron;
    double* activations;
    double* prediction;
  };

  struct generate_error_thread_info {
    int thread_num;
    std::vector<Perceptron*>* children;
    double* activations;
    Perceptron* perceptron;
  };



  Layer::Layer(int number_of_nodes, int num_feats, int lvl) {
    num_nodes = number_of_nodes;
    num_features = num_feats;
    level = lvl;
    inputs = NULL;
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
    struct feed_forward_thread_info* ptr = (struct feed_forward_thread_info*) info;
    std::cout << "thread num: " << ptr->thread_num << std::endl;
    ptr->perceptron->predict(ptr->activations, ptr->prediction);
    std::cout << ptr->thread_num << " end" << std::endl;
    return info;
  }

  double* Layer::feed_forward(double* activations, bool del_activations) {
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    void* status;

    pthread_t threads[num_nodes];
    double* predictions = new double[num_nodes];
    feed_forward_thread_info infos[num_nodes];

    for(int i = 0; i < num_nodes; ++i) {
      infos[i].thread_num = i;
      infos[i].perceptron = perceptrons.at(i);
      infos[i].activations = activations;
      infos[i].prediction = predictions + i;

      int rc = pthread_create(&threads[i], NULL, perceptron_predict, infos + i);
    }

    for(int i = 0; i < num_nodes; ++i) {
      int rc = pthread_join(threads[i], &status);
    }
    if(del_activations) {
      delete[] activations;
    }

    return predictions;
  }

  void Layer::set_inputs(double* in) {
    if(inputs != NULL) {
      delete[] inputs;
    }
    inputs = in;
  }

  void* Layer::calculate_single_perceptron_error(void* thread_info) {
    struct generate_error_thread_info* ptr = (struct generate_error_thread_info*) thread_info;
    std::cout << "Thread num: " << ptr->thread_num << std::endl;
    ptr->perceptron->generate_error_as_parent(ptr->activations, ptr->children);
  }

  void Layer::calculate_perceptron_errors(Layer* higher_layer) {
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    void* status;

    int num_threads = perceptrons.size();
    pthread_t threads[num_threads];
    generate_error_thread_info infos[num_threads];

    for(int i = 0; i < num_threads; ++i) {
      infos[i].thread_num = i;
      infos[i].perceptron = perceptrons.at(i);
      infos[i].activations = inputs;
      infos[i].children = &(higher_layer->perceptrons);

      int rc = pthread_create(&threads[i], NULL, calculate_single_perceptron_error, infos + i);
    }

    for(int i = 0; i < num_threads; ++i) {
      int rc = pthread_join(threads[i], &status);
    }

    std::cout << error << std::endl;
  }

  void Layer::calculate_root_error(double y) {
    if(position != 0) {
      //throw
    }
    perceptrons.at(0)->generate_error_as_root(y);
  }
}
