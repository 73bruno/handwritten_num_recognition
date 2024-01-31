# Artificial Vision and Machine Learning Practice Project
This project contains the code and experiments carried out for Practice 2 of the Intelligent Systems course at the University of Alicante during the 2023/2024 academic year.

## Project overview
The main goal of this practice was to implement and compare different machine learning algorithms for classification tasks, with a focus on digit recognition from the MNIST dataset.

The implementations follow the guidelines specified by the Computer Science and Artificial Intelligence Department at the University of Alicante. The key algorithms covered include:

AdaBoost classifier
Decision stump weak learner
Multi-class AdaBoost
AdaBoost with scikit-learn
MLP classifier with Keras
CNN classifier with Keras
Each algorithm is trained and tested on the MNIST dataset. Experiments are performed to determine optimal hyperparameters and compare performance between models.

## Implementations
The AdaBoost, DecisionStump, and Multi-class AdaBoost classes are implemented from scratch as specified by the university.

AdaBoost is also modeled using scikit-learn's AdaBoostClassifier. Experiments optimize the number of weak learners.

For deep learning models, an MLP and CNN are built with Keras. Hyperparameters like layers, neurons, learning rate etc. are tuned.
