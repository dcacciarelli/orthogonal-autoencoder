# Orthogonal autoencoder for anomaly detection

Model used in the paper "A novel fault detection and diagnosis approach based on orthogonal autoencoders". 

The full paper is available [here](https://www.sciencedirect.com/science/article/pii/S0098135422001910).

This repo contains:
1. `model`: torch implementation of a simple feed-forward autoencoder.
2. `training_utils`: functions for training the model using the orthogonality regularization.
3. `gradients`: functions to estimate the integrated gradients given a trained model, an input tensor and a baseline. The integrated gradients function is       implemented using two different approximation methods (trapezoidal and Monte Carlo).
4. `training.ipynb`: notebook showing how to use the previous methods with an open source dataset.
