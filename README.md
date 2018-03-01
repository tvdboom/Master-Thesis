## Master Thesis

This repository contains the programming codes used for my thesis of the master Astronomy and Data Science at Leiden University.

The project involved the development of an artificial neural network to discover hypervelocity stars in the Gaia catalog, using only the astrometric parameters of the stars as input features. The network is trained on a mock population.

List of codes:
1. neural_network.py: standalone neural network written in Python
2. machine_learning_functions.py: functions used by neural_network.py (gradient descent,      backpropagation, performance analysis, etc...)
3. tgas.py: neural network application to the Gaia DR1/TGAS catalog. Also includes a Monte-Carlo approach to include errors in the algorithm.
4. particle_swarm_optimization: PSO algorithm to tune the network hyperparameters.

5. nn_keras.py: same neural network as before, this time implemented using Keras.
6. tgas_keras.py: same as tgas.py but using the Keras neural network.
7. hyperparameters.py: Bayesian Optimizatin approach to tune the network hyperparameters.
