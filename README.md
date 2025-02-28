![GitHub](https://img.shields.io/github/license/ennanco/TSP_SAEC?style=flat-square) ![Julia](https://img.shields.io/badge/Julia-1.10.0-blueviolet?logo=Julia)


# Surrogate Evaluation with ANN in a Genetic Algorithm for the Traveling Salesman Problem (TSP)

This project is an example of how to develop an approach of the Surrogated-Assisted Evolutionary Algorithms (SAEA) using an Artificial Neural Network (ANN) within a Genetic Algorithm. This example focuses on the Traveling Salesman Problem (TSP) for an arbitrary number of cities.

## Key Concepts

* **Surrogate Evaluation:** Instead of directly computing the fitness of each individual in the Genetic Algorithm, we train an ANN to approximate the fitness function. This reduces computational cost, especially for complex fitness evaluations.
* **Progressive Training:** The ANN is trained more intensively during the initial generations when the population is diverse. As the algorithm converges, the training frequency is reduced, relying more on the ANN's learned representation. This allows for faster convergence while maintaining accuracy.
* **Computational Efficiency:** By using an ANN for fitness approximation, we significantly reduce the computational cost associated with evaluating complex fitness functions, leading to improved algorithm performance and faster deployment.

## Implementation Details

This implementation utilizes:

* **Julia:** The programming language for its performance and ease of use in scientific computing.
* **Flux.jl:** For building and training the Artificial Neural Network.
* **Plots.jl:** For visualizing the fitness evolution over generations.
* **Statistics.jl:** For statistical calculations used in performance tracking.


## Code Structure
There is a single file with all the code of this example `main.jl`

### Functions

* **main_fitness_function**: Computes the actual TSP fitness for an individual.
* **create_fitness_network**: Creates the ANN model for the evaluation.
* **ann_fitness**: Predicts fitness using the ANN.
* **train_fitness**: Trains the ANN.
* **track_fitness_differences**: Tracks the difference between real and predicted fitness.
* **evolve**: Executes a generation of the genetic algorithm, using the ANN for fitness evaluation according to a certain probability.
* There are other typical functions for the Genetic Algorithm

## Usage
To run the project, execute the `main.jl` script:

```bash
julia main.jl




# License
This project is licensed under the MIT License
