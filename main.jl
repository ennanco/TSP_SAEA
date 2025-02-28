using Random
using Flux
using Distributions
using Plots

# Some functions to support the creation of a basic Genetic Algorithm and the performance of the test with the ANN developed with Flux
"""
A Genetic Algorithm implementation to optimize solutions represented as floating-point vectors.

# Fields
- `population::Vector{Vector{Float32}}`: The set of individuals in the population.
- `fitnesses::Vector{Float32}`: Fitness values of the population.
- `fitness_function::Function`: Function to evaluate individuals.
- `mutation_rate::Float32`: Probability of mutation for each gene.
- `crossover_rate::Float32`: Probability of performing crossover.
- `pop_size::Int`: Number of individuals in the population.
- `chrom_length::Int`: Length of each individual's chromosome.
"""
struct GeneticAlgorithm
    population::Vector{Vector{Float32}}
    fitnesses:: Vector{Float32}
    fitness_function::Function
    mutation_rate::Float32
    crossover_rate::Float32
    pop_size ::Int
    chrom_length::Int

    function GeneticAlgorithm(pop_size::Int, chrom_length::Int, fitness_function::Function;
                               mutation_rate::Float32=0.01f0, crossover_rate::Float32=0.7f0)
        population = [rand(Float32, chrom_length) for _ in 1:pop_size]
        fitnesses = [fitness_function(ind) for ind in population]
        new(population, fitnesses, fitness_function, mutation_rate, crossover_rate, pop_size, chrom_length)
    end
end

# Roulette selection
function select_parents_idx(ga::GeneticAlgorithm)
    probs = ga.fitnesses / sum(ga.fitnesses)
    return rand(Categorical(probs)), rand(Categorical(probs))
end

# Single point crossover function; other strategies can be implemented
function crossover(parent1::Vector{Float32}, parent2::Vector{Float32}, rate::Float32)
    if rand() < rate
        point = rand(1:length(parent1)-1)
        return [parent1[1:point]; parent2[point+1:end]], [parent2[1:point]; parent1[point+1:end]]
    else
        return parent1, parent2
    end
end

# Mutation strategy random changes with probability between 0 and 1
function mutate(individual::Vector{Float32}, rate::Float32)
    return [rand(Float32) < rate ? rand(Float32) : gene for gene in individual]
end

# Parent replacement with an elitist strategy
function replacement!(ga::GeneticAlgorithm, parent1_idx::Int, parent2_idx::Int, element_list::Vector{Vector{Float32}}, fitness_list::Vector{Float32})
    ordered_list = sortperm(fitness_list, rev=true)
    ga.population[parent1_idx] = element_list[ordered_list[1]]
    ga.fitnesses[parent1_idx] = fitness_list[ordered_list[1]]
    ga.population[parent2_idx] = element_list[ordered_list[2]]
    ga.fitnesses[parent2_idx] = fitness_list[ordered_list[2]]
end

# Evolve for a single generation, it can used the ANN with the probability 'prob'
function evolve!(ga::GeneticAlgorithm, network::Chain, prob::Float32)
    for _ in 1:round(Int, ga.pop_size * ga.crossover_rate / 2) # Number of crossovers per new generation
        parent1_idx, parent2_idx = select_parents_idx(ga)
        offspring1, offspring2 = crossover(ga.population[parent1_idx], ga.population[parent2_idx], ga.crossover_rate)
        offspring1, offspring2 = mutate(offspring1, ga.mutation_rate), mutate(offspring2, ga.mutation_rate)

        # Evaluation
        element_list = [ga.population[parent1_idx], ga.population[parent2_idx], offspring1, offspring2]
        fitness_list = [ga.fitnesses[parent1_idx],
                        ga.fitnesses[parent2_idx],
                        rand() < prob ? ga.fitness_function(offspring1) : network(offspring1)[1],
                        rand() < prob ? ga.fitness_function(offspring2) : network(offspring2)[1]]

        # Replace parents
        replacement!(ga, parent1_idx, parent2_idx, element_list, fitness_list)
    end
end

# Travel Salesman Problem (TSP) evaluation
function main_fitness_function(individual::Vector{Float32})
    # Evaluate the fitness of the individual
    # Convert the values of the individual into a permutation
    permutation = sortperm(individual)

    # Calculate total distance (minimized, inverted to maximize fitness)
    permutation = vcat(permutation, permutation[1])  # Add the starting point at the end
    cities_ordered = CITIES[permutation]  # Extract coordinates in order
    distance = sum(sqrt((p2[1] - p1[1])^2 + (p2[2] - p1[2])^2) for (p1, p2) in zip(cities_ordered[1:end-1], cities_ordered[2:end]))

    return 1.0f0 / (distance + 1.0f-6)  # Avoid division by zero
end

# Simulated evaluation function (the neural network)
function create_fitness_network(input_size::Int)
    return Chain(
        Dense(input_size, 10, relu),
        Dense(10, 1, identity)  # The output should be a scalar (fitness value)
    )
end

function ann_fitness(ga::GeneticAlgorithm, network::Chain)
    input_matrix = hcat(ga.population...)  # Convert the vector of vectors into a matrix
    ann_output = network(input_matrix)[:]
    return ann_output, abs.(ann_output - ga.fitnesses)
end

# Function to train the neural network
function train_fitness!(network::Chain, population::Vector{Vector{Float32}}, fitnesses::Vector{Float32}; epochs::Int=2, learning_rate::Float32=0.01f0)
    input_matrix = hcat(population...)
    target_matrix = reshape(fitnesses, 1, :)
    data = [(input_matrix, target_matrix)]
    loss(m, x, y) = Flux.Losses.mse(m(x), y)
    optimizer = Flux.setup(Adam(learning_rate), network)
    for _ in 1:epochs
        Flux.train!(loss, network, data, optimizer)
    end
end

# This utalitatian function tries to evaluate the differences between the estimation and the real value to assest the performance
function track_fitness_differences(ga::GeneticAlgorithm, ann_fitness::Vector{Float32}, differences::Vector{Float32})
    best = maximum(ga.fitnesses)
    worst = minimum(ga.fitnesses)
    mean_fitness = mean(ga.fitnesses)
    ann_mean_fitness = mean(ann_fitness)
    ann_error = mean(differences)
    return (best, worst, mean_fitness, ann_mean_fitness, ann_error)
end

##### Main execution block ###
# Parameters for the TSP
const NUM_CITIES = 50
const CITIES = [(rand(Float32, 2)) for _ in 1:NUM_CITIES]

# define the ANN and the GA
network = create_fitness_network(NUM_CITIES)
ga = GeneticAlgorithm(100, NUM_CITIES, main_fitness_function)
generations = 50
fitness_history = []
gen_range = 1:generations

# Evolving the solution
for i in gen_range
    prob::Float32 = 1 - i / 20
    evolve!(ga, network, prob)
    ann_prediction, differences = ann_fitness(ga, network)
    train_fitness!(network, ga.population, ga.fitnesses)
    push!(fitness_history, track_fitness_differences(ga, ann_prediction, differences))
end

#println("Fitness history: ", fitness_history[1])

# Extract data for plotting
best_fitness_history = [h[1] for h in fitness_history]
worst_fitness_history = [h[2] for h in fitness_history]
mean_fitness_history = [h[3] for h in fitness_history]
ann_fitness_history = [h[4] for h in fitness_history]
error_history = [h[5] for h in fitness_history]

# Plot fitness evolution
graph_fitness = plot(gen_range, best_fitness_history, label="Best Fitness", lw=2)
plot!(gen_range, worst_fitness_history, label="Worst Fitness", lw=2, linestyle=:dash)
plot!(gen_range, mean_fitness_history, label="Mean Fitness", lw=2, linestyle=:dot)
plot!(gen_range, ann_fitness_history, label="ANN Average Fitness", lw=2, linestyle=:dashdot)
title!("Fitness Evolution")
xlabel!("Generations")
ylabel!("Fitness")
savefig("fitness_evolution.png")


