import numpy as np
import matplotlib.pyplot as plt

def calculate_payoff(player, strategy, types, beliefs):
    """
    Calculate the expected payoff for a player given their strategy and beliefs about other players' types.

    Parameters:
    player (int): Index of the player.
    strategy (int): Index of the strategy chosen by the player.
    types (list of ints): Types of all players.
    beliefs (list of lists): Beliefs about other players' types.

    Returns:
    float: Expected payoff for the player.
    """
    payoff = 0
    for other_player in range(len(beliefs)):
        if other_player != player:
            for other_strategy in range(len(beliefs[other_player])):
                payoff += beliefs[player][other_player][other_strategy] * (strategy == other_strategy)
    return payoff

def is_nash_equilibrium(strategies, beliefs):
    """
    Check if the given strategy profile constitutes a Nash Equilibrium.

    Parameters:
    strategies (list of ints): List of strategies chosen by all players.
    beliefs (list of lists): Beliefs about other players' types.

    Returns:
    bool: True if the strategy profile constitutes a Nash Equilibrium, False otherwise.
    """
    for player, strategy in enumerate(strategies):
        expected_payoff = calculate_payoff(player, strategy, range(len(beliefs)), beliefs)
        for other_strategy in range(len(beliefs[player])):
            if calculate_payoff(player, other_strategy, range(len(beliefs)), beliefs) > expected_payoff:
                return False
    return True

# Parameters
payoff_matrix = np.array([[2, 0], [3, 1]])  # Payoff matrix: [Hawk, Dove] x [Hawk, Dove]
mutation_rate = 0.01  # Mutation rate: Probability of a strategy mutating to another randomly

# Initial population proportions
population_size = 1000
initial_hawk_proportion = 0.5
initial_dove_proportion = 1 - initial_hawk_proportion

# Simulate evolution
num_generations = 100
hawk_proportions = np.zeros(num_generations)
dove_proportions = np.zeros(num_generations)

beliefs = [
    [[0.5, 0.5], [0.3, 0.7]],  # Player 1's beliefs about Player 2's type and strategy
    [[0.4, 0.6], [0.2, 0.8]]   # Player 2's beliefs about Player 1's type and strategy
]

strategies = [0, 1]  # Strategies chosen by Player 1 and Player 2, respectively

update_rules = [
    lambda beliefs: beliefs,  # Player 1's update rule
    lambda beliefs: beliefs   # Player 2's update rule
]

for generation in range(num_generations):
    # Calculate payoffs for each strategy
    total_payoff = np.dot([initial_hawk_proportion, initial_dove_proportion], payoff_matrix)
    
    # Update strategy proportions using replicator dynamics
    hawk_payoff = np.dot([initial_hawk_proportion, initial_dove_proportion], payoff_matrix[:, 0])
    dove_payoff = np.dot([initial_hawk_proportion, initial_dove_proportion], payoff_matrix[:, 1])
    hawk_proportions[generation] = initial_hawk_proportion
    dove_proportions[generation] = initial_dove_proportion
    initial_hawk_proportion += mutation_rate * (hawk_payoff - total_payoff) * initial_hawk_proportion
    initial_dove_proportion += mutation_rate * (dove_payoff - total_payoff) * initial_dove_proportion

    # Ensure proportions are within [0, 1]
    initial_hawk_proportion = np.clip(initial_hawk_proportion, 0, 1)
    initial_dove_proportion = np.clip(initial_dove_proportion, 0, 1)

# Plot results
generations = np.arange(num_generations)
plt.plot(generations, hawk_proportions, label='Hawk')
plt.plot(generations, dove_proportions, label='Dove')
plt.xlabel('Generations')
plt.ylabel('Proportion of Population')
plt.title('Evolution of Hawk-Dove Game')
plt.legend()
plt.show()

