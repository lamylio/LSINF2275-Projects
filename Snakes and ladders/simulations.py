from snakesandladders import SnakesAndLadders, Simulation, markovDecision
import numpy as np

print("\033[93m\033[1m\n===============\n[SIMULATIONS]\nWarning: computing the sub-optimal strategies is computer intensive\n\033[0m")

# By the way, you might want to set SnakesAndLadders.INFO to True in order to have full output ! 

GAMES_TO_PLAY = 500_000 # please consider using less than 100_000


# Simulations and Comparisons (Blank layout)
print("\n[BLANK LAYOUT]\n")

layout = np.zeros(15)
average_turns, best_strategy = markovDecision(layout, False)

sim = Simulation(layout, False, best_strategy)
empirical_turns = sim.empiricalTurns(GAMES_TO_PLAY)

print("\nThe empirical number of turns is", round(empirical_turns, 2), "compared to", round(average_turns[0], 2), "given by the MDP.")

# suboptimal strategy 1 (intuitive strategy)
fast_lane = [3] * 14
fast_lane[0] = 2
fast_lane[1] = 1
sim_fast_lane  = Simulation(layout, False, fast_lane)
print("Fast_lane strategy: ", sim_fast_lane.empiricalTurns(GAMES_TO_PLAY))

# Suboptimal strategies with one dice + random
print("Classical sub-optimal strategies are: ", sim.computeSubOptimalStrategies(GAMES_TO_PLAY))

# ==============
# Simulations and Comparisons (Circular Blank Layout)
print("\n[BLANK LAYOUT - CIRCLE]\n")

layout = np.zeros(15)
average_turns, best_strategy = markovDecision(layout, True)

sim = Simulation(layout, True, best_strategy)

empirical_turns = sim.empiricalTurns(GAMES_TO_PLAY)
print("\nThe empirical number of turns is", round(empirical_turns, 2), "compared to", round(average_turns[0], 2), "given by the MDP.")

# suboptimal strategy 1 (intuitive strategy)
end_safely = [3] * 14
end_safely[8] = 2
end_safely[9] = 1

end_safely[12] = 2
end_safely[13] = 1

sim_end_safely  = Simulation(layout, True, end_safely)
print("End_safely strategy: ", sim_end_safely.empiricalTurns(GAMES_TO_PLAY))
# Suboptimal strategies with one dice + random
print("Classical sub-optimal strategies are: ", sim.computeSubOptimalStrategies(GAMES_TO_PLAY))

# ==============
# # Simulations and Comparisons (Layout full of traps)
print("\n[TRAP ROTATION LAYOUT]\n")

layout = [0] + [1, 2, 3, 4] * 3 + [1, 0]
layout[0] = SnakesAndLadders.NOT_TRAP
layout[14] = SnakesAndLadders.NOT_TRAP
average_turns, best_strategy = markovDecision(layout, False)

sim = Simulation(layout, False, best_strategy)

empirical_turns = sim.empiricalTurns(GAMES_TO_PLAY)
print("\nThe empirical number of turns is", round(empirical_turns, 2), "compared to", round(average_turns[0], 2), "given by the MDP.")

# Suboptimal strategies with one dice + random
print("Classical sub-optimal strategies are: ", sim.computeSubOptimalStrategies(GAMES_TO_PLAY))


# ==============
# Simulations and Comparisons (Layout with some traps)
print("\n[SOME TRAPS LAYOUT]\n")

layout = np.zeros(15)

layout[1] = SnakesAndLadders.TRAP_PENALTY
layout[2] = SnakesAndLadders.TRAP_PENALTY
layout[3] = SnakesAndLadders.TRAP_PRISON
layout[5] = SnakesAndLadders.TRAP_PENALTY
layout[6] = SnakesAndLadders.TRAP_RESTART


average_turns, best_strategy = markovDecision(layout, True)

sim = Simulation(layout, True, best_strategy)

empirical_turns = sim.empiricalTurns(GAMES_TO_PLAY)
print("\nThe empirical number of turns is", round(empirical_turns, 2), "compared to", round(average_turns[0], 2), "given by the MDP.")

# Suboptimal strategies with one dice + random
print("Classical sub-optimal strategies are: ", sim.computeSubOptimalStrategies(GAMES_TO_PLAY))

# ==============
print("\n[END]\n")
# END