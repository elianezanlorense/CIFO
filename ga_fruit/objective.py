
import imageio
import numpy

from ga_operators import swap_mutation, random_mutation, scramble_mutation,
from ga_operators import singlepoint_crossover
from ga_operators import TournamentSelection,RouletteWheelSelection,steady_state_selection
from load_data import load_data
from initial_population import initial_population
from fitness_fun import fitness_fun
from save_images import save_images
import matplotlib

class ProblemObjective:
    """
    A problem objective can be Minimization or Maximization
    """
    Maximization    = "Maximization"
    Minimization    = "Minimization"
    MultiObjective  = "Multi-Objective"