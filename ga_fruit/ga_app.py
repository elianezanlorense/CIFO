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


#Load data
---target='bird.jpg'
---target='fruit.jpg'
target_im = load_data(target)


# default params
default_params = {
    "Population-Size"           : 5,
    "num_generation"     :       10,
    "mutation_probability"     : 0.1,
    "crossover_probability"      : 0.75,
    "initialization_approach"   : initialize_randomly,
    "parente_selection"        : RouletteWheelSelection, ##Selection:RouletteWheelSelection, TournamentSelection 
    "tournament_size"           : 5,
    "crossover_type"        : singlepoint_crossover,
    "mutation_type"          : single_point_mutation,

}



ga_fruit.run()

ga_fruit.plot_result()

