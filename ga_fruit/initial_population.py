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

def initial_population(target_im, population_size):
    """
    Creating an initial population randomly.
    """
    # Empty population of chromosomes accoridng to the population size specified.
    init_population = numpy.empty(shape=(population_size, 
                                  functools.reduce(operator.mul, target_im)),
                                  dtype=numpy.uint8)
    for indv_num in range(population_size):
        # Randomly generating initial population chromosomes genes values.
        init_population[indv_num, :] = numpy.random.random(
                                functools.reduce(operator.mul, target_im))*256
    return init_population