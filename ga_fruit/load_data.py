import itertools
import functools
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
def load_data(target):
    
    
    target_arr=imageio.imread(target)
    """
    Represents the image as a 1D vector.
    
    img_arr: The image to be converted into a vector.
    
    Returns the vector.
    """

    return numpy.reshape(a=target_arr, newshape=(functools.reduce(operator.mul, target_arr.shape)))