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

def save_images(curr_iteration, qualities, new_population, im_shape, 
                save_point, save_dir):
    """
    Saving best solution in a given generation as an image in the specified directory.
    Images are saved accoirding to stop points to avoid saving images from 
    all generations as saving mang images will make the algorithm slow.
    """
    if(numpy.mod(curr_iteration, save_point)==0):
        # Selecting best solution (chromosome) in the generation.
        best_solution_chrom = new_population[numpy.where(qualities == 
                                                         numpy.max(qualities))[0][0], :]
        # Decoding the selected chromosome to return it back as an image.
        best_solution_img = chromosome2img(best_solution_chrom, im_shape)
        # Saving the image in the specified directory.
        matplotlib.pyplot.imsave(save_dir+'solution_'+str(curr_iteration)+'.png', best_solution_img)