from random import uniform, randint, choices
from copy import deepcopy
import numpy as np


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


# -------------------------------------------------------------------------------------------------
# Random Initialization
# -------------------------------------------------------------------------------------------------
def initialize_randomly( problem, population_size, param ):
    """
    Initialize a population of solutions (feasible solution) for an evolutionary algorithm

    Required:

    @ problem - problem's build solution function knows how to create an individual in accordance with the encoding.

    @ population_size - to define the size of the population to be returned.
    """
    solution_list = []

    i = 0
    # generate a population of admissible solutions (individuals)
    for _ in range(0, population_size):
        s = problem.build_solution()

        # check if the solution is admissible
        while not problem.is_admissible( s ):
            s = problem.build_solution()

        s.id = [0, i]
        i += 1
        problem.evaluate_solution ( s )

        solution_list.append( s )

    population = Population(
        problem = problem ,
        maximum_size = population_size,
        solution_list = solution_list )

    return population

###################################################################################################
# SELECTION APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# class RouletteWheelSelection
# -------------------------------------------------------------------------------------------------
# TODO: implement Roulette Wheel for Minimization
class RouletteWheelSelection:
    """
    Main idea: better individuals get higher chance
    The chances are proportional to the fitness
    Implementation: roulette wheel technique
    Assign to each individual a part of the roulette wheel
    Spin the wheel n times to select n individuals

    REMARK: This implementation does not consider minimization problem
    """
    def select(self, population, objective, params):
        """
        select two different parents using roulette wheel
        """
        if objective == "Maximization":
            index1 = self._select_index_for_maximization(population = population)
            index2 = index1

            while index2 == index1:
                index2 = self._select_index_for_maximization( population = population )

            return population.get( index1 ), population.get( index2 )

        elif objective == "Minimization":
            index1 = self._select_index_for_minimization(population = population)
            index2 = index1

            while index2 == index1:
                index2 = self._select_index_for_minimization( population = population )

            return population.get( index1 ), population.get( index2 )



    def _select_index_for_maximization(self, population ):

        # Get the Total Fitness (all solutions in the population) to calculate the chances proportional to fitness
        total_fitness = 0
        for solution in population.solutions:
            total_fitness += solution.fitness

        # spin the wheel
        wheel_position = uniform( 0, 1 )

        # calculate the position which wheel should stop
        stop_position = 0
        index = 0
        for solution in population.solutions :
            stop_position += (solution.fitness / total_fitness)
            if stop_position > wheel_position :
                break
            index += 1

        return index

    def _select_index_for_minimization(self, population ):

        # Get the Total Fitness (all solutions in the population) to calculate the chances proportional to fitness
        total_fitness = 0
        for solution in population.solutions:
            total_fitness += 1/solution.fitness

        # spin the wheel
        wheel_position = uniform( 0, 1 )

        # calculate the position which wheel should stop
        stop_position = 0
        index = 0
        for solution in population.solutions :
            stop_position += ((1/solution.fitness) / total_fitness)
            if stop_position > wheel_position :
                break
            index += 1

        return index




# -------------------------------------------------------------------------------------------------
# class TournamentSelection
# -------------------------------------------------------------------------------------------------
class TournamentSelection:
    """
    """
    def select(self, population, objective, params):
        tournament_size = 2
        if "Tournament-Size" in params:
            self._tournament_size = params[ "Tournament-Size" ]

        index1 = self._select_index( population, tournament_size )
        index2 = index1

        while index2 == index1:
            index2 = self._select_index( population, tournament_size )

        return population.solutions[ index1 ], population.solutions[ index2 ]


    def _select_index(self, population, tournament_size ):

        index_temp      = -1
        index_selected  = randint(0, len( population.solutions )-1 )

        for _ in range( 0, tournament_size ):
            index_temp = randint(0, len( population.solutions )-1 )
            if population.get(index_temp).fitness < population.get(index_selected).fitness:
                index_selected = index_temp

        return index_selected

###################################################################################################
# CROSSOVER APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# Singlepoint crossover
# -------------------------------------------------------------------------------------------------
def singlepoint_crossover( problem, solution1, solution2):
    singlepoint = randint(0, len(solution1.representation)-1)
    #print(f" >> singlepoint: {singlepoint}")

    offspring1 = deepcopy(solution1) #solution1.clone()
    offspring2 = deepcopy(solution2) #.clone()

    for i in range(singlepoint, len(solution2.representation)):
        offspring1.representation[i] = solution2.representation[i]
        offspring2.representation[i] = solution1.representation[i]

    return offspring1, offspring2




###################################################################################################
# MUTATION APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# Singlepoint mutation
# -----------------------------------------------------------------------------------------------
def single_point_mutation( problem, solution):
    singlepoint = randint( 0, len( solution.representation )-1 )
    #print(f" >> singlepoint: {singlepoint}")

    encoding    = problem.encoding

    if encoding.encoding_type == EncodingDataType.choices :
        try:
            temp = deepcopy( encoding.encoding_data )

            temp.pop( solution.representation[ singlepoint ] )

            gene = temp[0]
            if len(temp) > 1 : gene = choices( temp )

            solution.representation[ singlepoint ] = gene

            return solution
        except:
            print('(!) Error: singlepoint mutation encoding.data issues)' )

    else :
        try:
            temp = deepcopy( encoding.encoding_data )

            temp.pop( solution.representation[ singlepoint ] )

            gene = temp[0]
            if len(temp) > 1 : gene = choices( temp )

            solution.representation[ singlepoint ] = gene

            return solution
        except:
            print('(!) Error: singlepoint mutation encoding.data issues)' )

    # return solution

# -------------------------------------------------------------------------------------------------
# Swap mutation
# -----------------------------------------------------------------------------------------------
#TODO: Implement Swap mutation
def swap_mutation( problem, solution):
    r_nr1= 0
    r_nr2 = 0

    while r_nr1 == r_nr2:
        r_nr1 = randint(0,len(solution.representation)- 1)
        r_nr2 = randint(0,len(solution.representation) - 1)

    array = deepcopy(solution.representation)
    array[r_nr1], array[r_nr2] = array[r_nr2], array[r_nr1]
    solution.representation = array

    return solution

def scramble_mutation(problem,solution):
    len_string = randint(2,len(solution.representation)//2)
    random_n = randint(0,len(solution.representation)-len_string)
    new_solution = deepcopy(solution)
    new_solution.representation[random_n:random_n+len_string] = np.array([-1]*len_string)
    un_randomized_indexes = list(range(random_n,random_n+len_string))
    for x in solution.representation[random_n:random_n+len_string]:
        draw = randint(0,len(un_randomized_indexes)-1)
        new_solution.representation[un_randomized_indexes[draw]] = x
        del un_randomized_indexes[draw]



    return new_solution


###################################################################################################
# REPLACEMENT APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# Standard replacement
# -----------------------------------------------------------------------------------------------
def standard_replacement(problem, current_population, new_population ):
    return deepcopy(new_population)

# -------------------------------------------------------------------------------------------------
# Elitism replacement
# -----------------------------------------------------------------------------------------------
def elitism_replacement(problem, current_population, new_population ):


    if problem.objective == ProblemObjective.Minimization :
        if current_population.fittest.fitness < new_population.fittest.fitness :
           new_population.solutions[0] = current_population.solutions[-1]

    elif problem.objective == ProblemObjective.Maximization :
        if current_population.fittest.fitness > new_population.fittest.fitness :
           new_population.solutions[0] = current_population.solutions[-1]

    return deepcopy(new_population)


def cycle_crossover(problem, solution1, solution2):

    ######################### Offspring 1
    parent1 = deepcopy(solution1) #solution1.clone()
    parent2 = deepcopy(solution2) #.clone()
    offspring1 = [None]*len(solution1.representation)
    i = 0
    while len(parent1.representation) > 0:
        if i%2 == 0:
            cycle = create_cycle(parent2,parent1)
            for j in cycle:
                offspring1[solution1.representation.index(j)] = j
        else:
            cycle = create_cycle(parent1,parent2)
            for j in cycle:
                offspring1[solution2.representation.index(j)] = j

        parent1.representation = [j for j in parent1.representation if j not in cycle]
        parent2.representation = [k for k in parent2.representation if k not in cycle]
        i += 1



    ######################### Offspring 2
    parent1 = deepcopy(solution1) #solution1.clone()
    parent2 = deepcopy(solution2) #.clone()
    offspring2 = [None]*len(solution2.representation)
    i = 1
    while len(parent2.representation) > 0:
        if i%2 == 0:
            cycle = create_cycle(parent2,parent1)
            for j in cycle:
                offspring2[solution1.representation.index(j)] = j
        else:
            cycle = create_cycle(parent1,parent2)
            for j in cycle:
                offspring2[solution2.representation.index(j)] = j

        parent1.representation = [j for j in parent1.representation if j not in cycle]
        parent2.representation = [k for k in parent2.representation if k not in cycle]
        i += 1


    offspring1 = LinearSolution(representation = offspring1, encoding_rule = problem._encoding_rule)
    offspring2 = LinearSolution(representation = offspring2, encoding_rule = problem._encoding_rule)
    return offspring1, offspring2


def create_cycle(parent1, parent2):
    start = parent1.representation[0]
    current1 = parent1.representation[0]
    current2 = parent2.representation[0]
    cycle = [current1]

    while current2 != start:
        current1 = current2
        current2 = parent2.representation[parent1.representation.index(current1)]
        cycle.append(current1)

    return cycle


def rank_selection(fitness, num_parents):
    
        """
        Selects the parents using the rank selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
        fitness_sorted.reverse()
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        if self.gene_type_single == True:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=self.gene_type)
        else:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=object)
        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[fitness_sorted[parent_num], :].copy()
        return parents

def random_selection( fitness, num_parents):

        """
        Selects the parents randomly. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        if self.gene_type_single == True:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=self.gene_type)
        else:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=object)

        rand_indices = numpy.random.randint(low=0.0, high=fitness.shape[0], size=num_parents)

        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[rand_indices[parent_num], :].copy()
        return parents

def random_mutation(self, offspring):
    
        """
        Applies the random mutation which changes the values of a number of genes randomly.
        The random value is selected either using the 'gene_space' parameter or the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        # If the mutation values are selected from the mutation space, the attribute 'gene_space' is not None. Otherwise, it is None.
        # When the 'mutation_probability' parameter exists (i.e. not None), then it is used in the mutation. Otherwise, the 'mutation_num_genes' parameter is used.

        if self.mutation_probability is None:
            # When the 'mutation_probability' parameter does not exist (i.e. None), then the parameter 'mutation_num_genes' is used in the mutation.
            if not (self.gene_space is None):
                # When the attribute 'gene_space' exists (i.e. not None), the mutation values are selected randomly from the space of values of each gene.
                offspring = self.mutation_by_space(offspring)
            else:
                offspring = self.mutation_randomly(offspring)
        else:
            # When the 'mutation_probability' parameter exists (i.e. not None), then it is used in the mutation.
            if not (self.gene_space is None):
                # When the attribute 'gene_space' does not exist (i.e. None), the mutation values are selected randomly based on the continuous range specified by the 2 attributes 'random_mutation_min_val' and 'random_mutation_max_val'.
                offspring = self.mutation_probs_by_space(offspring)
            else:
                offspring = self.mutation_probs_randomly(offspring)

        return offspring
  def steady_state_selection(fitness, num_parents):
    
        """
        Selects the parents using the steady-state selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
        fitness_sorted.reverse()
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        if self.gene_type_single == True:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=self.gene_type)
        else:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=object)
        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[fitness_sorted[parent_num], :].copy()
        return parents
 def mutation_randomly( offspring):
    
        """
        Applies the random mutation the mutation probability. For each gene, if its probability is <= that mutation probability, then it will be mutated randomly.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        # Random mutation changes one or more genes in each offspring randomly.
        for offspring_idx in range(offspring.shape[0]):
            mutation_indices = numpy.array(random.sample(range(0, self.num_genes), self.mutation_num_genes))
            for gene_idx in mutation_indices:
                # Generating a random value.
                random_value = numpy.random.uniform(low=self.random_mutation_min_val, 
                                                    high=self.random_mutation_max_val, 
                                                    size=1)
                # If the mutation_by_replacement attribute is True, then the random value replaces the current gene value.
                if self.mutation_by_replacement:
                    if self.gene_type_single == True:
                        random_value = self.gene_type(random_value)
                    else:
                        random_value = self.gene_type[gene_idx](random_value)
                        if type(random_value) is numpy.ndarray:
                            random_value = random_value[0]
               # If the mutation_by_replacement attribute is False, then the random value is added to the gene value.
                else:
                    if self.gene_type_single == True:
                        random_value = self.gene_type(offspring[offspring_idx, gene_idx] + random_value)
                    else:
                        random_value = self.gene_type[gene_idx](offspring[offspring_idx, gene_idx] + random_value)
                        if type(random_value) is numpy.ndarray:
                            random_value = random_value[0]

                offspring[offspring_idx, gene_idx] = random_value

                if self.allow_duplicate_genes == False:
                    offspring[offspring_idx], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[offspring_idx],
                                                                                         min_val=self.random_mutation_min_val,
                                                                                         max_val=self.random_mutation_max_val,
                                                                                         mutation_by_replacement=self.mutation_by_replacement,
                                                                                         gene_type=self.gene_type,
                                                                                         num_trials=10)

        return offspring

    def mutation_probs_randomly(offspring):

        """
        Applies the random mutation using the mutation probability. For each gene, if its probability is <= that mutation probability, then it will be mutated randomly.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        # Random mutation changes one or more gene in each offspring randomly.
        for offspring_idx in range(offspring.shape[0]):
            probs = numpy.random.random(size=offspring.shape[1])
            for gene_idx in range(offspring.shape[1]):
                if probs[gene_idx] <= self.mutation_probability:
                    # Generating a random value.
                    random_value = numpy.random.uniform(low=self.random_mutation_min_val, 
                                                        high=self.random_mutation_max_val, 
                                                        size=1)
                    # If the mutation_by_replacement attribute is True, then the random value replaces the current gene value.
                    if self.mutation_by_replacement:
                        if self.gene_type_single == True:
                            random_value = self.gene_type(random_value)
                        else:
                            random_value = self.gene_type[gene_idx](random_value)
                            if type(random_value) is numpy.ndarray:
                                random_value = random_value[0]
                    # If the mutation_by_replacement attribute is False, then the random value is added to the gene value.
                    else:
                        if self.gene_type_single == True:
                            random_value = self.gene_type(offspring[offspring_idx, gene_idx] + random_value)
                        else:
                            random_value = self.gene_type[gene_idx](offspring[offspring_idx, gene_idx] + random_value)
                            if type(random_value) is numpy.ndarray:
                                random_value = random_value[0]

                    offspring[offspring_idx, gene_idx] = random_value

                    if self.allow_duplicate_genes == False:
                        offspring[offspring_idx], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[offspring_idx],
                                                                                             min_val=self.random_mutation_min_val,
                                                                                             max_val=self.random_mutation_max_val,
                                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                                             gene_type=self.gene_type,
                                                                                             num_trials=10)
        return offspring