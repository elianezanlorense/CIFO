import numpy
import random
import matplotlib.pyplot
import pickle
import time
import warnings
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

class ga_fruit:

    supported_int_types = [int, numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]
    supported_float_types = [float, numpy.float, numpy.float16, numpy.float32, numpy.float64]
    supported_int_float_types = supported_int_types + supported_float_types

        # If suppress_warnings is bool and its valud is False, then print warning messages.
        if type(suppress_warnings) is bool:
            self.suppress_warnings = suppress_warnings
        else:
            self.valid_parameters = False
            raise TypeError("The expected type of the 'suppress_warnings' parameter is bool but {suppress_warnings_type} found.".format(suppress_warnings_type=type(suppress_warnings)))

        # Validating mutation_by_replacement
        if not (type(mutation_by_replacement) is bool):
            self.valid_parameters = False
            raise TypeError("The expected type of the 'mutation_by_replacement' parameter is bool but ({mutation_by_replacement_type}) found.".format(mutation_by_replacement_type=type(mutation_by_replacement)))

        self.mutation_by_replacement = mutation_by_replacement

        # Validate gene_space
        self.gene_space_nested = False
        if type(gene_space) is type(None):
            pass
        elif type(gene_space) in [list, tuple, range, numpy.ndarray]:
            if len(gene_space) == 0:
                self.valid_parameters = False
                raise TypeError("'gene_space' cannot be empty (i.e. its length must be >= 0).")
            else:
                for index, el in enumerate(gene_space):
                    if type(el) in [list, tuple, range, numpy.ndarray]:
                        if len(el) == 0:
                            self.valid_parameters = False
                            raise TypeError("The element indexed {index} of 'gene_space' with type {el_type} cannot be empty (i.e. its length must be >= 0).".format(index=index, el_type=type(el)))
                        else:
                            for val in el:
                                if not (type(val) in [type(None)] + GA.supported_int_float_types):
                                    raise TypeError("All values in the sublists inside the 'gene_space' attribute must be numeric of type int/float/None but ({val}) of type {typ} found.".format(val=val, typ=type(val)))
                        self.gene_space_nested = True
                    elif type(el) == type(None):
                        pass
                        # self.gene_space_nested = True
                    elif type(el) is dict:
                        if len(el.items()) == 2:
                            if ('low' in el.keys()) and ('high' in el.keys()):
                                pass
                            else:
                                self.valid_parameters = False
                                raise TypeError("When an element in the 'gene_space' parameter is of type dict, then it must have only 2 items with keys 'low' and 'high' but the following keys found: {gene_space_dict_keys}".format(gene_space_dict_keys=el.keys()))
                        else:
                            self.valid_parameters = False
                            raise TypeError("When an element in the 'gene_space' parameter is of type dict, then it must have only 2 items but ({num_items}) items found.".format(num_items=len(el.items())))
                        self.gene_space_nested = True
                    elif not (type(el) in GA.supported_int_float_types):
                        self.valid_parameters = False
                        raise TypeError("Unexpected type {el_type} for the element indexed {index} of 'gene_space'. The accepted types are list/tuple/range/numpy.ndarray of numbers, a single number (int/float), or None.".format(index=index, el_type=type(el)))

        elif type(gene_space) is dict:
            if len(gene_space.items()) == 2:
                if ('low' in gene_space.keys()) and ('high' in gene_space.keys()):
                    pass
                else:
                    self.valid_parameters = False
                    raise TypeError("When the 'gene_space' parameter is of type dict, then it must have only 2 items with keys 'low' and 'high' but the following keys found: {gene_space_dict_keys}".format(gene_space_dict_keys=gene_space.keys()))
            else:
                self.valid_parameters = False
                raise TypeError("When the 'gene_space' parameter is of type dict, then it must have only 2 items but ({num_items}) items found.".format(num_items=len(gene_space.items())))

        else:
            self.valid_parameters = False
            raise TypeError("The expected type of 'gene_space' is list, tuple, range, or numpy.ndarray but ({gene_space_type}) found.".format(gene_space_type=type(gene_space)))
            
        self.gene_space = gene_space

        # Validate init_range_low and init_range_high
        if type(init_range_low) in GA.supported_int_float_types:
            if type(init_range_high) in GA.supported_int_float_types:
                self.init_range_low = init_range_low
                self.init_range_high = init_range_high
            else:
                self.valid_parameters = False
                raise ValueError("The value passed to the 'init_range_high' parameter must be either integer or floating-point number but the value ({init_range_high_value}) of type {init_range_high_type} found.".format(init_range_high_value=init_range_high, init_range_high_type=type(init_range_high)))
        else:
            self.valid_parameters = False
            raise ValueError("The value passed to the 'init_range_low' parameter must be either integer or floating-point number but the value ({init_range_low_value}) of type {init_range_low_type} found.".format(init_range_low_value=init_range_low, init_range_low_type=type(init_range_low)))

        # Validate gene_type
        if gene_type in GA.supported_int_float_types:
            self.gene_type = gene_type
            self.gene_type_single = True
        elif type(gene_type) in [list, tuple, numpy.ndarray]:
            if len(gene_type) == num_genes:
                pass
            else:
                self.valid_parameters = False
                raise TypeError("When the parameter 'gene_type' is nested, then its length must be equal to the value passed to the 'num_genes' parameter. Instead, len(gene_type) ({len_gene_type}) != len(num_genes) ({num_genes})".format(len_gene_type=len(gene_type), num_genes=num_genes))
            for gene_type_idx, gene_type_val in enumerate(gene_type):
                if gene_type_val in GA.supported_int_float_types:
                    pass
                else:
                    self.valid_parameters = False
                    raise ValueError("When a list/tuple/numpy.ndarray is assigned to the 'gene_type' parameter, then the values must be of integer or floating-point data types but the value ({gene_type_val}) of type {gene_type_type} found at index {gene_type_idx}.".format(gene_type_val=gene_type_val, gene_type_type=type(gene_type_val), gene_type_idx=gene_type_idx))
            self.gene_type = gene_type
            self.gene_type_single = False
        else:
            self.valid_parameters = False
            raise ValueError("The value passed to the 'gene_type' parameter must be either a single integer or floating-point type or a list/tuple/numpy.ndarray of int/float data types but the value ({gene_type_val}) of type {gene_type_type} found.".format(gene_type_val=gene_type_val, gene_type_type=type(gene_type)))

        # Import the initial population
        initial_population =initial_population.initial_population()

        # Validating the number of parents to be selected for mating: num_parents_mating
        if (num_parents_mating > self.sol_per_pop):
            self.valid_parameters = False
            raise ValueError("The number of parents to select for mating ({num_parents_mating}) cannot be greater than the number of solutions in the population ({sol_per_pop}) (i.e., num_parents_mating must always be <= sol_per_pop).\n".format(num_parents_mating=num_parents_mating, sol_per_pop=self.sol_per_pop))

        self.num_parents_mating = num_parents_mating

        # crossover: Refers to the method that applies the crossover operator based on the selected type of crossover in the crossover_type property.
        # Validating the crossover type: crossover_type

        crossover_type = ga_operators.crossover_type=crossover_type
          
        # mutation: Refers to the method that applies the mutation operator based on the selected type of mutation in the mutation_type property.
        # Validating the mutation type: mutation_type
        # "adaptive" mutation is supported starting from PyGAD 2.10.0
       mutation_type = ga_operators.mutation_tipe=cromutation_typessover_type
        # Calculate the value of mutation_num_genes
        if not (self.mutation_type is None):
            if mutation_num_genes is None:
                # The mutation_num_genes parameter does not exist. Checking whether adaptive mutation is used.
                if (mutation_type != "adaptive"):
                    # The percent of genes to mutate is fixed not adaptive.
                    if mutation_percent_genes == 'default'.lower():
                        mutation_percent_genes = 10
                        # Based on the mutation percentage in the 'mutation_percent_genes' parameter, the number of genes to mutate is calculated.
                        mutation_num_genes = numpy.uint32((mutation_percent_genes*self.num_genes)/100)
                        # Based on the mutation percentage of genes, if the number of selected genes for mutation is less than the least possible value which is 1, then the number will be set to 1.
                        if mutation_num_genes == 0:
                            if self.mutation_probability is None:
                                if not self.suppress_warnings: warnings.warn("The percentage of genes to mutate (mutation_percent_genes={mutation_percent}) resutled in selecting ({mutation_num}) genes. The number of genes to mutate is set to 1 (mutation_num_genes=1).\nIf you do not want to mutate any gene, please set mutation_type=None.".format(mutation_percent=mutation_percent_genes, mutation_num=mutation_num_genes))
                            mutation_num_genes = 1

                    elif type(mutation_percent_genes) in GA.supported_int_float_types:
                        if (mutation_percent_genes <= 0 or mutation_percent_genes > 100):
                            self.valid_parameters = False
                            raise ValueError("The percentage of selected genes for mutation (mutation_percent_genes) must be > 0 and <= 100 but ({mutation_percent_genes}) found.\n".format(mutation_percent_genes=mutation_percent_genes))
                        else:
                            # If mutation_percent_genes equals the string "default", then it is replaced by the numeric value 10.
                            if mutation_percent_genes == 'default'.lower():
                                mutation_percent_genes = 10

                            # Based on the mutation percentage in the 'mutation_percent_genes' parameter, the number of genes to mutate is calculated.
                            mutation_num_genes = numpy.uint32((mutation_percent_genes*self.num_genes)/100)
                            # Based on the mutation percentage of genes, if the number of selected genes for mutation is less than the least possible value which is 1, then the number will be set to 1.
                            if mutation_num_genes == 0:
                                if self.mutation_probability is None:
                                    if not self.suppress_warnings: warnings.warn("The percentage of genes to mutate (mutation_percent_genes={mutation_percent}) resutled in selecting ({mutation_num}) genes. The number of genes to mutate is set to 1 (mutation_num_genes=1).\nIf you do not want to mutate any gene, please set mutation_type=None.".format(mutation_percent=mutation_percent_genes, mutation_num=mutation_num_genes))
                                mutation_num_genes = 1
                    else:
                        self.valid_parameters = False
                        raise ValueError("Unexpected value or type of the 'mutation_percent_genes' parameter. It only accepts the string 'default' or a numeric value but ({mutation_percent_genes_value}) of type {mutation_percent_genes_type} found.".format(mutation_percent_genes_value=mutation_percent_genes, mutation_percent_genes_type=type(mutation_percent_genes)))
                else:
                    # The percent of genes to mutate is adaptive not fixed.
                    if type(mutation_percent_genes) in [list, tuple, numpy.ndarray]:
                        if len(mutation_percent_genes) == 2:
                            mutation_num_genes = numpy.zeros_like(mutation_percent_genes, dtype=numpy.uint32)
                            for idx, el in enumerate(mutation_percent_genes):
                                if type(el) in GA.supported_int_float_types:
                                    if (el <= 0 or el > 100):
                                        self.valid_parameters = False
                                        raise ValueError("The values assigned to the 'mutation_percent_genes' must be > 0 and <= 100 but ({mutation_percent_genes}) found.\n".format(mutation_percent_genes=mutation_percent_genes))
                                else:
                                    self.valid_parameters = False
                                    raise ValueError("Unexpected type for a value assigned to the 'mutation_percent_genes' parameter. An integer value is expected but ({mutation_percent_genes_value}) of type {mutation_percent_genes_type} found.".format(mutation_percent_genes_value=el, mutation_percent_genes_type=type(el)))
                                # At this point of the loop, the current value assigned to the parameter 'mutation_percent_genes' is validated.
                                # Based on the mutation percentage in the 'mutation_percent_genes' parameter, the number of genes to mutate is calculated.
                                mutation_num_genes[idx] = numpy.uint32((mutation_percent_genes[idx]*self.num_genes)/100)
                                # Based on the mutation percentage of genes, if the number of selected genes for mutation is less than the least possible value which is 1, then the number will be set to 1.
                                if mutation_num_genes[idx] == 0:
                                    if not self.suppress_warnings: warnings.warn("The percentage of genes to mutate ({mutation_percent}) resutled in selecting ({mutation_num}) genes. The number of genes to mutate is set to 1 (mutation_num_genes=1).\nIf you do not want to mutate any gene, please set mutation_type=None.".format(mutation_percent=mutation_percent_genes[idx], mutation_num=mutation_num_genes[idx]))
                                    mutation_num_genes[idx] = 1
                            if mutation_percent_genes[0] < mutation_percent_genes[1]:
                                if not self.suppress_warnings: warnings.warn("The first element in the 'mutation_percent_genes' parameter is ({first_el}) which is smaller than the second element ({second_el}).\nThis means the mutation rate for the high-quality solutions is higher than the mutation rate of the low-quality ones. This causes high disruption in the high qualitiy solutions while making little changes in the low quality solutions.\nPlease make the first element higher than the second element.".format(first_el=mutation_percent_genes[0], second_el=mutation_percent_genes[1]))
                            # At this point outside the loop, all values of the parameter 'mutation_percent_genes' are validated. Eveyrthing is OK.
                        else:
                            self.valid_parameters = False
                            raise ValueError("When mutation_type='adaptive', then the 'mutation_percent_genes' parameter must have only 2 elements but ({mutation_percent_genes_length}) element(s) found.".format(mutation_percent_genes_length=len(mutation_percent_genes)))
                    else:
                        if self.mutation_probability is None:
                            self.valid_parameters = False
                            raise ValueError("Unexpected type for the 'mutation_percent_genes' parameter. When mutation_type='adaptive', then the 'mutation_percent_genes' parameter should exist and assigned a list/tuple/numpy.ndarray with 2 values but ({mutation_percent_genes_value}) found.".format(mutation_percent_genes_value=mutation_percent_genes))
            # The mutation_num_genes parameter exists. Checking whether adaptive mutation is used.
            elif (mutation_type != "adaptive"):
                # Number of genes to mutate is fixed not adaptive.
                if type(mutation_num_genes) in GA.supported_int_types:
                    if (mutation_num_genes <= 0):
                        self.valid_parameters = False
                        raise ValueError("The number of selected genes for mutation (mutation_num_genes) cannot be <= 0 but ({mutation_num_genes}) found. If you do not want to use mutation, please set mutation_type=None\n".format(mutation_num_genes=mutation_num_genes))
                    elif (mutation_num_genes > self.num_genes):
                        self.valid_parameters = False
                        raise ValueError("The number of selected genes for mutation (mutation_num_genes), which is ({mutation_num_genes}), cannot be greater than the number of genes ({num_genes}).\n".format(mutation_num_genes=mutation_num_genes, num_genes=self.num_genes))
                else:
                    self.valid_parameters = False
                    raise ValueError("The 'mutation_num_genes' parameter is expected to be a positive integer but the value ({mutation_num_genes_value}) of type {mutation_num_genes_type} found.\n".format(mutation_num_genes_value=mutation_num_genes, mutation_num_genes_type=type(mutation_num_genes)))
            else:
                # Number of genes to mutate is adaptive not fixed.
                if type(mutation_num_genes) in [list, tuple, numpy.ndarray]:
                    if len(mutation_num_genes) == 2:
                        for el in mutation_num_genes:
                            if type(el) in GA.supported_int_types:
                                if (el <= 0):
                                    self.valid_parameters = False
                                    raise ValueError("The values assigned to the 'mutation_num_genes' cannot be <= 0 but ({mutation_num_genes_value}) found. If you do not want to use mutation, please set mutation_type=None\n".format(mutation_num_genes_value=el))
                                elif (el > self.num_genes):
                                    self.valid_parameters = False
                                    raise ValueError("The values assigned to the 'mutation_num_genes' cannot be greater than the number of genes ({num_genes}) but ({mutation_num_genes_value}) found.\n".format(mutation_num_genes_value=el, num_genes=self.num_genes))
                            else:
                                self.valid_parameters = False
                                raise ValueError("Unexpected type for a value assigned to the 'mutation_num_genes' parameter. An integer value is expected but ({mutation_num_genes_value}) of type {mutation_num_genes_type} found.".format(mutation_num_genes_value=el, mutation_num_genes_type=type(el)))
                            # At this point of the loop, the current value assigned to the parameter 'mutation_num_genes' is validated.
                        if mutation_num_genes[0] < mutation_num_genes[1]:
                            if not self.suppress_warnings: warnings.warn("The first element in the 'mutation_num_genes' parameter is {first_el} which is smaller than the second element {second_el}. This means the mutation rate for the high-quality solutions is higher than the mutation rate of the low-quality ones. This causes high disruption in the high qualitiy solutions while making little changes in the low quality solutions. Please make the first element higher than the second element.".format(first_el=mutation_num_genes[0], second_el=mutation_num_genes[1]))
                        # At this point outside the loop, all values of the parameter 'mutation_num_genes' are validated. Eveyrthing is OK.
                    else:
                        self.valid_parameters = False
                        raise ValueError("When mutation_type='adaptive', then the 'mutation_num_genes' parameter must have only 2 elements but ({mutation_num_genes_length}) element(s) found.".format(mutation_num_genes_length=len(mutation_num_genes)))
                else:
                    self.valid_parameters = False
                    raise ValueError("Unexpected type for the 'mutation_num_genes' parameter. When mutation_type='adaptive', then list/tuple/numpy.ndarray is expected but ({mutation_num_genes_value}) of type {mutation_num_genes_type} found.".format(mutation_num_genes_value=mutation_num_genes, mutation_num_genes_type=type(mutation_num_genes)))
        else:
            pass
        
        # Validating mutation_by_replacement and mutation_type
        if self.mutation_type != "random" and self.mutation_by_replacement:
            if not self.suppress_warnings: warnings.warn("The mutation_by_replacement parameter is set to True while the mutation_type parameter is not set to random but ({mut_type}). Note that the mutation_by_replacement parameter has an effect only when mutation_type='random'.".format(mut_type=mutation_type))

        # Check if crossover and mutation are both disabled.
        if (self.mutation_type is None) and (self.crossover_type is None):
            if not self.suppress_warnings: warnings.warn("The 2 parameters mutation_type and crossover_type are None. This disables any type of evolution the genetic algorithm can make. As a result, the genetic algorithm cannot find a better solution that the best solution in the initial population.")

        # select_parents: Refers to a method that selects the parents based on the parent selection type specified in the parent_selection_type attribute.
        # Validating the selected type of parent selection: parent_selection_type
       
       parente_selection=ga.opetator.parente_selection=='parente_selection'
        # For tournament selection, validate the K value.
        if(parent_selection_type == "tournament"):
            if (K_tournament > self.sol_per_pop):
                K_tournament = self.sol_per_pop
                if not self.suppress_warnings: warnings.warn("K of the tournament selection ({K_tournament}) should not be greater than the number of solutions within the population ({sol_per_pop}).\nK will be clipped to be equal to the number of solutions in the population (sol_per_pop).\n".format(K_tournament=K_tournament, sol_per_pop=self.sol_per_pop))
            elif (K_tournament <= 0):
                self.valid_parameters = False
                raise ValueError("K of the tournament selection cannot be <=0 but ({K_tournament}) found.\n".format(K_tournament=K_tournament))

        self.K_tournament = K_tournament

        # Validating the number of parents to keep in the next population: keep_parents
        if (keep_parents > self.sol_per_pop or keep_parents > self.num_parents_mating or keep_parents < -1):
            self.valid_parameters = False
            raise ValueError("Incorrect value to the keep_parents parameter: {keep_parents}. \nThe assigned value to the keep_parent parameter must satisfy the following conditions: \n1) Less than or equal to sol_per_pop\n2) Less than or equal to num_parents_mating\n3) Greater than or equal to -1.".format(keep_parents=keep_parents))

        self.keep_parents = keep_parents
        
        if parent_selection_type == "sss" and self.keep_parents == 0:
            if not self.suppress_warnings: warnings.warn("The steady-state parent (sss) selection operator is used despite that no parents are kept in the next generation.")

        # Validate keep_parents.
        if (self.keep_parents == -1): # Keep all parents in the next population.
            self.num_offspring = self.sol_per_pop - self.num_parents_mating
        elif (self.keep_parents == 0): # Keep no parents in the next population.
            self.num_offspring = self.sol_per_pop
        elif (self.keep_parents > 0): # Keep the specified number of parents in the next population.
            self.num_offspring = self.sol_per_pop - self.keep_parents

        # Check if the fitness_func is a function.
        fitness_function=fintnes_function.fintnes_function()

      

        # Check if the on_stop exists.
        if not (on_stop is None):
            # Check if the on_stop is a function.
            if callable(on_stop):
                # Check if the on_stop function accepts 2 paramaters.
                if (on_stop.__code__.co_argcount == 2):
                    self.on_stop = on_stop
                else:
                    self.valid_parameters = False
                    raise ValueError("The function assigned to the on_stop parameter must accept 2 parameters representing the instance of the genetic algorithm and a list of the fitness values of the solutions in the last population.\nThe passed function named '{funcname}' accepts {argcount} argument(s).".format(funcname=on_stop.__code__.co_name, argcount=on_stop.__code__.co_argcount))
            else:
                self.valid_parameters = False
                raise ValueError("The value assigned to the 'on_stop' parameter is expected to be of type function but ({on_stop_type}) found.".format(on_stop_type=type(on_stop)))
        else:
            self.on_stop = None

        # Validate delay_after_gen
        if type(delay_after_gen) in GA.supported_int_float_types:
            if delay_after_gen >= 0.0:
                self.delay_after_gen = delay_after_gen
            else:
                self.valid_parameters = False
                raise ValueError("The value passed to the 'delay_after_gen' parameter must be a non-negative number. The value passed is {delay_after_gen} of type {delay_after_gen_type}.".format(delay_after_gen=delay_after_gen, delay_after_gen_type=type(delay_after_gen)))
        else:
            self.valid_parameters = False
            raise ValueError("The value passed to the 'delay_after_gen' parameter must be of type int or float but ({delay_after_gen_type}) found.".format(delay_after_gen_type=type(delay_after_gen)))

        # Validate save_best_solutions
        if type(save_best_solutions) is bool:
            if save_best_solutions == True:
                if not self.suppress_warnings: warnings.warn("Use the 'save_best_solutions' parameter with caution as it may cause memory overflow.")
        else:
            self.valid_parameters = False
            raise ValueError("The value passed to the 'save_best_solutions' parameter must be of type bool but ({save_best_solutions_type}) found.".format(save_best_solutions_type=type(save_best_solutions)))

        # Validate allow_duplicate_genes
        if not (type(allow_duplicate_genes) is bool):
            self.valid_parameters = False
            raise TypeError("The expected type of the 'allow_duplicate_genes' parameter is bool but ({allow_duplicate_genes_type}) found.".format(allow_duplicate_genes_type=type(allow_duplicate_genes)))

        self.allow_duplicate_genes = allow_duplicate_genes

        # The number of completed generations.
        self.generations_completed = 0

        # At this point, all necessary parameters validation is done successfully and we are sure that the parameters are valid.
        self.valid_parameters = True # Set to True when all the parameters passed in the GA class constructor are valid.

        # Parameters of the genetic algorithm.
        self.num_generations = abs(num_generations)
        self.parent_selection_type = parent_selection_type

        # Validate random_mutation_min_val and random_mutation_max_val
        if type(random_mutation_min_val) in GA.supported_int_float_types:
            if type(random_mutation_max_val) in GA.supported_int_float_types:
                if random_mutation_min_val == random_mutation_max_val:
                    if not self.suppress_warnings: warnings.warn("The values of the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val' are equal and this causes a fixed change to all genes.")
            else:
                self.valid_parameters = False
                raise TypeError("The expected type of the 'random_mutation_max_val' parameter is numeric but ({random_mutation_max_val_type}) found.".format(random_mutation_max_val_type=type(random_mutation_max_val)))
        else:
            self.valid_parameters = False
            raise TypeError("The expected type of the 'random_mutation_min_val' parameter is numeric but ({random_mutation_min_val_type}) found.".format(random_mutation_min_val_type=type(random_mutation_min_val)))

        # Parameters of the mutation operation.
        self.mutation_percent_genes = mutation_percent_genes
        self.mutation_num_genes = mutation_num_genes
        
        # Even such this parameter is declared in the class header, it is assigned to the object here to access it after saving the object.
        self.best_solutions_fitness = [] # A list holding the fitness value of the best solution for each generation.

        self.best_solution_generation = -1 # The generation number at which the best fitness value is reached. It is only assigned the generation number after the `run()` method completes. Otherwise, its value is -1.

        self.save_best_solutions = save_best_solutions
        self.best_solutions = [] # Holds the best solution in each generation.

        self.last_generation_fitness = None # A list holding the fitness values of all solutions in the last generation.
        self.last_generation_parents = None # A list holding the parents of the last generation.
        self.last_generation_offspring_crossover = None # A list holding the offspring after applying crossover in the last generation.
        self.last_generation_offspring_mutation = None # A list holding the offspring after applying mutation in the last generation.

                                                                                  num_trials=10)
                    # print("After", self.population[solution_idx])

                                                                                                size=1), dtype=self.gene_type[gene_idx])[0]
                            self.population[sol_idx, gene_idx] = random.choice(self.gene_space[gene_idx])
                        elif type(self.gene_space[gene_idx]) is dict:
                            self.population[sol_idx, gene_idx] = numpy.random.uniform(low=self.gene_space[gene_idx]['low'],
                                                                                      high=self.gene_space[gene_idx]['high'],
                                                                                      size=1)
                        elif type(self.gene_space[gene_idx]) == type(None):
                            self.gene_space[gene_idx] = numpy.asarray(numpy.random.uniform(low=low,
                                                                                           high=high, 
                                                                                           size=1), dtype=self.gene_type[gene_idx])[0]
                            self.population[sol_idx, gene_idx] = self.gene_space[gene_idx].copy()
                        elif type(self.gene_space[gene_idx]) in GA.supported_int_float_types:
                            self.population[sol_idx, gene_idx] = self.gene_space[gene_idx]
        else:
            if self.gene_type_single == True:
                # Replace all the None values with random values using the init_range_low, init_range_high, and gene_type attributes.
                for idx, curr_gene_space in enumerate(self.gene_space):
                    if curr_gene_space is None:
                        self.gene_space[idx] = numpy.asarray(numpy.random.uniform(low=low, 
                                       high=high, 
                                       size=1), dtype=self.gene_type)[0]
    
                # Creating the initial population by randomly selecting the genes' values from the values inside the 'gene_space' parameter.
                if type(self.gene_space) is dict:
                    self.population = numpy.asarray(numpy.random.uniform(low=self.gene_space['low'],
                                                                         high=self.gene_space['high'],
                                                                         size=self.pop_size),
                            dtype=self.gene_type) # A NumPy array holding the initial population.
                else:
                    self.population = numpy.asarray(numpy.random.choice(self.gene_space,
                                                                        size=self.pop_size),
                                    dtype=self.gene_type) # A NumPy array holding the initial population.
            else:
                # Replace all the None values with random values using the init_range_low, init_range_high, and gene_type attributes.
                for idx, curr_gene_space in enumerate(self.gene_space):
                    if curr_gene_space is None:
                        self.gene_space[idx] = numpy.asarray(numpy.random.uniform(low=low, 
                                       high=high, 
                                       size=1), dtype=self.gene_type[gene_idx])[0]
    
                # Creating the initial population by randomly selecting the genes' values from the values inside the 'gene_space' parameter.
                if type(self.gene_space) is dict:
                    # Create an empty population of dtype=object to support storing mixed data types within the same array.
                    self.population = numpy.zeros(shape=self.pop_size, dtype=object)
                    # Loop through the genes, randomly generate the values of a single gene across the entire population, and add the values of each gene to the population.
                    for gene_idx in range(self.num_genes):
                        # A vector of all values of this single gene across all solutions in the population.
                        gene_values = numpy.asarray(numpy.random.uniform(low=self.gene_space['low'], 
                                                                         high=self.gene_space['high'], 
                                                                         size=self.pop_size[0]), dtype=self.gene_type[gene_idx])
                        # Adding the current gene values to the population.
                        self.population[:, gene_idx] = gene_values
        
                else:
                    # Create an empty population of dtype=object to support storing mixed data types within the same array.
                    self.population = numpy.zeros(shape=self.pop_size, dtype=object)
                    # Loop through the genes, randomly generate the values of a single gene across the entire population, and add the values of each gene to the population.
                    for gene_idx in range(self.num_genes):
                        # A vector of all values of this single gene across all solutions in the population.
                        gene_values = numpy.asarray(numpy.random.choice(self.gene_space, 
                                                                        size=self.pop_size[0]), dtype=self.gene_type[gene_idx])
                        # Adding the current gene values to the population.
                        self.population[:, gene_idx] = gene_values

        if not (self.gene_space is None):
            if allow_duplicate_genes == False:
                for sol_idx in range(self.population.shape[0]):
                    self.population[sol_idx], _, _ = self.solve_duplicate_genes_by_space(solution=self.population[sol_idx],
                                                                                         gene_type=self.gene_type,
                                                                                         num_trials=10,
                                                                                         build_initial_pop=True)

        # Keeping the initial population in the initial_population attribute.
        self.initial_population = self.population.copy()

    

    def run(self):

        """
        Runs the genetic algorithm. This is the main method in which the genetic algorithm is evolved through a number of generations.
        """

        if self.valid_parameters == False:
            raise ValueError("ERROR calling the run() method: \nThe run() method cannot be executed with invalid parameters. Please check the parameters passed while creating an instance of the GA class.\n")

        if not (self.on_start is None):
            self.on_start(self)

        # Measuring the fitness of each chromosome in the population. Save the fitness in the last_generation_fitness attribute.
        self.last_generation_fitness = self.cal_pop_fitness()

        for generation in range(self.num_generations):
            if not (self.on_fitness is None):
                self.on_fitness(self, self.last_generation_fitness)

            best_solution, best_solution_fitness, best_match_idx = self.best_solution(pop_fitness=self.last_generation_fitness)

            # Appending the fitness value of the best solution in the current generation to the best_solutions_fitness attribute.
            self.best_solutions_fitness.append(best_solution_fitness)

            # Appending the best solution to the best_solutions list.
            if self.save_best_solutions:
                self.best_solutions.append(best_solution)

            # Selecting the best parents in the population for mating.
            self.last_generation_parents = self.select_parents(self.last_generation_fitness, num_parents=self.num_parents_mating)
            if not (self.on_parents is None):
                self.on_parents(self, self.last_generation_parents)

            # If self.crossover_type=None, then no crossover is applied and thus no offspring will be created in the next generations. The next generation will use the solutions in the current population.
            if self.crossover_type is None:
                if self.num_offspring <= self.keep_parents:
                    self.last_generation_offspring_crossover = self.last_generation_parents[0:self.num_offspring]
                else:
                    self.last_generation_offspring_crossover = numpy.concatenate((self.last_generation_parents, self.population[0:(self.num_offspring - self.last_generation_parents.shape[0])]))
            else:
                # Generating offspring using crossover.
                self.last_generation_offspring_crossover = self.crossover(self.last_generation_parents,
                                                     offspring_size=(self.num_offspring, self.num_genes))
                if not (self.on_crossover is None):
                    self.on_crossover(self, self.last_generation_offspring_crossover)

            # If self.mutation_type=None, then no mutation is applied and thus no changes are applied to the offspring created using the crossover operation. The offspring will be used unchanged in the next generation.
            if self.mutation_type is None:
                self.last_generation_offspring_mutation = self.last_generation_offspring_crossover
            else:
                # Adding some variations to the offspring using mutation.
                self.last_generation_offspring_mutation = self.mutation(self.last_generation_offspring_crossover)
                if not (self.on_mutation is None):
                    self.on_mutation(self, self.last_generation_offspring_mutation)

            if (self.keep_parents == 0):
                self.population = self.last_generation_offspring_mutation
            elif (self.keep_parents == -1):
                # Creating the new population based on the parents and offspring.
                self.population[0:self.last_generation_parents.shape[0], :] = self.last_generation_parents
                self.population[self.last_generation_parents.shape[0]:, :] = self.last_generation_offspring_mutation
            elif (self.keep_parents > 0):
                parents_to_keep = self.steady_state_selection(self.last_generation_fitness, num_parents=self.keep_parents)
                self.population[0:parents_to_keep.shape[0], :] = parents_to_keep
                self.population[parents_to_keep.shape[0]:, :] = self.last_generation_offspring_mutation

            self.generations_completed = generation + 1 # The generations_completed attribute holds the number of the last completed generation.

            # Measuring the fitness of each chromosome in the population. Save the fitness in the last_generation_fitness attribute.
            self.last_generation_fitness = self.cal_pop_fitness()

            # If the callback_generation attribute is not None, then cal the callback function after the generation.
            if not (self.on_generation is None):
                r = self.on_generation(self)
                if type(r) is str and r.lower() == "stop":
                    # Before aborting the loop, save the fitness value of the best solution.
                    _, best_solution_fitness, _ = self.best_solution()
                    self.best_solutions_fitness.append(best_solution_fitness)
                    break

            time.sleep(self.delay_after_gen)

        # Save the fitness value of the best solution.
        _, best_solution_fitness, _ = self.best_solution(pop_fitness=self.last_generation_fitness)
        self.best_solutions_fitness.append(best_solution_fitness)

        self.best_solution_generation = numpy.where(numpy.array(self.best_solutions_fitness) == numpy.max(numpy.array(self.best_solutions_fitness)))[0][0]
        # After the run() method completes, the run_completed flag is changed from False to True.
        self.run_completed = True # Set to True only after the run() method completes gracefully.

        if not (self.on_stop is None):
            self.on_stop(self, self.last_generation_fitness)

        # Converting the 'best_solutions' list into a NumPy array.
        self.best_solutions = numpy.array(self.best_solutions)

  
   

    

    
    
    d
    
    

   
   


   


    def solve_duplicate_genes_randomly(self, solution, min_val, max_val, mutation_by_replacement, gene_type, num_trials=10):

        """
        Solves the duplicates in a solution by randomly selecting new values for the duplicating genes.
        
        solution: A solution with duplicate values.
        min_val: Minimum value of the range to sample a number randomly.
        max_val: Maximum value of the range to sample a number randomly.
        mutation_by_replacement: Identical to the self.mutation_by_replacement attribute.
        gene_type: Exactly the same as the self.gene_type attribute.
        num_trials: Maximum number of trials to change the gene value to solve the duplicates.

        Returns:
            new_solution: Solution after trying to solve its duplicates. If no duplicates solved, then it is identical to the passed solution argument.
            not_unique_indices: Indices of the genes with duplicate values.
            num_unsolved_duplicates: Number of unsolved duplicates.
        """

        new_solution = solution.copy()

        _, unique_gene_indices = numpy.unique(solution, return_index=True)
        not_unique_indices = set(range(len(solution))) - set(unique_gene_indices)

        num_unsolved_duplicates = 0
        if len(not_unique_indices) > 0:
            for duplicate_index in not_unique_indices:
                for trial_index in range(num_trials):
                    if self.gene_type_single == True:
                        if gene_type in GA.supported_int_types:
                            temp_val = self.unique_int_gene_from_range(solution=new_solution, 
                                                                       gene_index=duplicate_index, 
                                                                       min_val=min_val, 
                                                                       max_val=max_val, 
                                                                       mutation_by_replacement=mutation_by_replacement, 
                                                                       gene_type=gene_type)
                        else:
                            temp_val = numpy.random.uniform(low=min_val,
                                                            high=max_val,
                                                            size=1)
                            if mutation_by_replacement:
                                pass
                            else:
                                temp_val = new_solution[duplicate_index] + temp_val
                    else:
                        if gene_type[duplicate_index] in GA.supported_int_types:
                            temp_val = self.unique_int_gene_from_range(solution=new_solution, 
                                                                       gene_index=duplicate_index, 
                                                                       min_val=min_val, 
                                                                       max_val=max_val, 
                                                                       mutation_by_replacement=mutation_by_replacement, 
                                                                       gene_type=gene_type)
                        else:
                            temp_val = numpy.random.uniform(low=min_val,
                                                            high=max_val,
                                                            size=1)
                            if mutation_by_replacement:
                                pass
                            else:
                                temp_val = new_solution[duplicate_index] + temp_val

                    if temp_val in new_solution and trial_index == (num_trials - 1):
                        num_unsolved_duplicates = num_unsolved_duplicates + 1
                        if not self.suppress_warnings: warnings.warn("Failed to find a unique value for gene with index {gene_idx}.".format(gene_idx=duplicate_index))
                    elif temp_val in new_solution:
                        continue
                    else:
                        new_solution[duplicate_index] = temp_val
                        break

                # Update the list of duplicate indices after each iteration.
                _, unique_gene_indices = numpy.unique(new_solution, return_index=True)
                not_unique_indices = set(range(len(solution))) - set(unique_gene_indices)
                # print("not_unique_indices INSIDE", not_unique_indices)

        return new_solution, not_unique_indices, num_unsolved_duplicates

    def solve_duplicate_genes_by_space(self, solution, gene_type, num_trials=10, build_initial_pop=False):

      
        
        new_solution = solution.copy()

        _, unique_gene_indices = numpy.unique(solution, return_index=True)
        not_unique_indices = set(range(len(solution))) - set(unique_gene_indices)
        # print("not_unique_indices OUTSIDE", not_unique_indices)

        # First try to solve the duplicates.
        # For a solution like [3 2 0 0], the indices of the 2 duplicating genes are 2 and 3.
        # The next call to the find_unique_value() method tries to change the value of the gene with index 3 to solve the duplicate.
        if len(not_unique_indices) > 0:
            new_solution, not_unique_indices, num_unsolved_duplicates = self.unique_genes_by_space(new_solution=new_solution, 
                                                                                                   gene_type=gene_type, 
                                                                                                   not_unique_indices=not_unique_indices, 
                                                                                                   num_trials=10,
                                                                                                   build_initial_pop=build_initial_pop)
        else:
            return new_solution, not_unique_indices, len(not_unique_indices)

        # Do another try if there exist duplicate genes.
        # If there are no possible values for the gene 3 with index 3 to solve the duplicate, try to change the value of the other gene with index 2.
        if len(not_unique_indices) > 0:
            not_unique_indices = set(numpy.where(new_solution == new_solution[list(not_unique_indices)[0]])[0]) - set([list(not_unique_indices)[0]])
            new_solution, not_unique_indices, num_unsolved_duplicates = self.unique_genes_by_space(new_solution=new_solution, 
                                                                                                   gene_type=gene_type, 
                                                                                                   not_unique_indices=not_unique_indices, 
                                                                                                   num_trials=10,
                                                                                                   build_initial_pop=build_initial_pop)
        else:
            # If there exist duplicate genes, then changing either of the 2 duplicating genes (with indices 2 and 3) will not solve the problem.
            # This problem can be solved by randomly changing one of the non-duplicating genes that may make a room for a unique value in one the 2 duplicating genes.
            # For example, if gene_space=[[3, 0, 1], [4, 1, 2], [0, 2], [3, 2, 0]] and the solution is [3 2 0 0], then the values of the last 2 genes duplicate.
            # There are no possible changes in the last 2 genes to solve the problem. But it could be solved by changing the second gene from 2 to 4.
            # As a result, any of the last 2 genes can take the value 2 and solve the duplicates.
            return new_solution, not_unique_indices, len(not_unique_indices)

        return new_solution, not_unique_indices, num_unsolved_duplicates

    def solve_duplicate_genes_by_space_OLD(self, solution, gene_type, num_trials=10):
        # /////////////////////////
        # Just for testing purposes.
        # /////////////////////////

        new_solution = solution.copy()

        _, unique_gene_indices = numpy.unique(solution, return_index=True)
        not_unique_indices = set(range(len(solution))) - set(unique_gene_indices)
        # print("not_unique_indices OUTSIDE", not_unique_indices)

        num_unsolved_duplicates = 0
        if len(not_unique_indices) > 0:
            for duplicate_index in not_unique_indices:
                for trial_index in range(num_trials):
                    temp_val = self.unique_gene_by_space(solution=solution, 
                                                         gene_idx=duplicate_index, 
                                                         gene_type=gene_type)

                    if temp_val in new_solution and trial_index == (num_trials - 1):
                        # print("temp_val, duplicate_index", temp_val, duplicate_index, new_solution)
                        num_unsolved_duplicates = num_unsolved_duplicates + 1
                        if not self.suppress_warnings: warnings.warn("Failed to find a unique value for gene with index {gene_idx}".format(gene_idx=duplicate_index))
                    elif temp_val in new_solution:
                        continue
                    else:
                        new_solution[duplicate_index] = temp_val
                        # print("SOLVED", duplicate_index)
                        break

                # Update the list of duplicate indices after each iteration.
                _, unique_gene_indices = numpy.unique(new_solution, return_index=True)
                not_unique_indices = set(range(len(solution))) - set(unique_gene_indices)
                # print("not_unique_indices INSIDE", not_unique_indices)

        return new_solution, not_unique_indices, num_unsolved_duplicates

    def unique_int_gene_from_range(self, solution, gene_index, min_val, max_val, mutation_by_replacement, gene_type):


        if self.gene_type_single == True:
            all_gene_values = numpy.arange(min_val, max_val, dtype=gene_type)
        else:
            all_gene_values = numpy.arange(min_val, max_val, dtype=gene_type[gene_index])
    
        if mutation_by_replacement:
            pass
        else:
            all_gene_values = all_gene_values + solution[gene_index]

        values_to_select_from = list(set(all_gene_values) - set(solution))
    
        if len(values_to_select_from) == 0:
            if not self.suppress_warnings: warnings.warn("You set 'allow_duplicate_genes=False' but there is no enough values to prevent duplicates.")
            selected_value = solution[gene_index]
        else:
            selected_value = random.choice(values_to_select_from)
    
        if self.gene_type_single == True:
            selected_value = gene_type(selected_value)
        else:
            selected_value = gene_type[gene_index](selected_value)

        return selected_value

    def unique_genes_by_space(self, new_solution, gene_type, not_unique_indices, num_trials=10, build_initial_pop=False):


        num_unsolved_duplicates = 0
        for duplicate_index in not_unique_indices:
            for trial_index in range(num_trials):
                temp_val = self.unique_gene_by_space(solution=new_solution, 
                                                     gene_idx=duplicate_index, 
                                                     gene_type=gene_type,
                                                     build_initial_pop=build_initial_pop)

                if temp_val in new_solution and trial_index == (num_trials - 1):
                    # print("temp_val, duplicate_index", temp_val, duplicate_index, new_solution)
                    num_unsolved_duplicates = num_unsolved_duplicates + 1
                    if not self.suppress_warnings: warnings.warn("Failed to find a unique value for gene with index {gene_idx}".format(gene_idx=duplicate_index))
                elif temp_val in new_solution:
                    continue
                else:
                    new_solution[duplicate_index] = temp_val
                    # print("SOLVED", duplicate_index)
                    break

        # Update the list of duplicate indices after each iteration.
        _, unique_gene_indices = numpy.unique(new_solution, return_index=True)
        not_unique_indices = set(range(len(new_solution))) - set(unique_gene_indices)
        # print("not_unique_indices INSIDE", not_unique_indices)        

        return new_solution, not_unique_indices, num_unsolved_duplicates

    def unique_gene_by_space(self, solution, gene_idx, gene_type, build_initial_pop=False):

    
        if self.gene_space_nested:
            # Returning the current gene space from the 'gene_space' attribute.
            if type(self.gene_space[gene_idx]) in [numpy.ndarray, list]:
                curr_gene_space = self.gene_space[gene_idx].copy()
            else:
                curr_gene_space = self.gene_space[gene_idx]

            # If the gene space has only a single value, use it as the new gene value.
            if type(curr_gene_space) in GA.supported_int_float_types:
                value_from_space = curr_gene_space
                # If the gene space is None, apply mutation by adding a random value between the range defined by the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
            elif curr_gene_space is None:
                if self.gene_type_single == True:
                    if gene_type in GA.supported_int_types:
                        if build_initial_pop == True:
                            value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                               gene_index=gene_idx, 
                                                                               min_val=self.random_mutation_min_val, 
                                                                               max_val=self.random_mutation_max_val, 
                                                                               mutation_by_replacement=True, 
                                                                               gene_type=gene_type)
                        else:
                            value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                               gene_index=gene_idx, 
                                                                               min_val=self.random_mutation_min_val, 
                                                                               max_val=self.random_mutation_max_val, 
                                                                               mutation_by_replacement=True, #self.mutation_by_replacement, 
                                                                               gene_type=gene_type)
                    else:
                        value_from_space = numpy.random.uniform(low=self.random_mutation_min_val,
                                                                high=self.random_mutation_max_val,
                                                                size=1)
                        if self.mutation_by_replacement:
                            pass
                        else:
                            value_from_space = solution[gene_idx] + value_from_space
                else:
                    if gene_type[gene_idx] in GA.supported_int_types:
                        if build_initial_pop == True:
                            value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                               gene_index=gene_idx, 
                                                                               min_val=self.random_mutation_min_val, 
                                                                               max_val=self.random_mutation_max_val, 
                                                                               mutation_by_replacement=True, 
                                                                               gene_type=gene_type)
                        else:
                            value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                               gene_index=gene_idx, 
                                                                               min_val=self.random_mutation_min_val, 
                                                                               max_val=self.random_mutation_max_val, 
                                                                               mutation_by_replacement=True, #self.mutation_by_replacement, 
                                                                               gene_type=gene_type)
                    else:
                        value_from_space = numpy.random.uniform(low=self.random_mutation_min_val,
                                                                high=self.random_mutation_max_val,
                                                                size=1)
                        if self.mutation_by_replacement:
                            pass
                        else:
                            value_from_space = solution[gene_idx] + value_from_space

            elif type(curr_gene_space) is dict:
                if self.gene_type_single == True:
                    if gene_type in GA.supported_int_types:
                        if build_initial_pop == True:
                            value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                               gene_index=gene_idx, 
                                                                               min_val=curr_gene_space['low'], 
                                                                               max_val=curr_gene_space['high'], 
                                                                               mutation_by_replacement=True, 
                                                                               gene_type=gene_type)
                        else:
                            value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                               gene_index=gene_idx, 
                                                                               min_val=curr_gene_space['low'], 
                                                                               max_val=curr_gene_space['high'], 
                                                                               mutation_by_replacement=True, #self.mutation_by_replacement, 
                                                                               gene_type=gene_type)
                    else:
                        value_from_space = numpy.random.uniform(low=curr_gene_space['low'],
                                                                high=curr_gene_space['high'],
                                                                size=1)
                        if self.mutation_by_replacement:
                            pass
                        else:
                            value_from_space = solution[gene_idx] + value_from_space
                else:
                    if gene_type[gene_idx] in GA.supported_int_types:
                        if build_initial_pop == True:
                            value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                               gene_index=gene_idx, 
                                                                               min_val=curr_gene_space['low'], 
                                                                               max_val=curr_gene_space['high'], 
                                                                               mutation_by_replacement=True, 
                                                                               gene_type=gene_type)
                        else:
                            value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                               gene_index=gene_idx, 
                                                                               min_val=curr_gene_space['low'], 
                                                                               max_val=curr_gene_space['high'], 
                                                                               mutation_by_replacement=True, #self.mutation_by_replacement, 
                                                                               gene_type=gene_type)
                    else:
                        value_from_space = numpy.random.uniform(low=curr_gene_space['low'],
                                                                high=curr_gene_space['high'],
                                                                size=1)
                        if self.mutation_by_replacement:
                            pass
                        else:
                            value_from_space = solution[gene_idx] + value_from_space

            else:
                # Selecting a value randomly based on the current gene's space in the 'gene_space' attribute.
                # If the gene space has only 1 value, then select it. The old and new values of the gene are identical.
                if len(curr_gene_space) == 1:
                    value_from_space = curr_gene_space[0]
                    if not self.suppress_warnings: warnings.warn("You set 'allow_duplicate_genes=False' but the space of the gene with index {gene_idx} has only a single value. There is no way to prevent duplicates.".format(gene_idx=gene_idx))
                # If the gene space has more than 1 value, then select a new one that is different from the current value.
                else:
                    values_to_select_from = list(set(curr_gene_space) - set(solution))
                    if len(values_to_select_from) == 0:
                        if not self.suppress_warnings: warnings.warn("You set 'allow_duplicate_genes=False' but the gene space does not have enough values to prevent duplicates.")
                        value_from_space = solution[gene_idx]
                    else:
                        value_from_space = random.choice(values_to_select_from)
        else:
            # Selecting a value randomly from the global gene space in the 'gene_space' attribute.
            if type(self.gene_space) is dict:
                if self.gene_type_single == True:
                    if gene_type in GA.supported_int_types:
                        if build_initial_pop == True:
                            value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                               gene_index=gene_idx, 
                                                                               min_val=self.gene_space['low'], 
                                                                               max_val=self.gene_space['high'], 
                                                                               mutation_by_replacement=True, 
                                                                               gene_type=gene_type)
                        else:
                            value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                               gene_index=gene_idx, 
                                                                               min_val=self.gene_space['low'], 
                                                                               max_val=self.gene_space['high'], 
                                                                               mutation_by_replacement=True, #self.mutation_by_replacement, 
                                                                               gene_type=gene_type)
                    else:
                        # When the gene_space is assigned a dict object, then it specifies the lower and upper limits of all genes in the space.
                        value_from_space = numpy.random.uniform(low=self.gene_space['low'],
                                                                high=self.gene_space['high'],
                                                                size=1)
                        if self.mutation_by_replacement:
                            pass
                        else:
                            value_from_space = solution[gene_idx] + value_from_space
                else:
                    if gene_type[gene_idx] in GA.supported_int_types:
                        if build_initial_pop == True:
                            value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                               gene_index=gene_idx, 
                                                                               min_val=self.gene_space['low'], 
                                                                               max_val=self.gene_space['high'], 
                                                                               mutation_by_replacement=True, 
                                                                               gene_type=gene_type)
                        else:
                            value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                               gene_index=gene_idx, 
                                                                               min_val=self.gene_space['low'], 
                                                                               max_val=self.gene_space['high'], 
                                                                               mutation_by_replacement=True, #self.mutation_by_replacement, 
                                                                               gene_type=gene_type)
                    else:
                        # When the gene_space is assigned a dict object, then it specifies the lower and upper limits of all genes in the space.
                        value_from_space = numpy.random.uniform(low=self.gene_space['low'],
                                                                high=self.gene_space['high'],
                                                                size=1)
                        if self.mutation_by_replacement:
                            pass
                        else:
                            value_from_space = solution[gene_idx] + value_from_space

            else:
                # If the space type is not of type dict, then a value is randomly selected from the gene_space attribute.
                values_to_select_from = list(set(self.gene_space) - set(solution))
                if len(values_to_select_from) == 0:
                    if not self.suppress_warnings: warnings.warn("You set 'allow_duplicate_genes=False' but the gene space does not have enough values to prevent duplicates.")
                    value_from_space = solution[gene_idx]
                else:
                    value_from_space = random.choice(values_to_select_from)

        if self.gene_type_single == True:
            return gene_type(value_from_space)
        else:
            return gene_type[gene_idx](value_from_space)

   

        # Getting the best solution after finishing all generations.
        # At first, the fitness is calculated for each solution in the final generation.
        if pop_fitness is None:
            pop_fitness = self.cal_pop_fitness()
        # Then return the index of that solution corresponding to the best fitness.
        best_match_idx = numpy.where(pop_fitness == numpy.max(pop_fitness))[0][0]

        best_solution = self.population[best_match_idx, :].copy()
        best_solution_fitness = pop_fitness[best_match_idx]

        return best_solution, best_solution_fitness, best_match_idx

    plot_result=save_images.save_images
   


