############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: NSGA-II (Non-Dominated Sorting Genetic Algorithm II)

# Citation: 
# PEREIRA, V. (2021). Project: pyMultiojective, File: n_iii.py, GitHub repository: <https://github.com/Valdecy/pyMultiojective>

############################################################################

# Required Libraries
import copy
import numpy  as np
import random
import os

############################################################################

# Function 1
def func_1():
    return

# Function 2
def func_2():
    return

############################################################################

# Function: Initialize Variables
def initial_population(population_size = 5, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2]):
    population = np.zeros((population_size, len(min_values) + len(list_of_functions)))
    for i in range(0, population_size):
        for j in range(0, len(min_values)):
             population[i,j] = random.uniform(min_values[j], max_values[j])      
        for k in range (1, len(list_of_functions) + 1):
            population[i,-k] = list_of_functions[-k](list(population[i,0:population.shape[1]-len(list_of_functions)]))
    return population

############################################################################

# Function: Fast Non-Dominated Sorting
def fast_non_dominated_sorting(population, number_of_functions = 2):
    S     = [[] for i in range(0, population.shape[0])]
    front = [[]]
    n     = [0 for i in range(0, population.shape[0])]
    rank  = [0 for i in range(0, population.shape[0])]
    for p in range(0, population.shape[0]):
        S[p] = []
        n[p] = 0
        for q in range(0, population.shape[0]):
            if ((population[p,-number_of_functions:] <= population[q,-number_of_functions:]).all()):
                if (q not in S[p]):
                    S[p].append(q)
            elif ((population[q,-number_of_functions:] <= population[p,-number_of_functions:]).all()):
                n[p] = n[p] + 1
        if (n[p] == 0):
            rank[p] = 0
            if (p not in front[0]):
                front[0].append(p)
    i = 0
    while (front[i] != []):
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if(n[q] == 0):
                    rank[q] = i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)
    del front[len(front)-1]
    rank = np.zeros((population.shape[0], 1))
    for i in range(0, len(front)):
        for j in range(0, len(front[i])):
            rank[front[i][j], 0] = i + 1
    return rank

# Function: Sort Population by Rank
def sort_population_by_rank(population, rank):
    idx        = np.argsort(rank[:,0], axis = 0).tolist()
    rank       = rank[idx,:]
    population = population[idx,:]
    return population, rank

# Function: Crowding Distance (Adapted from PYMOO)
def crowding_distance_function(pop, M):
    infinity   = 1e+11
    population = copy.deepcopy(pop[:,-M:])
    population = population.reshape((pop.shape[0], M))
    if (population.shape[0] <= 2):
        return np.full(population.shape[0], infinity)
    else:
        arg_1      = np.argsort(population, axis = 0, kind = 'mergesort')
        population = population[arg_1, np.arange(M)]
        dist       = np.concatenate([population, np.full((1, M), np.inf)]) - np.concatenate([np.full((1, M), -np.inf), population])
        idx        = np.where(dist == 0)
        a          = np.copy(dist)
        b          = np.copy(dist)
        for i, j in zip(*idx):
            a[i, j] = a[i - 1, j]
        for i, j in reversed(list(zip(*idx))):
            b[i, j] = b[i + 1, j]
        norm            = np.max(population, axis = 0) - np.min(population, axis = 0)
        norm[norm == 0] = np.nan
        a, b            = a[:-1]/norm, b[1:]/norm
        a[np.isnan(a)]  = 0.0
        b[np.isnan(b)]  = 0.0
        arg_2           = np.argsort(arg_1, axis = 0)
        crowding        = np.sum(a[arg_2, np.arange(M)] + b[arg_2, np.arange(M)], axis = 1) / M
    crowding[np.isinf(crowding)] = infinity
    crowding                     = crowding.reshape((-1,1))
    return crowding

# Function:Crowded Comparison Operator
def crowded_comparison_operator(rank, crowding_distance, individual_1 = 0, individual_2 = 1):
    selection = False
    if (rank[individual_1,0] < rank[individual_2,0]) or ((rank[individual_1,0] == rank[individual_2,0]) and (crowding_distance[individual_1,0] > crowding_distance[individual_2,0])):
        selection = True      
    return selection

# Function: Offspring
def breeding(population, rank, crowding_distance, min_values = [-5,-5], max_values = [5,5], mu = 1, list_of_functions = [func_1, func_2]):
    offspring   = np.copy(population)
    parent_1    = 0
    parent_2    = 1
    b_offspring = 0
    for i in range (0, offspring.shape[0]):
        i1, i2, i3, i4 = random.sample(range(0, len(population) - 1), 4)
        if (crowded_comparison_operator(rank, crowding_distance, individual_1 = i1, individual_2 = i2) == True):
            parent_1 = i1
        elif (crowded_comparison_operator(rank, crowding_distance, individual_1 = i2, individual_2 = i1) == True):
            parent_1 = i2
        else:
            rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            if (rand > 0.5):
                parent_1 = i1
            else:
                parent_1 = i2
        if (crowded_comparison_operator(rank, crowding_distance, individual_1 = i3, individual_2 = i4) == True):
            parent_2 = i3
        elif (crowded_comparison_operator(rank, crowding_distance, individual_1 = i4, individual_2 = i3) == True):
            parent_2 = i4
        else:
            rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            if (rand > 0.5):
                parent_2 = i3
            else:
                parent_2 = i4
        for j in range(0, offspring.shape[1] - len(list_of_functions)):
            rand   = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            rand_b = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)  
            rand_c = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)                              
            if (rand <= 0.5):
                b_offspring = 2*(rand_b)
                b_offspring = b_offspring**(1/(mu + 1))
            elif (rand > 0.5):  
                b_offspring = 1/(2*(1 - rand_b))
                b_offspring = b_offspring**(1/(mu + 1))       
            if (rand_c >= 0.5):
                offspring[i,j] = np.clip(((1 + b_offspring)*population[parent_1, j] + (1 - b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j])           
            else:   
                offspring[i,j] = np.clip(((1 - b_offspring)*population[parent_1, j] + (1 + b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j]) 
        for k in range (1, len(list_of_functions) + 1):
            offspring[i,-k] = list_of_functions[-k](offspring[i,0:offspring.shape[1]-len(list_of_functions)])
    return offspring 

# Function: Mutation
def mutation(offspring, mutation_rate = 0.1, eta = 1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2]):
    d_mutation = 0            
    for i in range (0, offspring.shape[0]):
        for j in range(0, offspring.shape[1] - len(list_of_functions)):
            probability = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            if (probability < mutation_rate):
                rand   = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                rand_d = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)                                     
                if (rand <= 0.5):
                    d_mutation = 2*(rand_d)
                    d_mutation = d_mutation**(1/(eta + 1)) - 1
                elif (rand > 0.5):  
                    d_mutation = 2*(1 - rand_d)
                    d_mutation = 1 - d_mutation**(1/(eta + 1))                
                offspring[i,j] = np.clip((offspring[i,j] + d_mutation), min_values[j], max_values[j])                        
        for k in range (1, len(list_of_functions) + 1):
            offspring[i,-k] = list_of_functions[-k](offspring[i,0:offspring.shape[1]-len(list_of_functions)])
    return offspring 

############################################################################

# NSGA II Function
def non_dominated_sorting_genetic_algorithm_II(population_size = 5, mutation_rate = 0.1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2], generations = 50, mu = 1, eta = 1, verbose = True):        
    count      = 0   
    population = initial_population(population_size = population_size, min_values = min_values, max_values = max_values, list_of_functions = list_of_functions)  
    offspring  = initial_population(population_size = population_size, min_values = min_values, max_values = max_values, list_of_functions = list_of_functions)  
    while (count <= generations):       
        if (verbose == True):
            print('Generation = ', count)
        population        = np.vstack([population, offspring])
        rank              = fast_non_dominated_sorting(population, number_of_functions = len(list_of_functions))
        population, rank  = sort_population_by_rank(population, rank)
        population, rank  = population[0:population_size,:], rank[0:population_size,:] 
        crowding_distance = crowding_distance_function(population, len(list_of_functions))
        offspring         = breeding(population, rank, crowding_distance, mu = mu, min_values = min_values, max_values = max_values, list_of_functions = list_of_functions)
        offspring         = mutation(offspring, mutation_rate = mutation_rate, eta = eta, min_values = min_values, max_values = max_values, list_of_functions = list_of_functions)             
        count             = count + 1              
    return population

############################################################################
