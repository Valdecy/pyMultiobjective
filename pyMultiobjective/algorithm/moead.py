############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: MOEA/D (Multiobjective Evolutionary Algorithm Based on Decomposition)

# Citation: 
# PEREIRA, V. (2021). Project: pyMultiojective, File: moead.py, GitHub repository: <https://github.com/Valdecy/pyMultiojective>

############################################################################

# Required Libraries
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

# Function: Offspring
def breeding(population, neighbours, min_values = [-5,-5], max_values = [5,5], mu = 1, list_of_functions = [func_1, func_2]):
    offspring   = np.copy(population)
    parent_1    = 0
    parent_2    = 1
    b_offspring = 0  
    for i in range (0, offspring.shape[0]):
        canditates = list(range(0, population.shape[0]))
        canditates.remove(i)
        canditates = canditates[:neighbours]
        i1, i2, i3, i4 = random.sample(canditates, 4)
        rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        if (rand > 0.5):
            parent_1 = i1
        else:
            parent_1 = i2
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

# Function: Reference Points
def reference_points(M, p):
    def generator(r_points, M, Q, T, D):
        points = []
        if (D == M - 1):
            r_points[D] = Q / T
            points.append(r_points)
        elif (D != M - 1):
            for i in range(Q + 1):
                r_points[D] = i / T
                points.extend(generator(r_points.copy(), M, Q - i, T, D + 1))
        return points
    ref_points = np.array(generator(np.zeros(M), M, p, p, 0))
    return ref_points

# Function: Selection
def selection(population, offspring, M, weights, theta):
    z_min      = np.min(np.vstack([population[:,-M:], offspring[:,-M:]]), axis = 0)
    population = np.vstack([population, offspring])
    pbi        = np.zeros((population.shape[0], weights.shape[0]))
    for i in range(0, population.shape[0]):
        for j in range(0, weights.shape[0]):
            d1       = np.linalg.norm(np.dot((population[i,-M:].reshape(1, M) - z_min).T, weights[j,:].reshape(1, M) ))/np.linalg.norm(weights[j,:])
            d2       = np.linalg.norm(population[i,-M:] - z_min - d1*(weights[j,:]/np.linalg.norm(weights[j,:])))
            pbi[i,j] = d1 + theta*d2
    idx = []
    arg = np.argmin(pbi, axis = 1)
    d   = np.amin(pbi, axis = 1)
    for ind in np.unique(arg).tolist():
        f = [i[0] for i in np.argwhere(arg == ind).tolist()]
        idx.append(f[d[f].argsort()[0]])
    idx.extend([x for x in list(range(0, population.shape[0])) if x not in idx])
    population = population[idx, :]
    population = population[:weights.shape[0], :]
    return population

############################################################################

# MOEA/D Function
def multiobjective_evolutionary_algorithm_based_on_decomposition(references = 5, mutation_rate = 0.1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2], generations = 50, mu = 1, eta = 1, theta = 4, k = 4, verbose = True):       
    count      = 0
    references = max(5, references)
    M          = len(list_of_functions)
    weights    = reference_points(M = M, p = references)
    size       = k*weights.shape[0]
    neighbours = max(4, int(size//10))
    population = initial_population(size, min_values, max_values, list_of_functions)   
    print('Total Number of Points on Reference Hyperplane: ', int(weights.shape[0]), ' Population Size: ', int(size))
    while (count <= generations):       
        if (verbose == True):
            print('Generation = ', count)
        offspring  = breeding(population, neighbours, min_values, max_values, mu, list_of_functions)
        offspring  = mutation(offspring, mutation_rate, eta, min_values, max_values, list_of_functions)  
        population = selection(population, offspring, M, weights, theta)
        count      = count + 1              
    return population

############################################################################
