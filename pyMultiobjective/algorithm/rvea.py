############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: RVEA (Reference Vector Guided Evolutionary Algorithm)

# Citation: 
# PEREIRA, V. (2021). Project: pyMultiojective, File: rvea.py, GitHub repository: <https://github.com/Valdecy/pyMultiojective>

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
def breeding(population, min_values = [-5,-5], max_values = [5,5], mu = 1, list_of_functions = [func_1, func_2], size = 5):
    offspring   = np.zeros((size, population.shape[1]))
    parent_1    = 0
    parent_2    = 1
    b_offspring = 0  
    for i in range (0, offspring.shape[0]):
        if (len(population) - 1 >= 3):
            i1, i2 = random.sample(range(0, len(population) - 1), 2)
        elif (len(population) - 1 == 0):
            i1 = 0
            i2 = 0
        else:
            i1 = 0
            i2 = 1
        rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        if (rand > 0.5):
            parent_1 = i1
            parent_2 = i2
        else:
            parent_1 = i2
            parent_2 = i1
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
            r_points[D] = Q / (1.0 * T)
            points.append(r_points)
        elif (D != M - 1):
            for i in range(Q + 1):
                r_points[D] = i / T
                points.extend(generator(r_points.copy(), M, Q - i, T, D + 1))
        return points
    ref_points = np.array(generator(np.zeros(M), M, p, p, 0))
    return ref_points

# Function: Nearest Vectors 
def nearest_vectors(weights):
    sorted_cosine     = -np.sort( -np.dot(weights, weights.T), axis = 1 )
    arccosine_weights = np.arccos( np.clip(sorted_cosine[:,1], 0, 1 )   )
    return arccosine_weights

# Function: Angle Penalized Distance Selection
def selection(population, offspring, M, weights, neighbours, alpha, t, t_max):
    population = np.vstack([population, offspring])
    z_min      = np.min(population[:,-M:], axis = 0)
    f          = population[:,-M:] - z_min
    cos        = np.dot(f, weights.T) / ( np.linalg.norm(f, axis = 1).reshape(-1, 1) + 1e-21 )
    arc_c      = np.arccos( np.clip(cos, 0, 1) )
    idx        = np.argmax(cos, axis = 1)
    niche      = dict( zip( np.arange(weights.shape[0]), [[]] * weights.shape[0]) )
    idx_u      = set(idx)
    for i in idx_u:
        niche.update({i: list(np.where(idx == i)[0])})
    idx_ = []
    for i in range(0, weights.shape[0]):
        if (len(niche[i]) != 0):
            individual = niche[i]
            arc_c_ind  = arc_c[individual, i]
            arc_c_ind  = arc_c_ind / neighbours[i]
            d          = np.linalg.norm(population[individual, -M:] - z_min, axis = 1) * (1 + M * ((t / t_max) ** alpha ) * arc_c_ind)
            idx_adp    = np.argmin(d)
            idx_.append( individual[idx_adp] )
    return population[idx_ , :]

# Function: Adaptation
def adaptation(population, weights, weights_, M):
    z_min      = np.min(population[:,-M:], axis = 0)
    z_max      = np.max(population[:,-M:], axis = 0)
    weights    = weights_*(z_max - z_min)
    weights    = weights / (np.linalg.norm(weights, axis = 1).reshape(-1, 1) )
    neighbours = nearest_vectors(weights)
    return weights, neighbours

############################################################################

# RVEA Function
def reference_vector_guided_evolutionary_algorithm(references = 5, mutation_rate = 0.1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2], generations = 50, mu = 1, eta = 1, k = 4, alpha = 2, fr = 0.2, verbose = True):       
    count      = 0
    references = max(5, references)
    M          = len(list_of_functions)
    weights    = reference_points(M = M, p = references)
    weights    = weights/np.linalg.norm(weights)
    weights_   = np.copy(weights)
    neighbours = nearest_vectors(weights)
    size       = k*weights.shape[0]
    population = initial_population(size, min_values, max_values, list_of_functions)   
    print('Total Number of Points on Reference Hyperplane: ', int(weights.shape[0]), ' Population Size: ', int(size))
    while (count <= generations):       
        if (verbose == True):
            print('Generation = ', count)
        offspring  = breeding(population, min_values, max_values, mu, list_of_functions, size)
        offspring  = mutation(offspring, mutation_rate, eta, min_values, max_values, list_of_functions) 
        population = selection(population, offspring, M, weights, neighbours, alpha, count,  generations)
        if ( (count/generations) // fr == 0 and count != 0):
            weights, neighbours = adaptation(population, weights, weights_, M)
        count = count + 1              
    return population[:weights.shape[0],:]

##############################################################################
