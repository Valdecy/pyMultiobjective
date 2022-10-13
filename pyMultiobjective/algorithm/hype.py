############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: HypE (Hypervolume Estimation Multiobjective Optimization Algorithm)

# Citation: 
# PEREIRA, V. (2022). Project: pyMultiojective, File: hype.py, GitHub repository: <https://github.com/Valdecy/pyMultiojective>

############################################################################

# Required Libraries
import copy
import numpy as np
import pygmo as pg
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
            r_points[D] = Q / T
            points.append(r_points)
        elif (D != M - 1):
            for i in range(Q + 1):
                r_points[D] = i / T
                points.extend(generator(r_points.copy(), M, Q - i, T, D + 1))
        return points
    ref_points = np.array(generator(np.zeros(M), M, p, p, 0))
    return ref_points

# Function: Normalize Objective Functions
def normalization(population, z_min, z_max, number_of_functions):
    M                 = number_of_functions
    population[:,-M:] = np.clip(population[:,-M:] /(z_max - z_min + 0.00000000000000001), 0, 1)
    return population

# Function: Distance from Point (p3) to a Line (p1, p2).    
def point_to_line(p1, p2, p3):
    p2 = p2 - p1
    dp = np.dot(p3, p2.T)
    pp = dp/np.linalg.norm(p2.T, axis = 0)
    pn = np.linalg.norm(p3, axis = 1)
    pn = np.array([pn,]*pp.shape[1]).transpose()
    dl = np.sqrt(pn**2 - pp**2)
    return dl

# Function: Association
def association(srp, population, z_min, z_max, M):
    p    = copy.deepcopy(population)
    p    = normalization(p, z_min, z_max, M)
    p1   = np.zeros((1, M))
    p2   = srp
    p3   = p[:,-M:]
    g    = point_to_line(p1, p2, p3) # Matrix (Population, Reference)
    idx  = []
    arg  = np.argmin(g, axis = 1)
    hv_c = pg.hypervolume(p[:,-M:])
    hv   = hv_c.contributions([1.05]*M)
    d    = 1/(hv + 0.00000000000000001)
    for ind in np.unique(arg).tolist():
        f = [i[0] for i in np.argwhere(arg == ind).tolist()]
        idx.append(f[d[f].argsort()[0]])
    if (len(idx) < 5):   
        idx.extend([x for x in list(range(0, population.shape[0])) if x not in idx])
        idx = idx[:5]
    return idx

############################################################################

# HypE Function
def hypervolume_estimation_mooa(references = 5, mutation_rate = 0.1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2], generations = 5, mu = 1, eta = 1, k = 4, verbose = True):       
    count      = 0
    references = max(5, references)
    M          = len(list_of_functions)
    srp        = reference_points(M = M, p = references)
    size       = k*srp.shape[0]
    population = initial_population(size, min_values, max_values, list_of_functions)  
    offspring  = initial_population(size, min_values, max_values, list_of_functions)  
    z_min      = np.min(population[:,-M:], axis = 0)
    z_max      = np.max(population[:,-M:], axis = 0)
    print('Total Number of Points on Reference Hyperplane: ', int(srp.shape[0]), ' Population Size: ', int(size))
    while (count <= generations):       
        if (verbose == True):
            print('Generation = ', count)
        population = np.vstack([population, offspring])
        z_min      = np.vstack([z_min, np.min(population[:,-M:], axis = 0)])
        z_min      = np.min(z_min, axis = 0)
        z_max      = np.vstack([z_max, np.max(population[:,-M:], axis = 0)])
        z_max      = np.max(z_max, axis = 0)
        idx        = association(srp, population, z_min, z_max, M)
        population = population[idx, :]
        population = population[:size,:]
        offspring  = breeding(population, min_values, max_values, mu, list_of_functions, size)
        offspring  = mutation(offspring, mutation_rate, eta, min_values, max_values, list_of_functions)             
        count      = count + 1              
    return population[:srp.shape[0], :]

############################################################################
