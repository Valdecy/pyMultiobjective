############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: SMS-EMOA (S-Metric Selection Evolutionary Multiobjective Optimization Algorithm)

# Citation: 
# PEREIRA, V. (2021). Project: pyMultiojective, File: sms_emoa.py, GitHub repository: <https://github.com/Valdecy/pyMultiojective>

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
def sort_population_by_rank(population, rank, rp = 'none'):
    control = 1
    if (rp == 'none'):
        idx        = np.argsort(rank[:,0], axis = 0).tolist()
        population = population[idx,:]
    else:
        idx = np.where(rank <= rp)[0].tolist()
        while len(idx) < 5:
            idx     = np.where(rank <= rp + control)[0].tolist()
            control = control + 1
        population = population[idx,:]
    return population

############################################################################

# Function: Offspring
def breeding(population, min_values = [-5,-5], max_values = [5,5], mu = 1, list_of_functions = [func_1, func_2]):
    offspring   = np.copy(population)
    parent_1    = 0
    parent_2    = 1
    b_offspring = 0
    for i in range (0, offspring.shape[0]):
        i1, i2, i3, i4 = random.sample(range(0, len(population) - 1), 4)
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

# Function: Normalize Objective Functions
def normalization(population, number_of_functions):
    M                 = number_of_functions
    z_min             = np.min(population[:,-M:], axis = 0)
    population[:,-M:] = population[:,-M:] - z_min
    w                 = np.zeros((M, M)) + 0.0000001
    np.fill_diagonal(w, 1)
    z_max             = []
    for i in range(0, M):
       z_max.append(np.argmin(np.max(population[:,-M:]/w[i], axis = 1)))
    if ( len(z_max) != len(set(z_max)) or M == 1):
        a     = np.max(population[:,-M:], axis = 0)
    else:
        k     = np.ones((M, 1))
        z_max = np.vstack((population[z_max,-M:]))
        a     = np.matrix.dot(np.linalg.inv(z_max), k)
        a     = (1/a).reshape(1, M)
    population[:,-M:] = population[:,-M:] /(a - z_min)
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
def association(srp, population, z_max, number_of_functions):
    M    = number_of_functions
    p    = copy.deepcopy(population)
    p    = normalization(p, M)
    p1   = np.zeros((1, M))
    p2   = srp
    p3   = p[:,-M:]
    g    = point_to_line(p1, p2, p3) # Matrix (Population, Reference)
    idx  = []
    arg  = np.argmin(g, axis = 1)
    hv_c = pg.hypervolume(p[:,-M:])
    hv   = hv_c.contributions(z_max)
    d    = 1/(hv + 0.0000000000000001)
    for ind in np.unique(arg).tolist():
        f = [i[0] for i in np.argwhere(arg == ind).tolist()]
        idx.append(f[d[f].argsort()[0]])
    if (len(idx) < 5):   
        idx.extend([x for x in list(range(0, population.shape[0])) if x not in idx])
        idx = idx[:5]
    return idx

############################################################################

# SMS-EMOA Function
def s_metric_selection_evolutionary_multiobjective_optimization_algorithm(references = 5, mutation_rate = 0.1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2], generations = 5, mu = 1, eta = 1, k = 4, verbose = True):       
    count      = 0
    references = max(5, references)
    M          = len(list_of_functions)
    srp        = reference_points(M = M, p = references)
    size       = k*srp.shape[0]
    population = initial_population(size, min_values, max_values, list_of_functions)  
    offspring  = initial_population(size, min_values, max_values, list_of_functions)  
    z_max      = np.max(population[:,-M:], axis = 0)
    print('Total Number of Points on Reference Hyperplane: ', int(srp.shape[0]), ' Population Size: ', int(size))
    while (count <= generations):       
        if (verbose == True):
            print('Generation = ', count)
        population = np.vstack([population, offspring])
        rank       = fast_non_dominated_sorting(population,  M)
        population = sort_population_by_rank(population, rank) 
        z_max      = np.vstack([z_max, np.max(population[:,-M:], axis = 0)])
        z_max      = np.max(z_max, axis = 0)
        idx        = association(srp, population, z_max, M)
        population = population[idx, :]
        population = population[0:size,:]
        offspring  = breeding(population, min_values, max_values, mu, list_of_functions)
        offspring  = mutation(offspring, mutation_rate, eta, min_values, max_values, list_of_functions)             
        count      = count + 1              
    return population[:srp.shape[0], :]

############################################################################