############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: NSGA-III

# Citation: 
# PEREIRA, V. (2021). Project: pyMultiojective, File: n_iii.py, GitHub repository: <https://github.com/Valdecy/pyMultiojective>

############################################################################

# Required Libraries
import copy
import numpy  as np
import math
import matplotlib.pyplot as plt
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

# Function: Dominance
def dominance_function(solution_1, solution_2, number_of_functions = 2):
    count     = 0
    dominance = True
    for k in range (1, number_of_functions + 1):
        if (solution_1[-k] <= solution_2[-k]):
            count = count + 1
    if (count == number_of_functions):
        dominance = True
    else:
        dominance = False       
    return dominance

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
            if (dominance_function(solution_1 = population[p,:], solution_2 = population[q,:], number_of_functions = number_of_functions)):
                if (q not in S[p]):
                    S[p].append(q)
            elif (dominance_function(solution_1 = population[q,:], solution_2 = population[p,:], number_of_functions = number_of_functions)):
                n[p] = n[p] + 1
        if (n[p] == 0):
            rank[p] = 0
            if (p not in front[0]):
                front[0].append(p)
    i = 0
    while(front[i] != []):
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
    idx            = np.argsort(rank, axis = 0)
    rank_new       = np.zeros((population.shape[0], 1))
    population_new = np.zeros((population.shape[0], population.shape[1]))  
    for i in range(0, population.shape[0]):
        rank_new[i, 0] = rank[idx[i], 0] 
        for k in range(0, population.shape[1]):
            population_new[i,k] = population[idx[i],k]
    return population_new, rank_new

# Function: Offspring
def breeding(population, rank, min_values = [-5,-5], max_values = [5,5], mu = 1, list_of_functions = [func_1, func_2]):
    offspring   = np.copy(population)
    parent_1    = 0
    parent_2    = 1
    b_offspring = 0
    for i in range (0, offspring.shape[0]):
        i1, i2, i3, i4 = random.sample(range(0, len(population) - 1), 4)
        rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
        if (rand > 0.5):
            parent_1 = i1
        else:
            parent_1 = i2
        rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
        if (rand > 0.5):
            parent_2 = i3
        else:
            parent_2 = i4
        for j in range(0, offspring.shape[1] - len(list_of_functions)):
            rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            rand_b = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                                
            if (rand <= 0.5):
                b_offspring = 2*(rand_b)
                b_offspring = b_offspring**(1/(mu + 1))
            elif (rand > 0.5):  
                b_offspring = 1/(2*(1 - rand_b))
                b_offspring = b_offspring**(1/(mu + 1))       
            offspring[i,j] = np.clip(((1 + b_offspring)*population[parent_1, j] + (1 - b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j])           
            if(i < population.shape[0] - 1):   
                offspring[i+1,j] = np.clip(((1 - b_offspring)*population[parent_1, j] + (1 + b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j]) 
        for k in range (1, len(list_of_functions) + 1):
            offspring[i,-k] = list_of_functions[-k](offspring[i,0:offspring.shape[1]-len(list_of_functions)])
    return offspring 

# Function: Mutation
def mutation(offspring, mutation_rate = 0.1, eta = 1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2]):
    d_mutation = 0            
    for i in range (0, offspring.shape[0]):
        for j in range(0, offspring.shape[1] - len(list_of_functions)):
            probability = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            if (probability < mutation_rate):
                rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                rand_d = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                                     
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

# Function: n-Simplex Projection Sort
def projection_simplex_sort(v, z = 1):
    u     = np.sort(v)[::-1]
    cssv  = np.cumsum(u) - z
    ind   = np.arange(v.shape[0]) + 1
    cond  = u - cssv / ind > 0
    rho   = ind [cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w     = np.maximum(v - theta, 0)
    return w

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
    z_max             = np.argmax(population[:,-M:], axis = 0).tolist()
    if ( len(z_max) != len(set(z_max))):
        a     = np.max(population[:,-M:], axis = 0)
    else:
        k     = np.ones((M, 1))
        z_max = np.vstack((population[z_max,-M:]))
        a     = np.matrix.dot(np.linalg.inv(z_max), k)
        a     = (1/a).reshape(1, M)
    population[:,-M:] = population[:,-M:] /(a - z_min)
    return population

# Function: Association
def association(srp, population, number_of_functions):
    M  = number_of_functions
    p  = copy.deepcopy(population)
    p  = normalization(p, M)
    p1 = np.zeros((1, M))
    g  = np.zeros((srp.shape[0], p.shape[0]), dtype = float)
    for i in range(0, srp.shape[0]):
        p2 = srp[i,:]
        for j in range(0, p.shape[0]):
            p3     = p[j,-M:]
            g[i,j] = np.abs(np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1))
    idx = []
    for _ in range(0, 4):
        for i in range(0, g.shape[0]):
            arg = np.argmin(g[i,:])
            idx.append(arg)
            g[:,idx] = float('+inf')
    return idx[:p.shape[0]]

# Function: Sort Population by Association
def sort_population_by_association(srp, population, rank, number_of_functions):
    M          = number_of_functions
    idx        = association(srp, population, M)
    idx        = idx[:srp.shape[0]]
    idx.extend([x for x in list(range(0, population.shape[0])) if x not in idx])
    population = population[idx, :]
    rank       = rank[idx, :]
    return population, rank

############################################################################