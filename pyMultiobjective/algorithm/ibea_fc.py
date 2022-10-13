############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: IBEA-FC (Indicator-Based Evolutionary Algorithm with Fast Comparison Indicator)

# Citation: 
# PEREIRA, V. (2022). Project: pyMultiojective, File: ibea_fc.py, GitHub repository: <https://github.com/Valdecy/pyMultiojective>

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

# Function:  Pareto Front  
def pareto_front_points(pts, pf_min = True):
    def pareto_front(pts, pf_min):
        pf = np.zeros(pts.shape[0], dtype = np.bool_)
        for i in range(0, pts.shape[0]):
            cost = pts[i, :]
            if (pf_min == True):
                g_cost = np.logical_not(np.any(pts > cost, axis = 1))
                b_cost = np.any(pts < cost, axis = 1)
            else:
                g_cost = np.logical_not(np.any(pts < cost, axis = 1))
                b_cost = np.any(pts > cost, axis = 1)
            dominated = np.logical_and(g_cost, b_cost)
            if  (np.any(pf) == True):
                if (np.any(np.all(pts[pf] == cost, axis = 1)) == True):
                    continue
            if not (np.any(dominated[:i]) == True or np.any(dominated[i + 1 :]) == True):
                pf[i] = True
        return pf
    idx     = np.argsort(((pts - pts.mean(axis = 0))/(pts.std(axis = 0) + 1e-7)).sum(axis = 1))
    pts     = pts[idx]
    pf      = pareto_front(pts, pf_min)
    pf[idx] = pf.copy()
    return pf

############################################################################

# Function: Fitness Selection   
def selection(population, population_size, M, z_min, z_max):
    f          = (population[:, -M:] - z_min)/(z_max - z_min)
    eps        = np.sum(f, axis = 1)
    idx        = np.where(eps <= 1)[0].tolist()
    population = population[idx,:]
    population = population[:population_size,:]
    return population

############################################################################

# IBEA-FC Function
def indicator_based_evolutionary_algorithm_fc(population_size = 5, mutation_rate = 0.1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2], generations = 50, mu = 1, eta = 1, verbose = True):       
    count      = 0
    M          = len(list_of_functions)
    population = initial_population(population_size, min_values, max_values, list_of_functions) 
    z_min      = np.min(population[:,-M:], axis = 0)
    z_max      = np.max(population[:,-M:], axis = 0)
    while (count <= generations - 1):
        if (verbose == True):
            print('Generation = ', count)
        offspring  = breeding(population, min_values, max_values, mu, list_of_functions, population_size)
        offspring  = mutation(offspring, mutation_rate, eta, min_values, max_values, list_of_functions)  
        population = np.vstack([population, offspring])
        z_min      = np.vstack([z_min, np.min(population[:,-M:], axis = 0)])
        z_min      = np.min(z_min, axis = 0)
        z_max      = np.vstack([z_max, np.max(population[:,-M:], axis = 0)])
        z_max      = np.max(z_max, axis = 0)
        population = selection(population, population_size, M, z_min, z_max)
        idx        = pareto_front_points(population [:, -M:], pf_min = True)
        population = population [idx,:]
        count      = count + 1  
    return population

############################################################################
