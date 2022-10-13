############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: NAEMO (Neighborhood-sensitive Archived Evolutionary Many-objective Optimization)

# Citation: 
# PEREIRA, V. (2021). Project: pyMultiojective, File: naemo.py, GitHub repository: <https://github.com/Valdecy/pyMultiojective>

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
    if rp == 'none':
        idx = np.argsort(rank[:,0], axis = 0).tolist()
        population  = population[idx,:]
    else:
        idx = np.where(rank <= rp)[0].tolist()
        population  = population[idx,:]
    return population

############################################################################

# Function: Offspring
def breeding(archive, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2], k = 4):
    offspring   = np.zeros((k*len(archive), archive[0][0].shape[0]))
    parent_1    = 0
    parent_2    = 0
    b_offspring = 0  
    d_mutation  = 0  
    count       = 0
    for i in range (0, len(archive)):
        for _ in range(0, k):
            if (len(archive[i]) > 2):
                i1, i2 = random.sample(range(0, len(archive[i])), 2)
            elif (len(archive[i]) == 1):
                i1 = 0
                i2 = 0
            elif (len(archive[i]) == 2):
                i1 = 0
                i2 = 1
            rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            if (rand > 0.5 and i1 != i2):
                parent_1 = i1
                parent_2 = i2
            else:
                parent_1 = i2
                parent_2 = i1
            n   = random.randrange(offspring.shape[1] - len(list_of_functions)) + 1
            dim = random.sample(range(0, offspring.shape[1] - len(list_of_functions)), n)
            for j in range(0, offspring.shape[1] - len(list_of_functions)):
                if (i1 != i2):
                    rand   = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                    rand_b = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1) 
                    rand_c = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1) 
                    mu     = np.random.normal(0, 5, 1)[0]                      
                    if (rand <= 0.5):
                        b_offspring = 2*(rand_b)
                        b_offspring = b_offspring**(1/(mu + 1))
                    elif (rand > 0.5):  
                        b_offspring = 1/(2*(1 - rand_b))
                        b_offspring = b_offspring**(1/(mu + 1))       
                    if (rand_c >= 0.5):
                        offspring[i,j] = np.clip(((1 + b_offspring)*archive[parent_1, j] + (1 - b_offspring)*archive[parent_2, j])/2, min_values[j], max_values[j])           
                    else:   
                        offspring[i,j] = np.clip(((1 - b_offspring)*archive[parent_1, j] + (1 + b_offspring)*archive[parent_2, j])/2, min_values[j], max_values[j])          
                elif (i1 == i2 and j in dim):
                    eta    = np.random.normal(0, 0.1, 1)[0]
                    rand   = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                    rand_d = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)                                     
                    if (rand <= 0.5):
                        d_mutation = 2*(rand_d)
                        d_mutation = d_mutation**(1/(eta + 1)) - 1
                    elif (rand > 0.5):  
                        d_mutation  = 2*(1 - rand_d)
                        d_mutation  = 1 - d_mutation**(1/(eta + 1))                
                    offspring[count,j] = np.clip((offspring[count,j] + d_mutation), min_values[j], max_values[j]) 
            for m in range (1, len(list_of_functions) + 1):
                offspring[count,-m] = list_of_functions[-m](offspring[count,0:offspring.shape[1]-len(list_of_functions)])
            count = count + 1
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

# Function: Association to Reference Point
def association(population, weights, M, theta = 5, solution = 'best'):
    z_min = np.min(population[:,-M:], axis = 0)
    pbi   = np.zeros((population.shape[0], weights.shape[0]))
    for i in range(0, population.shape[0]):
        for j in range(0, weights.shape[0]):
            d1       = np.linalg.norm(np.dot((population[i,-M:].reshape(1, M) - z_min).T, weights[j,:].reshape(1, M) ))/np.linalg.norm(weights[j,:])
            d2       = np.linalg.norm(population[i,-M:] - z_min - d1*(weights[j,:]/np.linalg.norm(weights[j,:])))
            pbi[i,j] = d1 + theta*d2       
    arg   = np.argmin(pbi, axis = 1)
    d     = np.amin(pbi, axis = 1)
    niche = dict( zip( np.arange(weights.shape[0]), [[]] * weights.shape[0]) )
    nc    = dict( zip( np.arange(weights.shape[0]), [[]] * weights.shape[0]) )
    idx_u = set(arg)
    idx_  = []
    for i in idx_u:
        X = list(np.where(arg == i)[0])
        Y = list(d[X])
        Z = [x for _, x in sorted(zip(Y, X))]
        niche.update({i: Z})
    for i in range(0, weights.shape[0]):
        if (len(niche[i]) != 0 and solution == 'best'):
            individual = niche[i]
            idx_adp    = np.argmin(d[individual])
            idx_.append( individual[idx_adp] )
            nc.update({i: [individual[idx_adp]]})
        elif(len(niche[i]) != 0 and solution != 'best'):
            individual = niche[i]
            idx_.append( individual )
            nc.update({i: [individual]})    
    if (solution == 'best'):
        population = population[idx_ , :]
    else:
        archive = [[] for _ in range(0, weights.shape[0])]
        for i in range(0, weights.shape[0]):
            idx = niche[i]
            if (len(idx) != 0):
                for j in range(0, len(idx)):
                    archive[i].append(population[idx[j],:])
        population = archive 
    return population, nc

# Function: Create Arquive for Each Reference Vector
def create_arquive(weights, size, min_values, max_values, list_of_functions, theta, k):
    archive = [[] for _ in range(0, weights.shape[0])]
    for _ in range(0, k) :
        candidates     = initial_population(size*15, min_values, max_values, list_of_functions)
        candidates, nc = association(candidates, weights, len(list_of_functions), theta)
        j              = 0
        for key in nc:
            if (len(nc[key]) != 0 and len(archive[key]) <= k):
                archive[key].append(candidates[key-j,:])
            else:
                j = j + 1
    for i in range(0, len(archive)):
        if (len(archive[i]) == 0):
            candidate = initial_population(1, min_values, max_values, list_of_functions)
            archive[i].append(candidate[0,:])
    return archive

# Function: Select Only Non-Dominated Individuals
def clean_arquive(archive, M, k):
    new_archive = [[] for _ in range(0, len(archive))]
    for i in range(0, len(archive)):
        rank = fast_non_dominated_sorting(np.asarray(archive[i]), M)
        for j in range(0, rank.shape[0]):
            if (rank[j, 0] == 1 and len(new_archive[i]) <= k):
                new_archive[i].append(archive[i][j])
    return new_archive

############################################################################

# NAEMO Function neighborhood_sensitive_archived_evolutionary_many_objective_optimization
def neighborhood_sensitive_archived_evolutionary_many_objective_optimization(references = 5, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2], generations = 50, theta = 5, k = 4, verbose = True):       
    count      = 0
    references = max(5, references)
    M          = len(list_of_functions)
    weights    = reference_points(M = M, p = references)
    size       = k*weights.shape[0]
    archive    = create_arquive(weights, size, min_values, max_values, list_of_functions, theta, k)
    print('Total Number of Points on Reference Hyperplane: ', int(weights.shape[0]), ' Population Size: ', int(size))
    while (count <= generations): 
        if (verbose == True):
            print('Generation = ', count)
        archive    = clean_arquive(archive, M, k)
        offspring  = breeding(archive, min_values, max_values, list_of_functions, k)
        arch_1, _  = association(offspring, weights, M, theta, solution = '')
        archive    = [archive[i] + arch_1[i] for i in range(0, len(archive)) ]
        count      = count + 1
    archive     = clean_arquive(archive, M, k)
    archive     = [item for sublist in archive for item in sublist]
    solution    = np.array(archive)
    solution, _ = association(solution, weights, M, theta)
    return solution

############################################################################