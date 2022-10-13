############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: CTAEA (Constrained Two Archive Evolutionary Algorithm)

# Citation: 
# PEREIRA, V. (2021). Project: pyMultiojective, File: ctaea.py, GitHub repository: <https://github.com/Valdecy/pyMultiojective>

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
def sort_population_by_rank(ca, rank, rp = 'none'):
    if rp == 'none':
        idx = np.argsort(rank[:,0], axis = 0).tolist()
        ca  = ca[idx,:]
    else:
        idx = np.where(rank <= rp)[0].tolist()
        ca  = ca[idx,:]
    return ca

############################################################################

# Function: Offspring
def breeding(ca, da, min_values = [-5,-5], max_values = [5,5], mu = 1, list_of_functions = [func_1, func_2], size = 5):
    offspring   = np.zeros((size, ca.shape[1]))
    cada        = np.vstack([ca, da])
    parent_1    = 0
    parent_2    = 1
    b_offspring = 0  
    rank_ca     = fast_non_dominated_sorting(ca, number_of_functions = len(list_of_functions))
    rank_ca     = rank_ca[rank_ca == 1]
    p_ca        = rank_ca.shape[0]/(ca.shape[0] + da.shape[0])
    rank_da     = fast_non_dominated_sorting(da, number_of_functions = len(list_of_functions))
    rank_da     = rank_da[rank_da == 1]
    p_da        = rank_da.shape[0]/ (ca.shape[0] + da.shape[0])
    for i in range (0, offspring.shape[0]):
        if (p_ca > p_da):
            parent_1 = random.sample(range(0, len(ca) - 1), 1)[0]
        else:
            parent_1 = random.sample(range(0, len(da) - 1), 1)[0] + ca.shape[0]
        rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        if (rand < p_ca):
            parent_2 = random.sample(range(0, len(ca) - 1), 1)[0]
        else:
            parent_2 = random.sample(range(0, len(da) - 1), 1)[0] + ca.shape[0]
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
                offspring[i,j] = np.clip(((1 + b_offspring)*cada[parent_1, j] + (1 - b_offspring)*cada[parent_2, j])/2, min_values[j], max_values[j])           
            else:   
                offspring[i,j] = np.clip(((1 - b_offspring)*cada[parent_1, j] + (1 + b_offspring)*cada[parent_2, j])/2, min_values[j], max_values[j])            
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

# Function: Normalize Objective Functions
def normalization(ca, number_of_functions):
    M         = number_of_functions
    z_min     = np.min(ca[:,-M:], axis = 0)
    ca[:,-M:] = ca[:,-M:] - z_min
    w         = np.zeros((M, M)) + 0.0000001
    np.fill_diagonal(w, 1)
    z_max     = []
    for i in range(0, M):
        z_max.append(np.argmin(np.max(ca[:,-M:]/w[i], axis = 1)))
    if ( len(z_max) != len(set(z_max)) or M == 1):
        a     = np.max(ca[:,-M:], axis = 0)
    else:
        k     = np.ones((M, 1))
        z_max = np.vstack((ca[z_max,-M:]))
        a     = np.matrix.dot(np.linalg.inv(z_max), k)
        a     = (1/a).reshape(1, M)
    ca[:,-M:] = ca[:,-M:] /(a - z_min)
    return ca

# Function: Distance from Point (p3) to a Line (p1, p2)   
def point_to_line(p1, p2, p3):
    p2 = p2 - p1
    dp = np.dot(p3, p2.T)
    pp = dp/np.linalg.norm(p2.T, axis = 0)
    pn = np.linalg.norm(p3, axis = 1)
    pn = np.array([pn,]*pp.shape[1]).transpose()
    dl = np.sqrt(pn**2 - pp**2)
    return dl

# Function: Association to Reference Point
def association(ca, weights, M):
    p     = copy.deepcopy(ca)
    p     = normalization(p, M)
    p1    = np.zeros((1, M))
    p2    = weights
    p3    = p[:,-M:]
    d     = point_to_line(p1, p2, p3) # Matrix (Population, Reference)
    idx   = np.argmin(d, axis = 1)
    niche = dict( zip( np.arange(weights.shape[0]), [[]] * weights.shape[0]) )
    n_ca  = dict( zip( np.arange(weights.shape[0]), [[]] * weights.shape[0]) )
    idx_u = set(idx)
    for i in idx_u:
        niche.update({i: list(np.where(idx == i)[0])})
    idx_ = []
    for i in range(0, weights.shape[0]):
        if (len(niche[i]) != 0):
            individual = niche[i]
            idx_adp    = np.argmin(np.amin(d[individual,:], axis = 1))
            idx_.append( individual[idx_adp] )
            n_ca.update({i: [individual[idx_adp]]})
    return ca[idx_ , :], n_ca

# Function: Update CA
def update_ca(ca, offspring, weights, M):
    ca   = np.vstack([ca, offspring])
    rank = fast_non_dominated_sorting(ca, M)
    ca   = sort_population_by_rank(ca, rank, 1)
    ca,_ = association(ca, weights, M)
    return ca[:weights.shape[0],:]

# Function: Update DA
def update_da(ca, da, offspring, weights, M):
    da      = np.vstack([da, offspring])
    _, n_da = association(da, weights, M)
    _, n_ca = association(ca, weights, M)
    s       = np.zeros((weights.shape[0], ca.shape[1]))
    idx_del = []
    s.fill(float('+inf'))
    for i in range(0, weights.shape[0]):
        if (len(n_ca[i]) != 0 and len(n_da[i]) != 0):
           if ( dominance_function(ca[n_ca[i][0], :], da[n_da[i][0],:], M)):
               s[i,:] = ca[n_ca[i][0], :]
           else:
               s[i,:] = da[n_da[i][0],:]
        elif (len(n_ca[i]) == 0 and len(n_da[i]) != 0):
            s[i,:] = da[n_da[i][0],:]
        elif (len(n_ca[i]) != 0 and len(n_da[i]) == 0):
            s[i,:] = ca[n_ca[i][0], :]
        elif (len(n_ca[i]) == 0 and len(n_da[i]) == 0):
            idx_del.append(i)
    da = np.delete(s, idx_del, axis = 0)
    return da

############################################################################

# CTAEA Function
def constrained_two_archive_evolutionary_algorithm(references = 5, mutation_rate = 0.1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2], generations = 50, mu = 1, eta = 1, k = 4, verbose = True):       
    count      = 0
    references = max(5, references)
    M          = len(list_of_functions)
    weights    = reference_points(M = M, p = references)
    size       = k*weights.shape[0]
    ca         = initial_population(size, min_values, max_values, list_of_functions) 
    da         = initial_population(size, min_values, max_values, list_of_functions)
    print('Total Number of Points on Reference Hyperplane: ', int(weights.shape[0]), ' Population Size: ', int(size))
    while (count <= generations): 
        if (verbose == True):
            print('Generation = ', count)
        offspring  = breeding(ca, da, min_values, max_values, mu, list_of_functions, size)
        offspring  = mutation(offspring, mutation_rate, eta, min_values, max_values, list_of_functions) 
        ca         = np.vstack([ca, offspring])
        ca, _      = association(ca, weights, M)
        ca         = update_ca(ca, offspring, weights, M)
        da         = update_da(ca, da, offspring, weights, M)
        count      = count + 1
    return da

############################################################################
