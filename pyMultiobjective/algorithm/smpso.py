############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: SMPSO

# Citation: 
# PEREIRA, V. (2021). Project: pyMultiojective, File: smpso.py, GitHub repository: <https://github.com/Valdecy/pyMultiojective>

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
def initial_position(swarm_size = 5, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2]):
    position = np.zeros((swarm_size, len(min_values) + len(list_of_functions)))
    for i in range(0, swarm_size):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j])      
        for k in range (1, len(list_of_functions) + 1):
            position[i,-k] = list_of_functions[-k](list(position[i,0:position.shape[1]-len(list_of_functions)]))
    return position 

# Function: Updtade Position
def update_position(position, velocity, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2]):
    for i in range(0, position.shape[0]):
        for j in range(0, len(min_values)):
            position[i,j] = np.clip((position[i,j] + velocity[i,j]),  min_values[j],  max_values[j])
        for k in range (1, len(list_of_functions) + 1):
            position[i,-k] = list_of_functions[-k](list(position[i,0:position.shape[1]-len(list_of_functions)]))
    return position

# Function: Initialize Velocity
def initial_velocity(position, min_values = [-5,-5], max_values = [5,5]):
    velocity = np.zeros((position.shape[0], len(min_values)))
    for i in range(0, velocity.shape[0]):
        for j in range(0, velocity.shape[1]):
            velocity[i,j] = random.uniform(min_values[j], max_values[j])
    return velocity

# Function: Velocity
def velocity_vector(position, velocity_, archive, M, min_values = [-5,-5], max_values = [5,5]):
    r1  = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    r2  = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    w   = np.random.uniform(low = 0.1, high = 0.5, size = 1)[0]
    c1  = np.random.uniform(low = 1.5, high = 2.5, size = 1)
    c2  = np.random.uniform(low = 1.5, high = 2.5, size = 1)[0]
    phi = 0
    if (c1 + c2 > 4):
        phi = c1 + c2
    else:
        phi = 0
    chi      = 2 / (2 - phi - ( (phi**2) - 4*phi )**(1/2))
    velocity = np.zeros((position.shape[0], velocity_.shape[1]))
    crowding = crowding_distance_function(archive, M)
    delta    = [(max_values[i] - min_values[i])/2 for i in range(0, len(min_values))]
    if (archive.shape[0] > 2):
        ind_1, ind_2 = random.sample(range(0, len(archive) - 1), 2)
        if (crowding[ind_1,0] < crowding[ind_2,0]):
            ind_1, ind_2 = ind_2, ind_1
    else:
        ind_1 = 0
        ind_2 = 0
    for i in range(0, velocity.shape[0]):
        for j in range(0, velocity.shape[1]):
            velocity[i,j] = (w*velocity_[i,j] + c1*r1*(archive[ind_1, j] - position[i,j]) + c2*r2*(archive[ind_2, j] - position[i,j]))*chi
            velocity[i,j] = np.clip(velocity[i,j], -delta[j], delta[j]) 
    return velocity

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
def fast_non_dominated_sorting(position, number_of_functions = 2):
    S     = [[] for i in range(0, position.shape[0])]
    front = [[]]
    n     = [0 for i in range(0, position.shape[0])]
    rank  = [0 for i in range(0, position.shape[0])]
    for p in range(0, position.shape[0]):
        S[p] = []
        n[p] = 0
        for q in range(0, position.shape[0]):
            if (dominance_function(solution_1 = position[p,:], solution_2 = position[q,:], number_of_functions = number_of_functions)):
                if (q not in S[p]):
                    S[p].append(q)
            elif (dominance_function(solution_1 = position[q,:], solution_2 = position[p,:], number_of_functions = number_of_functions)):
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
    rank = np.zeros((position.shape[0], 1))
    for i in range(0, len(front)):
        for j in range(0, len(front[i])):
            rank[front[i][j], 0] = i + 1
    return rank

# Function: Crowding Distance (Adapted from PYMOO)
def crowding_distance_function(pop, M):
    infinity = 1e+11
    position = copy.deepcopy(pop[:,-M:])
    position =  position.reshape((pop.shape[0], M))
    if (position.shape[0] <= 2):
        return np.full( position.shape[0], infinity)
    else:
        arg_1    = np.argsort( position, axis = 0, kind = 'mergesort')
        position = position[arg_1, np.arange(M)]
        dist     = np.concatenate([ position, np.full((1, M), np.inf)]) - np.concatenate([np.full((1, M), -np.inf), position])
        idx      = np.where(dist == 0)
        a        = np.copy(dist)
        b        = np.copy(dist)
        for i, j in zip(*idx):
            a[i, j] = a[i - 1, j]
        for i, j in reversed(list(zip(*idx))):
            b[i, j] = b[i + 1, j]
        norm            = np.max( position, axis = 0) - np.min(position, axis = 0)
        norm[norm == 0] = np.nan
        a, b            = a[:-1]/norm, b[1:]/norm
        a[np.isnan(a)]  = 0.0
        b[np.isnan(b)]  = 0.0
        arg_2           = np.argsort(arg_1, axis = 0)
        crowding        = np.sum(a[arg_2, np.arange(M)] + b[arg_2, np.arange(M)], axis = 1) / M
    crowding[np.isinf(crowding)] = infinity
    crowding                     = crowding.reshape((-1,1))
    return crowding

############################################################################

# Function: Selection
def selection(position, archive, M):
    archive = np.vstack([position, archive])
    rank    = fast_non_dominated_sorting(archive, M)
    arg     = np.argsort(rank , axis = 0).tolist()
    arg     = [i[0] for i in arg]
    archive = archive[arg, :]
    rank    = rank[arg, :]
    idx     = np.where(rank == 1)[0]
    if (len(idx) > 1):
        archive = archive[idx, :]
    archive = archive[:2*position.shape[0], :]
    return archive

############################################################################

# Function: Mutation
def mutation(position, mutation_rate = 0.1, eta = 1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2]):
    d_mutation = 0            
    for i in range (0, position.shape[0]):
        for j in range(0, position.shape[1] - len(list_of_functions)):
            probability = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            if (probability < mutation_rate):
                rand   = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                rand_d = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                                     
                if (rand <= 0.5):
                    d_mutation = 2*(rand_d)
                    d_mutation = d_mutation**(1/(eta + 1)) - 1
                elif (rand > 0.5):  
                    d_mutation = 2*(1 - rand_d)
                    d_mutation = 1 - d_mutation**(1/(eta + 1))                
                position[i,j] = np.clip((position[i,j] + d_mutation), min_values[j], max_values[j])                        
        for k in range (1, len(list_of_functions) + 1):
            position[i,-k] = list_of_functions[-k](position[i,0:position.shape[1]-len(list_of_functions)])
    return position

############################################################################

# SMPSO Function
def speed_constrained_multiobjective_particle_swarm_optimization(swarm_size = 5, min_values = [-5,-5], max_values = [5,5], iterations = 50, list_of_functions = [func_1, func_2], mutation_rate = 0.1, eta = 3, verbose = True):    
    count    = 0
    M        = len(list_of_functions)
    position = initial_position( swarm_size, min_values, max_values, list_of_functions)
    archive  = copy.deepcopy(position)
    velocity = initial_velocity(position, min_values, max_values)
    while (count <= iterations):
        if (verbose == True):
            print('Generation = ', count)
        position = update_position(position, velocity, min_values, max_values, list_of_functions) 
        position = mutation(position, mutation_rate, eta, min_values, max_values, list_of_functions)
        archive  = selection(position, archive, M)
        velocity = velocity_vector(position, velocity, archive, M, min_values, max_values)         
        count    = count + 1       
    return archive

############################################################################
