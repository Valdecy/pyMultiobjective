############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: SMPSO (Speed-Constrained Multiobjective Particle Swarm Optimization)

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

############################################################################

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
    r1  = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    r2  = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
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

# Function: Epsilon Dominance
def selection_dominance(eps_dom, position, M):
    solution = np.vstack([eps_dom, position])
    solution = np.unique(solution, axis = 0)
    eps_dom  = np.copy(solution)
    idx      = pareto_front_points(solution[:, -M:], pf_min = True)
    eps_dom  = eps_dom[idx,:]
    return eps_dom

############################################################################

# Function: Mutation
def mutation(position, mutation_rate = 0.1, eta = 1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2]):
    d_mutation = 0            
    for i in range (0, position.shape[0]):
        for j in range(0, position.shape[1] - len(list_of_functions)):
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
                position[i,j] = np.clip((position[i,j] + d_mutation), min_values[j], max_values[j])                        
        for k in range (1, len(list_of_functions) + 1):
            position[i,-k] = list_of_functions[-k](position[i,0:position.shape[1]-len(list_of_functions)])
    return position

############################################################################

# SMPSO Function
def speed_constrained_multiobjective_particle_swarm_optimization(swarm_size = 5, min_values = [-5,-5], max_values = [5,5], iterations = 50, list_of_functions = [func_1, func_2], mutation_rate = 0.1, eta = 3, verbose = True):    
    count    = 0
    M        = len(list_of_functions)
    position = initial_position(swarm_size, min_values, max_values, list_of_functions)
    eps_dom  = initial_position(swarm_size, min_values, max_values, list_of_functions)
    velocity = initial_velocity(position, min_values, max_values)
    while (count <= iterations):
        if (verbose == True):
            print('Generation = ', count)
        position = update_position(position, velocity, min_values, max_values, list_of_functions) 
        position = mutation(position, mutation_rate, eta, min_values, max_values, list_of_functions)
        eps_dom  = selection_dominance(eps_dom, position, M)
        if (eps_dom.shape[0] > swarm_size):
            crowding = crowding_distance_function(eps_dom, M)
            arg      = np.argsort(crowding , axis = 0)[::-1].tolist()
            try:
                arg = [i[0] for i in arg ]
            except:
                arg = [i for i in arg ]
            if (len(arg) > 0):
                eps_dom = eps_dom[arg, :]
            eps_dom = eps_dom[:swarm_size, :]
        velocity = velocity_vector(position, velocity, eps_dom, M, min_values, max_values)
        count    = count + 1     
    return eps_dom

############################################################################
