############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: GrEA (Grid-based Evolutionary Algorithm)

# Citation: 
# PEREIRA, V. (2021). Project: pyMultiojective, File: grea.py, GitHub repository: <https://github.com/Valdecy/pyMultiojective>

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

# Function: Dominance Solution 1 over Solution 2
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

# Function: Offspring
def breeding(population, population_size, k, grid, gcd, min_values = [-5,-5], max_values = [5,5], mu = 1, list_of_functions = [func_1, func_2]):
    offspring   = np.zeros((population_size*k, population.shape[1]))
    parent_1    = 0
    parent_2    = 1
    b_offspring = 0  
    for i in range (0, offspring.shape[0]):
        if (len(population)>4):
            i1, i2, i3, i4 = random.sample(range(0, len(population) - 1), 4)
            if (dominance_function(population[i1, :], population[i2, :], len(list_of_functions)) or grid_dominance_function(grid[i1, :], grid[i2, :])):
                parent_1 = i1
            elif (dominance_function(population[i2, :], population[i1, :], len(list_of_functions)) or grid_dominance_function(grid[i2, :], grid[i1, :])):
                parent_1 = i2
            elif (gcd[i1] < gcd[i2]):
                parent_1 = i1
            elif (gcd[i1] > gcd[i2]):
                parent_1 = i2
            elif (int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1) > 0.5):
                parent_1 = i1
            else:
                parent_1 = i2
            if (dominance_function(population[i3, :], population[i4, :], len(list_of_functions)) or grid_dominance_function(grid[i3, :], grid[i4, :])):
                parent_2 = i3
            elif (dominance_function(population[i4, :], population[i3, :], len(list_of_functions)) or grid_dominance_function(grid[i4, :], grid[i3, :])):
                parent_2 = i4
            elif (gcd[i3] < gcd[i4]):
                parent_2 = i3
            elif (gcd[i3] > gcd[i4]):
                parent_2 = i4
            elif (int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1) > 0.5):
                parent_2 = i3
            else:
                parent_2 = i4
        else:
            parent_1, parent_2 = random.sample(range(0, len(population) - 1), 2)
        rand   = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        if (rand > 0.5):
            parent_1, parent_2 = parent_2, parent_1
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
        for m in range (1, len(list_of_functions) + 1):
            offspring[i,-m] = list_of_functions[-m](offspring[i,0:offspring.shape[1]-len(list_of_functions)])
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

# Function: Grid 
def grid_location(population, divisions = 9, list_of_functions = [func_1, func_2]):
    M         = len(list_of_functions)
    grid      = np.zeros((population.shape[0], M))
    min_f     = np.amin(population[:,population.shape[1]-M:], axis = 0)
    max_f     = np.amax(population[:,population.shape[1]-M:], axis = 0)
    lower     = [min_f[i]  - ( max_f[i]  - min_f[i] )/(2*divisions) for i in range(0, M)]
    upper     = [max_f[i]  - ( max_f[i]  - min_f[i] )/(2*divisions) for i in range(0, M)]
    hyper_box = [(upper[i] - lower[i])/divisions for i in range(0, len(lower))]
    for i in range(0, grid.shape[0]):
        for j in range(0, grid.shape[1]):
            grid[i,j] = np.floor((population[i, population.shape[1]-M+j] - lower[j])/hyper_box[j])
    return grid

# Function: GD
def grid_diff(grid):
    gd = np.zeros((grid.shape[0], grid.shape[0]))
    for i in range(0, grid.shape[0]):
        for j in range(i, grid.shape[0]):
            if (i != j):
                gd[i, j] = np.sum(np.abs(grid[i,:] - grid[j, :]))
                gd[j, i] = gd[i, j]
    return gd

# Function: GR
def grid_rank(grid):
    gr = np.zeros((grid.shape[0]))
    for i in range(0, gr.shape[0]):
        gr[i] = np.sum(grid[i,:])
    return gr

# Function: GR Adjustment
def grid_rank_adjustment(offspring, ix, grid, gr, gd, M):
    e_g_ng_n_pd = np.zeros((offspring.shape[0], 5))
    for i in range(0, offspring.shape[1]):
        if (gd[i, ix] == 0):
            e_g_ng_n_pd[i,0] = 1
        if (grid_dominance_function(grid[i,:], grid[ix,:])):
            e_g_ng_n_pd[i,1] = 1
        else:
            e_g_ng_n_pd[i,2] = 0  
        if (gd[i, ix] < M):
            e_g_ng_n_pd[i,3] = 1
    for i in range(0, offspring.shape[1]):
        if (e_g_ng_n_pd[i,0] == 1):
            gr[i] = gr[i] + M + 2
        if (e_g_ng_n_pd[i,1] == 1):
            gr[i] = gr[i] + M
        if (e_g_ng_n_pd[i,0] == 0 and e_g_ng_n_pd[i,2] == 1):
            e_g_ng_n_pd[i,-1] == 0
    for i in range(0, offspring.shape[1]):
        if (e_g_ng_n_pd[i,2] == 1 and e_g_ng_n_pd[i,2] == 1 and e_g_ng_n_pd[i,0] == 0):
            if (e_g_ng_n_pd[i,-1] <= M - gd[i, ix]):
                e_g_ng_n_pd[i,-1] == M - gd[i, ix]
                for j in range(0, offspring.shape[1]):
                    if (e_g_ng_n_pd[j,-1] < e_g_ng_n_pd[i,-1]):
                        e_g_ng_n_pd[j,-1] = e_g_ng_n_pd[i,-1]
    for i in range(0, offspring.shape[1]):
        if (e_g_ng_n_pd[i,3] == 1 and  e_g_ng_n_pd[i,0] == 0):
            gr[i] = gr[i] + e_g_ng_n_pd[i,-1]
    return gr

# Function: GCD
def grid_crowding_distance(gd, list_of_functions):
    M        = len(list_of_functions)
    gcd = np.zeros((gd.shape[0]))
    for i in range(0, gcd.shape[0]):
        for j in range(0, gd.shape[1]):
            if (gd[i, j] < M and i != j):
                gcd[i] = gcd[i] + (M - gcd[i])
    return gcd

# Function: GCPD
def grid_coordinate_point_distance(population, grid, divisions = 9, list_of_functions = [func_1, func_2]):
    M         = len(list_of_functions)
    gcpd      = np.zeros((population.shape[0]))
    min_f     = np.amin(population[:,population.shape[1]-M:], axis = 0)
    max_f     = np.amax(population[:,population.shape[1]-M:], axis = 0)
    lower     = [min_f[i]  - ( max_f[i]  - min_f[i] )/(2*divisions) for i in range(0, M)]
    upper     = [max_f[i]  - ( max_f[i]  - min_f[i] )/(2*divisions) for i in range(0, M)]
    hyper_box = [(upper[i] - lower[i])/divisions for i in range(0, len(lower))]
    for i in range(0, gcpd.shape[0]):
        value = 0
        for j in range(0, M):
            value = value + ((population[i,population.shape[1]-M+j] - (lower[j] - grid[i,j]*hyper_box[j]))/hyper_box[j])**2
        gcpd[i] = value**(1/2)
    return gcpd

# Function: Grid Dominance Solution 1 over Solution 2
def grid_dominance_function(solution_1, solution_2):
    count     = 0
    dominance = True
    for k in range (0, solution_1.shape[0]):
        if (solution_1[k] <= solution_2[k]):
            count = count + 1
    if (count == solution_1.shape[0]):
        dominance = True
    else:
        dominance = False       
    return dominance

############################################################################

# Function: Find Best Solution Q
def find_best(offspring, q, ix, ix_list, grid, gr, gd, gcd, gcpd, M):
    for i in range(0, offspring.shape[0]):
        if (i not in ix_list):
            if (gr[i] < gr[ix]):
                q = np.copy(offspring[i, :])
            elif (gr[i] == gr[ix]):
                if (gcd[i] < gcd[ix]):
                    q = np.copy(offspring[i, :])
                elif (gcd[i] == gcd[ix]):
                    if (gcpd[i] < gcpd[ix]):
                        q = np.copy(offspring[i, :])
    for i in range(0, offspring.shape[0]):
        if (gd[i, ix] < M):
            gcd[i] = gcd[i] + (M - gd[i, ix])
    gr = grid_rank_adjustment(offspring, ix, grid, gr, gd, M)
    return q, gr, gcd

# Function: Select Archive
def selection(offspring_, grid, gr, gd, gcd, gcpd, population_size, M):
    offspring = np.copy(offspring_)
    archive   = np.zeros((population_size, offspring.shape[1]))
    rank      = fast_non_dominated_sorting(offspring, M)
    q         = sort_population_by_rank(offspring, rank, rp = 1)[0, :]
    ix_list   = []
    ix        = np.where(offspring == q )[0][0]
    ix_list.append(ix)
    archive[-1,:] = np.copy(q)
    for i in range(archive.shape[0]-2, -1, -1):
        q, gr, gcd = find_best(offspring, q, ix, ix_list, grid, gr, gd, gcd, gcpd, M)
        ix         = np.where(offspring == q)[0][0]
        if (ix not in ix_list):
            ix_list.append(ix)
            archive[i,:] = np.copy(q) 
        else:
            archive = np.delete(archive, i, 0)
    return archive

############################################################################

# GrEA Function
def grid_based_evolutionary_algorithm(population_size = 5, divisions = 10, mutation_rate = 0.1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2], generations = 50, mu = 1, eta = 1, k = 4, verbose = True):       
    count      = 0
    M          = len(list_of_functions)
    population = initial_population(population_size*k, min_values, max_values, list_of_functions) 
    while (count <= generations - 1): 
        if (verbose == True):
            print('Generation = ', count)
        grid_p     = grid_location(population, divisions, list_of_functions)
        gr_p       = grid_rank(grid_p)
        gd_p       = grid_diff(grid_p)
        gcd_p      = grid_crowding_distance(gd_p, list_of_functions)
        gcpd_p     = grid_coordinate_point_distance(population, grid_p, divisions, list_of_functions)
        archive_p  = selection(population, grid_p, gr_p, gd_p, gcd_p, gcpd_p, population_size, M)
        offspring  = breeding(population, population_size, k, grid_p, gcd_p, min_values, max_values, mu, list_of_functions)
        offspring  = mutation(offspring, mutation_rate, eta, min_values, max_values, list_of_functions)  
        grid_o     = grid_location(offspring, divisions, list_of_functions)
        gr_o       = grid_rank(grid_o)
        gd_o       = grid_diff(grid_o)
        gcd_o      = grid_crowding_distance(gd_o, list_of_functions)
        gcpd_o     = grid_coordinate_point_distance(offspring, grid_o, divisions, list_of_functions)
        archive_o  = selection(offspring, grid_o, gr_o, gd_o, gcd_o, gcpd_o, population_size, M)
        archive    = np.vstack([archive_p, archive_o])
        population = np.vstack([archive, offspring])[0:population_size*k,:]
        rank       = fast_non_dominated_sorting(population, M)
        population = sort_population_by_rank(population, rank, rp = 1)
        count      = count + 1  
    return population

############################################################################