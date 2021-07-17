############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: C-NSGA-II

# Citation: 
# PEREIRA, V. (2021). Project: pyMultiojective, File: c_n_ii.py, GitHub repository: <https://github.com/Valdecy/pyMultiojective>

############################################################################

# Required Libraries
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
def sort_population_by_rank(population, rank):
    idx        = np.argsort(rank[:,0], axis = 0).tolist()
    rank       = rank[idx,:]
    population = population[idx,:]
    return population, rank

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

# Function: Raw Fitness
def raw_fitness_function(population, number_of_functions = 2):    
    strength    = np.zeros((population.shape[0], 1))
    raw_fitness = np.zeros((population.shape[0], 1))
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if (i != j):
                if dominance_function(solution_1 = population[i,:], solution_2 = population[j,:], number_of_functions = number_of_functions):
                    strength[i,0] = strength[i,0] + 1
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if (i != j):
                if dominance_function(solution_1 = population[i,:], solution_2 = population[j,:], number_of_functions = number_of_functions):
                    raw_fitness[j,0] = raw_fitness[j,0] + strength[i,0]
    return raw_fitness

# Function: Build Distance Matrix
def euclidean_distance(coordinates):
   a = coordinates
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

# Function: Fitness
def fitness_calculation(population, raw_fitness, number_of_functions = 2):
    k        = int(len(population)**(1/2)) - 1
    fitness  = np.zeros((population.shape[0], 1))
    distance = euclidean_distance(population[:,population.shape[1]-number_of_functions:])
    for i in range(0, fitness.shape[0]):
        distance_ordered = (distance[distance[:,i].argsort()]).T
        fitness[i,0]     = raw_fitness[i,0] + 1/(distance_ordered[i,k] + 2)
    return fitness


# Function: Selection
def roulette_wheel(fitness_new): 
    fitness = np.zeros((fitness_new.shape[0], 2))
    for i in range(0, fitness.shape[0]):
        fitness[i,0] = 1/(1+ fitness[i,0] + abs(fitness[:,0].min()))
    fit_sum      = fitness[:,0].sum()
    fitness[0,1] = fitness[0,0]
    for i in range(1, fitness.shape[0]):
        fitness[i,1] = (fitness[i,0] + fitness[i-1,1])
    for i in range(0, fitness.shape[0]):
        fitness[i,1] = fitness[i,1]/fit_sum
    ix     = 0
    random = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if (random <= fitness[i, 1]):
          ix = i
          break
    return ix

############################################################################

# Function: Offspring
def breeding(population, fitness, min_values = [-5,-5], max_values = [5,5], mu = 1, list_of_functions = [func_1, func_2]):
    offspring   = np.copy(population)
    b_offspring = 0
    for i in range (0, offspring.shape[0]):
        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1 == parent_2:
            parent_2 = random.sample(range(0, len(population) - 1), 1)[0]
        for j in range(0, offspring.shape[1] - len(list_of_functions)):
            rand   = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
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
                rand   = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
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

# C-NSGA II Function
def clustered_non_dominated_sorting_genetic_algorithm_II(population_size = 5, mutation_rate = 0.1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2], generations = 50, mu = 1, eta = 1):        
    count      = 0   
    population = initial_population(population_size = population_size, min_values = min_values, max_values = max_values, list_of_functions = list_of_functions)  
    offspring  = initial_population(population_size = population_size, min_values = min_values, max_values = max_values, list_of_functions = list_of_functions)  
    while (count <= generations):       
        print("Generation = ", count)
        population        = np.vstack([population, offspring])
        rank              = fast_non_dominated_sorting(population, number_of_functions = len(list_of_functions))
        population, rank  = sort_population_by_rank(population, rank)
        population, rank  = population[0:population_size,:], rank[0:population_size,:] 
        raw_fitness       = raw_fitness_function(population, number_of_functions = len(list_of_functions))
        fitness           = fitness_calculation(population, raw_fitness, number_of_functions = len(list_of_functions))  
        offspring         = breeding(population, fitness, mu = mu, min_values = min_values, max_values = max_values, list_of_functions = list_of_functions)
        offspring         = mutation(offspring, mutation_rate = mutation_rate, eta = eta, min_values = min_values, max_values = max_values, list_of_functions = list_of_functions)             
        count             = count + 1              
    return population

######################## Part 1 - Usage ####################################

# Schaffer Function 1
def schaffer_f1(variables_values = [0]):
    y = variables_values[0]**2
    return y

# Schaffer Function 2
def schaffer_f2(variables_values = [0]):
    y = (variables_values[0]-2)**2
    return y

# Shaffer Pareto Front
x        = np.arange(0.0, 2.0, 0.01)
schaffer = np.zeros((x.shape[0], 3))
for i in range (0, x.shape[0]):
    schaffer[i,0] = x[i]
    schaffer[i,1] = schaffer_f1(variables_values = [schaffer[i,0]])
    schaffer[i,2] = schaffer_f2(variables_values = [schaffer[i,0]])

schaffer_1 = schaffer[:,1]
schaffer_2 = schaffer[:,2]

# Calling C-NSGA II Function
c_nsga_II_schaffer = clustered_non_dominated_sorting_genetic_algorithm_II(population_size = 25, mutation_rate = 0.1, min_values = [-5], max_values = [5], list_of_functions = [schaffer_f1, schaffer_f2], generations = 100, mu = 10, eta = 10)

# Graph Pareto Front Solutions
plt.style.use('bmh')
plt.figure(figsize = (15, 15))
plt.xlabel('Function 1', fontsize = 12)
plt.ylabel('Function 2', fontsize = 12)
plt.scatter(c_nsga_II_schaffer[:,-2], c_nsga_II_schaffer[:,-1],  c = 'r', s = 45, marker = 'o', label = 'C-NSGA-II')
plt.scatter(schaffer_1,             schaffer_2,              c = 'k', s = 2,  marker = '.', label = 'Solutions')
plt.legend(loc = 'upper right')
plt.show()

######################## Part 2 - Usage ####################################

# Kursawe Function 1
def kursawe_f1(variables_values = [0, 0]):
    f1 = 0
    if (len(variables_values) == 1):
        f1 = f1 - 10 * math.exp(-0.2 * math.sqrt(variables_values[0]**2 + variables_values[0]**2))
    else:
        for i in range(0, len(variables_values)-1):
            f1 = f1 - 10 * math.exp(-0.2 * math.sqrt(variables_values[i]**2 + variables_values[i + 1]**2))
    return f1

# Kursawe Function 2
def kursawe_f2(variables_values = [0, 0]):
    f2 = 0
    for i in range(0, len(variables_values)):
        f2 = f2 + abs(variables_values[i])**0.8 + 5 * math.sin(variables_values[i]**3)
    return f2

# Kursawe Pareto Front
x       = np.arange(-4, 4, 0.1)
kursawe = np.zeros((x.shape[0]**2, 4))
count   = 0
for j in range (0, x.shape[0]):
    for k in range (0, x.shape[0]):
            kursawe[count,0] = x[j]
            kursawe[count,1] = x[k]
            count            = count + 1
        
for i in range (0, kursawe.shape[0]):
    kursawe[i,2] = kursawe_f1(variables_values = [kursawe[i,0], kursawe[i,1]])
    kursawe[i,3] = kursawe_f2(variables_values = [kursawe[i,0], kursawe[i,1]])

kursawe_1 = kursawe[:,2]
kursawe_2 = kursawe[:,3]

# Calling C-NSGA II Function
c_nsga_II_kursawe = clustered_non_dominated_sorting_genetic_algorithm_II(population_size = 75, mutation_rate = 0.15, min_values = [-5,-5], max_values = [5,5], list_of_functions = [kursawe_f1, kursawe_f2], generations = 2500, mu = 1, eta = 1)

# Graph Pareto Front Solutions
plt.style.use('bmh')
plt.figure(figsize = (15,15))
plt.xlabel('Function 1', fontsize = 12)
plt.ylabel('Function 2', fontsize = 12)
plt.scatter(c_nsga_II_kursawe[:,-2], c_nsga_II_kursawe[:,-1], c = 'r', s = 45, marker = 'o', label = 'C-NSGA-II')
plt.scatter(kursawe_1,              kursawe_2,              c = 'k', s = 2,  marker = '.', label = 'Solutions')
plt.legend(loc = 'upper right')
plt.show()

######################## Part 3 - Usage ####################################

# Dent Function 1
def dent_f1(variables_values = [0, 0]):
    d  = 0.1 * math.exp(-(variables_values[0] - variables_values[1]) ** 2)  
    f1 = 0.50 * (math.sqrt(1 + (variables_values[0] + variables_values[1]) ** 2) + math.sqrt(1 + (variables_values[0] - variables_values[1]) ** 2) + variables_values[0] - variables_values[1]) + d
    return f1

# Dent Function 2
def dent_f2(variables_values = [0, 0]):
    d  = 0.1 * math.exp(-(variables_values[0] - variables_values[1]) ** 2)  
    f2 = 0.50 * (math.sqrt(1 + (variables_values[0] + variables_values[1]) ** 2) + math.sqrt(1 + (variables_values[0] - variables_values[1]) ** 2) - variables_values[0] - variables_values[1]) + d
    return f2

# Dent Data Points
x     = np.arange(-1.5, 1.6, 0.25)
dent  = np.zeros((x.shape[0]**2, 4))
count = 0
for j in range (0, x.shape[0]):
    for k in range (0, x.shape[0]):
            dent[count,0] = x[j]
            dent[count,1] = x[k]
            count         = count + 1
        
for i in range (0, dent.shape[0]):
    dent[i,2] = dent_f1(variables_values = [dent[i,0], dent[i,1]])
    dent[i,3] = dent_f2(variables_values = [dent[i,0], dent[i,1]])

dent_1 = dent[:,2]
dent_2 = dent[:,3]

# Calling C-NSGA II Function
c_nsga_II = clustered_non_dominated_sorting_genetic_algorithm_II(population_size = 16, mutation_rate = 0.10, min_values =  [-1.5,-1.5], max_values = [1.5,1.5], list_of_functions = [dent_f1, dent_f2], generations = 500, mu = 1, eta = 1)

# Graph Pareto Front Solutions
plt.style.use('bmh')
plt.figure(figsize = (15,15))
plt.xlabel('Function 1', fontsize = 12)
plt.ylabel('Function 2', fontsize = 12)
plt.scatter(c_nsga_II[:,-2], c_nsga_II[:,-1], c = 'r', s = 35,  marker = 'o', label = 'C-NSGA-II')
plt.scatter(dent_1,           dent_2,           c = 'k', s = 10,  marker = 'o', label = 'Solutions', alpha = 0.7)
plt.legend(loc = 'upper right')
plt.show()

######################## Part 4 - Usage ####################################

# DTZL2 Function 1
def DTZL2_1(variables_values = []):
    g = 0
    for i in range(2, len(variables_values)):
        g = g + (variables_values[i] - 0.5)**2 
    f = (1 + g)*(math.cos(variables_values[0]*math.pi*(1/2)) * math.cos(variables_values[1]*math.pi*(1/2)))
    return f

# DTZL2 Function 2
def DTZL2_2(variables_values = []):
    g = 0
    for i in range(2, len(variables_values)):
        g = g + (variables_values[i] - 0.5)**2 
    f = (1 + g)*(math.cos(variables_values[0]*math.pi*(1/2)) * math.sin(variables_values[1]*math.pi*(1/2)))
    return f

# DTZL2 Function 3
def DTZL2_3(variables_values = []):
    g = 0
    for i in range(2, len(variables_values)):
        g = g + (variables_values[i] - 0.5)**2 
    f = (1 + g)*(math.sin(variables_values[0]*math.pi*(1/2)))
    return f

# DTZL2 Data Points
dtzl2 = np.loadtxt('Datasets-B-DTLZ2.txt', delimiter = ';')

# Calling C-NSGA II Function
c_nsga_II_dtzl2 = clustered_non_dominated_sorting_genetic_algorithm_II(population_size = 25, mutation_rate = 0.10, min_values = [0]*12, max_values = [1]*12, list_of_functions = [DTZL2_1, DTZL2_2, DTZL2_3], generations = 5000, mu = 3, eta = 3)

# Graph Pareto Front Solutions
plt.style.use('bmh')
fig = plt.figure(figsize = (15, 15))
ax  = fig.add_subplot(111, projection = '3d')
ax.set_xlabel('$f_1$', fontsize = 25, labelpad = 20)
ax.set_ylabel('$f_2$', fontsize = 25, labelpad = 20)
ax.set_zlabel('$f_3$', fontsize = 25, labelpad = 20)
ax.scatter( c_nsga_II_dtzl2[:,-3], c_nsga_II_dtzl2[:,-2], c_nsga_II_dtzl2[:,-1], c = 'r', s = 35,  marker = 'o', label = 'C-NSGA-II')
ax.scatter( dtzl2[:,-3], dtzl2[:,-2], dtzl2[:,-1], c = 'b', s = 35,  marker = 'o', label = 'PF')
ax.legend(loc = 'upper right')
plt.show()

######################## Part 5 - Usage ####################################

# Viennet1 Function 1 
def viennet1_f1(variables_values = [0, 0]):
    f1 = 0.5*(variables_values[0]**2 + variables_values[1]**2) + math.sin(variables_values[0]**2 + variables_values[1]**2)
    return f1

# Viennet1 Function 2
def viennet1_f2(variables_values = [0, 0]):
    f2 = ((3*variables_values[0] - 2*variables_values[1] + 4)**2)/8 + ((variables_values[0] - variables_values[1] + 1)**2)/27 + 15
    return f2

# Viennet1 Function 3
def viennet1_f3(variables_values = [0, 0]):
    f3 = 1/(variables_values[0]**2 + variables_values[1]**2 + 1) - 1.1*math.exp(-(variables_values[0]**2 + variables_values[1]**2))
    return f3

# Viennet1 Data Points
x        = np.arange(-3, 3.1, 0.1)
viennet1 = np.zeros((x.shape[0]**2, 5))
count    = 0
for j in range (0, x.shape[0]):
    for k in range (0, x.shape[0]):
            viennet1[count,0] = x[j]
            viennet1[count,1] = x[k]
            count             = count + 1
        
for i in range (0, viennet1.shape[0]):
    viennet1[i,2] = viennet1_f1(variables_values = [viennet1[i,0], viennet1[i,1]])
    viennet1[i,3] = viennet1_f2(variables_values = [viennet1[i,0], viennet1[i,1]])
    viennet1[i,4] = viennet1_f3(variables_values = [viennet1[i,0], viennet1[i,1]])

viennet1_1 = viennet1[:,2]
viennet1_2 = viennet1[:,3]
viennet1_3 = viennet1[:,4]

# Calling C-NSGA II Function
c_nsga_II_viennet1 = clustered_non_dominated_sorting_genetic_algorithm_II(population_size = 25, mutation_rate = 0.10, min_values = [-3,-3], max_values =  [3, 3], list_of_functions = [viennet1_f1, viennet1_f2, viennet1_f3], generations = 2500, mu = 1, eta = 1)

# Graph Pareto Front Solutions
plt.style.use('bmh')
fig = plt.figure(figsize = (15, 15))
ax  = fig.add_subplot(111, projection = '3d')
ax.set_xlabel('$f_1$', fontsize = 25, labelpad = 20)
ax.set_ylabel('$f_2$', fontsize = 25, labelpad = 20)
ax.set_zlabel('$f_3$', fontsize = 25, labelpad = 20)
ax.scatter(viennet1_1,             viennet1_2,     viennet1_3,   c = 'k', s = 2,  marker = 'o', label = 'Solutions', alpha = 0.3)
ax.scatter(c_nsga_II_viennet1 [:,-3], c_nsga_II_viennet1 [:,-2], c_nsga_II_viennet1 [:,-1], c = 'r', s = 35,  marker = 'o', label = 'NSGA-II')
ax.legend(loc = 'upper right')
plt.show()

##############################################################################