############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: CTAEA

# Citation: 
# PEREIRA, V. (2021). Project: pyMultiojective, File: ctaea.py, GitHub repository: <https://github.com/Valdecy/pyMultiojective>

############################################################################

# Required Libraries
import numpy  as np
import random
import os

import copy
import math
import matplotlib.pyplot as plt

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
    parent_1    = 0
    parent_2    = 1
    b_offspring = 0  
    rank_ca     = fast_non_dominated_sorting(ca, number_of_functions = len(list_of_functions))
    rank_ca     = rank_ca[rank_ca == 1]
    p_ca        = rank_ca.shape[0]/ca.shape[0]
    rank_da     = fast_non_dominated_sorting(da, number_of_functions = len(list_of_functions))
    rank_da     = rank_da[rank_da == 1]
    p_da        = rank_da.shape[0]/da.shape[0]
    for i in range (0, offspring.shape[0]):
        if (p_ca > p_da):
            parent_1 = random.sample(range(0, len(ca) - 1), 1)
        else:
            parent_1 = random.sample(range(0, len(da) - 1), 1)
        rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
        if (rand < p_ca):
            parent_2 = random.sample(range(0, len(ca) - 1), 1)
        else:
            parent_2 = random.sample(range(0, len(da) - 1), 1)
        for j in range(0, offspring.shape[1] - len(list_of_functions)):
            rand   = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            rand_b = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                                
            if (rand <= 0.5):
                b_offspring = 2*(rand_b)
                b_offspring = b_offspring**(1/(mu + 1))
            elif (rand > 0.5):  
                b_offspring = 1/(2*(1 - rand_b))
                b_offspring = b_offspring**(1/(mu + 1))       
            offspring[i,j] = np.clip(((1 + b_offspring)*ca[parent_1, j] + (1 - b_offspring)*ca[parent_2, j])/2, min_values[j], max_values[j])           
            if(i < ca.shape[0] - 1):   
                offspring[i+1,j] = np.clip(((1 - b_offspring)*ca[parent_1, j] + (1 + b_offspring)*ca[parent_2, j])/2, min_values[j], max_values[j]) 
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

# Function: Perpendicular Distance (Adapted from PYMOO)
#def perpendicular_distance(x, weights):
    #u           = np.tile(weights, (len(x), 1))
    #v           = np.repeat(x, len(weights), axis = 0)
    #n_u         = np.linalg.norm(u, axis = 1)
    #scalar_proj = np.sum(v * u, axis = 1) / n_u
    #proj        = scalar_proj[:, None] * u / n_u[:, None]
    #matrix      = np.reshape(np.linalg.norm(proj - v, axis = 1), (len(x), len(weights)))
    #return matrix

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
def constrained_two_archive_evolutionary_algorithm(references = 5, mutation_rate = 0.1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2], generations = 50, mu = 1, eta = 1, k = 4, alpha = 2, fr = 0.2):       
    count      = 0
    references = max(5, references)
    M          = len(list_of_functions)
    weights    = reference_points(M = M, p = references)
    size       = k*weights.shape[0]
    ca         = initial_population(size, min_values, max_values, list_of_functions) 
    da         = initial_population(size, min_values, max_values, list_of_functions)
    print('Total Number of Points on Reference Hyperplane: ', int(weights.shape[0]), ' Population Size: ', int(size))
    while (count <= generations):       
        print('Generation = ', count)
        offspring  = breeding(ca, da, min_values, max_values, mu, list_of_functions, size)
        offspring  = mutation(offspring, mutation_rate, eta, min_values, max_values, list_of_functions) 
        ca         = np.vstack([ca, offspring])
        ca, _      = association(ca, weights, M)
        ca         = update_ca(ca, offspring, weights, M)
        da         = update_da(ca, da, offspring, weights, M)
        count      = count + 1
    return da

######################## Part 1 - Usage ####################################

# Schaffer Function 1
def schaffer_f1(variables_values = [0]):
    y = variables_values[0]**2
    return y

# Schaffer Function 2
def schaffer_f2(variables_values = [0]):
    y = (variables_values[0]-2)**2
    return y

# Shaffer Data Points
x        = np.arange(0.0, 2.0, 0.01)
schaffer = np.zeros((x.shape[0], 3))
for i in range (0, x.shape[0]):
    schaffer[i,0] = x[i]
    schaffer[i,1] = schaffer_f1(variables_values = [schaffer[i,0]])
    schaffer[i,2] = schaffer_f2(variables_values = [schaffer[i,0]])

schaffer_1 = schaffer[:,1]
schaffer_2 = schaffer[:,2]
rfp        = 5
ref        = reference_points(M = 2, p = rfp)

# Calling CTAEA Function
ctaea_schaffer = constrained_two_archive_evolutionary_algorithm(references = rfp, mutation_rate = 0.15, min_values = [-5], max_values = [5], list_of_functions = [schaffer_f1, schaffer_f2], generations = 500, mu = 1, eta = 1, k = 7)

# Graph Pareto Front Solutions
plt.style.use('bmh')
plt.figure(figsize = (15, 15))
plt.xlabel('Function 1', fontsize = 12)
plt.ylabel('Function 2', fontsize = 12)
plt.plot([0,0], [4,0], c = 'b', alpha = 0.5, linewidth = 0.5)
for i in range(1, ref.shape[0]):
    x1, y1 = [0, 0]
    x2, y2 = ref[i,:]
    a      =  (y2 - y1) / (x2 - x1)
    b      = y1 - a * x1
    x      = np.linspace(0, 4, 100)
    y      = a * x + b
    y      = y[ y <= 4]
    plt.plot(x[:y.shape[0]], y, c = 'b', alpha = 0.5, linewidth = 0.5)
plt.scatter(ctaea_schaffer[:,-2], ctaea_schaffer[:,-1], c = 'r', s = 45, marker = 'o', label = 'CTEA')
plt.scatter(schaffer_1,              schaffer_2,              c = 'k', s = 2,  marker = '.', label = 'Solutions')
plt.scatter(ref[:, 0],               ref[:, 1],               c = 'b', s = 45, marker = 'o', label = 'Reference')
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

# Kursawe Data Points
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

# Calling CTAEA Function
ctaea_kursawe = constrained_two_archive_evolutionary_algorithm(references = 25, mutation_rate = 0.1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [kursawe_f1, kursawe_f2], generations = 250, mu = 1, eta = 1, k = 4)

# Graph Pareto Front Solutions
plt.style.use('bmh')
plt.figure(figsize = (15, 15))
plt.xlabel('Function 1', fontsize = 12)
plt.ylabel('Function 2', fontsize = 12)
plt.scatter(ctaea_kursawe[:,-2], ctaea_kursawe[:,-1], c = 'r', s = 45, marker = 'o', label = 'CTAEA')
plt.scatter(kursawe_1,              kursawe_2,        c = 'k', s = 2,  marker = '.', label = 'Solutions')
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

# Calling CTAEA Function
ctaea_dent = constrained_two_archive_evolutionary_algorithm(references = 25, mutation_rate = 0.1, min_values =  [-1.5,-1.5], max_values =  [1.5, 1.5], list_of_functions = [dent_f1, dent_f2], generations = 500, mu = 1, eta = 1, k = 4)

# Graph Pareto Front Solutions
plt.style.use('bmh')
plt.figure(figsize = (15, 15))
plt.xlabel('Function 1', fontsize = 12)
plt.ylabel('Function 2', fontsize = 12)
plt.scatter(dent_1,           dent_2,           c = 'k', s = 10,  marker = 'o', label = 'Solutions', alpha = 0.7)
plt.scatter(ctaea_dent[:,-2], ctaea_dent[:,-1], c = 'r', s = 35,  marker = 'o', label = 'CTAEA')
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
dtzl2 = np.loadtxt('DTLZ2_B.txt', delimiter = ';')

# Calling CTAEA Function
ctaea_dtzl2 = constrained_two_archive_evolutionary_algorithm(references = 12, mutation_rate = 0.1, min_values =  [0]*12, max_values =  [1]*12, list_of_functions = [DTZL2_1, DTZL2_2, DTZL2_3], generations = 1000, mu = 1, eta = 1, k = 2)

# Graph Pareto Front Solutions
plt.style.use('bmh')
fig = plt.figure(figsize = (15, 15))
ax  = fig.add_subplot(111, projection = '3d')
ax.set_xlabel('$f_1$', fontsize = 25, labelpad = 20)
ax.set_ylabel('$f_2$', fontsize = 25, labelpad = 20)
ax.set_zlabel('$f_3$', fontsize = 25, labelpad = 20)
ax.scatter( ctaea_dtzl2[:,-3], ctaea_dtzl2[:,-2], ctaea_dtzl2[:,-1], c = 'r', s = 35,  marker = 'o', label = 'CTAEA')
ax.scatter( dtzl2[:,-3], dtzl2[:,-2], dtzl2[:,-1], c = 'k', s = 15,  marker = '.', label = 'PF')
ax.legend(loc = 'upper right')
end   = 13
start = 0
gl_0 = list(range(start , start  + end ))
for i in range(0, len(gl_0)):
    gl_1 = list(range(start , start  + end ))
    if (i == 0):
        gl_2 = [i]
        gl_3 = [i]
        for j in range(0, len(gl_1)-1):
            gl_2.append(gl_2[-1]+ end - j)
        gl_0 = copy.deepcopy(gl_2)
    else:
        for j in range(0, len(gl_2)):
            gl_2[j] = gl_2[j]+1
        del gl_2[-1]
        for j in range(0, len(gl_3)):
            if ( gl_3[j] + 1 < dtzl2.shape[0]):
                gl_3[j] = gl_3[j] + 1
        gl_3.append(gl_0[i])
    ax.plot(dtzl2[gl_1,-3], dtzl2[gl_1,-2], dtzl2[gl_1,-1], c = 'k', linewidth = 0.3)
    ax.plot(dtzl2[gl_2,-3], dtzl2[gl_2,-2], dtzl2[gl_2,-1], c = 'k', linewidth = 0.3)
    ax.plot(dtzl2[gl_3,-3], dtzl2[gl_3,-2], dtzl2[gl_3,-1], c = 'k', linewidth = 0.3)
    start  = start  + end
    end    = end - 1
plt.show()

############################################################################