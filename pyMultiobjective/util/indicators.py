############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Multivariate Indicators

# Citation: 
# PEREIRA, V. (2022). GitHub repository: <https://github.com/Valdecy/pyMultiojective>

############################################################################

# Required Libraries
import itertools
import numpy as np
import pygmo as pg

from scipy import spatial

############################################################################

# Available Indicators:
    
    # GD          (https://apps.dtic.mil/sti/pdfs/ADA364478.pdf)
    # GD+         (https://doi.org/10.1007/978-3-319-15892-1_8)
    # IGD         (https://doi.org/10.1007/978-3-540-24694-7_71)
    # IGD+        (https://doi.org/10.1007/978-3-319-15892-1_8)
    # MS          (https://doi.org/10.1162/106365600568202)
    # SP          (https://doi.org/10.1109/TEVC.2006.882428)
    # Hypervolume (https://scholar.afit.edu/cgi/viewcontent.cgi?article=6130&context=etd)
    
############################################################################

# Helper Functions
 
# Functon: Generate Data Points
def generate_points(min_values = [-5, -5], max_values = [5, 5], list_of_functions = [], step = [0.1, 0.1], pf_min = True):
    x = []
    for j in range(0, len(min_values)):
        values = np.arange(min_values[j], max_values[j] + step[j], step[j])
        x.append(values)
    cartesian_product = list(itertools.product(*x)) 
    front             = np.array(cartesian_product, dtype = np.dtype('float'))
    front             = np.c_[ front, np.zeros( (len(cartesian_product), len(list_of_functions))) ]
    for j in range(0, len(list_of_functions)):
        value = [list_of_functions[j](item) for item in cartesian_product]
        front[:, len(min_values) + j] = value
    return front

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

# GD - Generational Distance

# Function: GD
def gd_indicator(min_values = [-5, -5], max_values = [5, 5], list_of_functions = [], step = [0.1, 0.1], solution = [], custom_pf = [], pf_min = True):
    if (solution.shape[1] > len(min_values)):
        sol = solution[:,len(min_values):]
    elif (solution.shape[1] == len(min_values)):
        sol = np.copy(solution)
    if (len(custom_pf) > 0):
        front = np.copy(custom_pf)
    else:
        front = generate_points(min_values, max_values, list_of_functions, step, pf_min)
        pf    = pareto_front_points(pts = front[:,len(min_values):], pf_min = pf_min)
        front = front[pf, len(min_values):]
    d_i = [ ( spatial.KDTree(front).query(sol[i,:]) ) for i in range(0, sol.shape[0]) ]
    d   = [item[0] for item in d_i]
    gd  = np.sqrt(sum(d))/len(d)
    return gd

############################################################################

# GD+ - Generational Distance Plus

# Function: GD+
def gd_plus_indicator(min_values = [-5, -5], max_values = [5, 5], list_of_functions = [], step = [0.1, 0.1], solution = [], custom_pf = [], pf_min = True):
    if (solution.shape[1] > len(min_values)):
        sol = solution[:,len(min_values):]
    elif (solution.shape[1] == len(min_values)):
        sol = np.copy(solution)
    if (len(custom_pf) > 0):
        front = np.copy(custom_pf)
    else:
        front = generate_points(min_values, max_values, list_of_functions, step, pf_min)
        pf    = pareto_front_points(pts = front[:,len(min_values):], pf_min = pf_min)
        front = front[pf, len(min_values):]
    d_i = [ ( spatial.KDTree(front).query(sol[i,:]) ) for i in range(0, sol.shape[0]) ]
    idx = [item[1] for item in d_i]
    s   = [max(max(sol[i,:] - front[idx[i],:]), 0)**2 for i in range(0, sol.shape[0])]
    gdp = np.sqrt(sum(s))/len(s)
    return gdp

############################################################################

# IGD - Inverted Generational Distance

# Function: IGD
def igd_indicator(min_values = [-5, -5], max_values = [5, 5], list_of_functions = [], step = [0.1, 0.1], solution = [], custom_pf = [], pf_min = True):
    if (solution.shape[1] > len(min_values)):
        sol = solution[:,len(min_values):]
    elif (solution.shape[1] == len(min_values)):
        sol = np.copy(solution)
    if (len(custom_pf) > 0):
        front = np.copy(custom_pf)
    else:
        front = generate_points(min_values, max_values, list_of_functions, step, pf_min)
        pf    = pareto_front_points(pts = front[:,len(min_values):], pf_min = pf_min)
        front = front[pf, len(min_values):]
    d_i = [ ( spatial.KDTree(sol).query(front[i,:]) ) for i in range(0, front.shape[0]) ]
    d   = [item[0] for item in d_i]
    igd = np.sqrt(sum(d))/len(d)
    return igd

############################################################################

# IGD+ - Inverted Generational Distance Plus

# Function: IGD+
def igd_plus_indicator(min_values = [-5, -5], max_values = [5, 5], list_of_functions = [], step = [0.1, 0.1], solution = [], custom_pf = [], pf_min = True):
    if (solution.shape[1] > len(min_values)):
        sol = solution[:,len(min_values):]
    elif (solution.shape[1] == len(min_values)):
        sol = np.copy(solution)
    if (len(custom_pf) > 0):
        front = np.copy(custom_pf)
    else:
        front = generate_points(min_values, max_values, list_of_functions, step, pf_min)
        pf    = pareto_front_points(pts = front[:,len(min_values):], pf_min = pf_min)
        front = front[pf, len(min_values):]
    d_i  = [ ( spatial.KDTree(sol).query(front[i,:]) ) for i in range(0, front.shape[0]) ]
    idx  = [item[1] for item in d_i]
    s    = [max(max(sol[idx[i],:] - front[i,:]), 0)**2 for i in range(0, front.shape[0])]
    igdp = np.sqrt(sum(s))/len(s)
    return igdp

############################################################################

# MS - Maximum Spread

# Function:  Maximum Spread
def ms_indicator(min_values = [-5, -5], max_values = [5, 5], list_of_functions = [], step = [0.1, 0.1], solution = [], custom_pf = [], pf_min = True):
    if (solution.shape[1] > len(min_values)):
        sol = solution[:,len(min_values):]
    elif (solution.shape[1] == len(min_values)):
        sol = np.copy(solution)
    if (len(custom_pf) > 0):
        front = np.copy(custom_pf)
    else:
        front = generate_points(min_values, max_values, list_of_functions, step, pf_min)
        pf    = pareto_front_points(pts = front[:,len(min_values):], pf_min = pf_min)
        front = front[pf, len(min_values):]
    s_max = np.max(sol, axis = 0)
    s_min = np.min(sol, axis = 0)
    f_max = np.max(front, axis = 0)
    f_min = np.min(front, axis = 0)
    ms = 0
    for i in range(0, len(list_of_functions)):
        ms = ms + ((min(s_max[i], f_max[i]) - max(s_min[i], f_min[i]))/(f_max[i] - f_min[i]))**2
    ms = np.sqrt(ms/len(list_of_functions))
    return ms

# SP - Spacing

# Function:  Spacing
def sp_indicator(min_values = [-5, -5], max_values = [5, 5], list_of_functions = [], step = [0.1, 0.1], solution = [], custom_pf = [], pf_min = True):
    if (solution.shape[1] > len(min_values)):
        sol = solution[:,len(min_values):]
    elif (solution.shape[1] == len(min_values)):
        sol = np.copy(solution)
    dm = np.zeros(sol.shape[0])
    for i in range(0, sol.shape[0]):
        dm[i] = min([np.linalg.norm(sol[i] - sol[j]) for j in range(0, sol.shape[0]) if i != j])
    d_mean  = np.mean(dm)
    spacing = np.sqrt(np.sum((dm - d_mean)**2)/sol.shape[0])
    return spacing

############################################################################

# Hypervolume (S-Metric)

# Function: Hypervolume
def hv_indicator(solution = [], n_objs = 3, ref_point = [], normalize = False):
    if (solution.shape[1] > n_objs):
        sol = solution[:,-n_objs:]
    elif (solution.shape[1] == n_objs):
        sol = np.copy(solution)
    if (normalize == True):
        z_min     = np.min(sol, axis = 0)
        z_max     = np.max(sol, axis = 0)
        sol       = np.clip((sol - z_min)/(z_max - z_min + 0.000000001), 0, 1)
        ref_point = [1]*n_objs
    if (len(ref_point) == 0):
        ref_point = [np.max(sol[:,j]) for j in range(0, sol.shape[1])]
    else:
        for j in range(0, len(ref_point)):
            if (ref_point[j] < np.max(sol[:,j])):
                print('Reference Point is Invalid: Outside Boundary')
                print('Correcting Position', j, '; Reference Point Value', ref_point[j], 'was changed to', np.max(sol[:,j]))
                print('')
                ref_point[j] = np.max(sol[:,j])
        print('Used Reference Point: ', ref_point, '; Normalization Procedure: ', normalize)
        print('')
    hv_c = pg.hypervolume(sol)
    hv   = hv_c.compute(ref_point)
    return hv

############################################################################
