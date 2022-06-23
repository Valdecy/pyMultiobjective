############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Test Functions

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMultiojective>

############################################################################

# Required Libraries
import numpy as np

############################################################################

# Available Test Functions:

    # Dent            (https://doi.org/10.1007/978-3-319-44003-3_12)
    # DTLZ1           (https://doi.org/10.1109/CEC.2002.1007032)
    # DTLZ2           (https://doi.org/10.1109/CEC.2002.1007032)
    # DTLZ3           (https://doi.org/10.1109/CEC.2002.1007032)
    # DTLZ4           (https://doi.org/10.1109/CEC.2002.1007032)
    # DTLZ5           (https://doi.org/10.1109/CEC.2002.1007032)
    # DTLZ6           (https://doi.org/10.1109/CEC.2002.1007032)
    # DTLZ7           (https://doi.org/10.1109/CEC.2002.1007032)
    # Fonseca-Fleming (https://doi.org/10.1162/evco.1995.3.1.1)
    # Kursawe         (https://doi.org/10.1007/BFb0029752)
    # Poloni          (https://www.researchgate.net/publication/243686783_Hybrid_GA_for_multi_objective_aerodynamic_shape_optimization)
    # Schaffer1       (https://www.researchgate.net/publication/236443691_Some_Experiments_in_Machine_Learning_Using_Vector_Evaluated_Genetic_Algorithms)
    # Schaffer2       (https://www.researchgate.net/publication/236443691_Some_Experiments_in_Machine_Learning_Using_Vector_Evaluated_Genetic_Algorithms)
    # ZDT1            (https://doi.org/10.1162/106365600568202) 
    # ZDT2            (https://doi.org/10.1162/106365600568202)
    # ZDT3            (https://doi.org/10.1162/106365600568202)
    # ZDT4            (https://doi.org/10.1162/106365600568202)
    # ZDT6            (https://doi.org/10.1162/106365600568202)
    # Viennet1        (https://doi.org/10.1080/00207729608929211)
    # Viennet2        (https://doi.org/10.1080/00207729608929211)
    # Viennet3        (https://doi.org/10.1080/00207729608929211)
    
############################################################################

# Dent (2 Target Functions; 2 Decision Variables; Domain -5 <= x1, x2 <= 5)

# Dent - Target Function 1
def dent_f1(variables_values = [0, 0]):
    d  = 0.85 * np.exp(-(variables_values[0] - variables_values[1]) ** 2)  
    f1 = 0.50 * (np.sqrt(1 + (variables_values[0] + variables_values[1]) ** 2) + np.sqrt(1 + (variables_values[0] - variables_values[1]) ** 2) + variables_values[0] - variables_values[1]) + d
    return f1

# Dent - Target Function 2
def dent_f2(variables_values = [0, 0]):
    d  = 0.85 * np.exp(-(variables_values[0] - variables_values[1]) ** 2)  
    f2 = 0.50 * (np.sqrt(1 + (variables_values[0] + variables_values[1]) ** 2) + np.sqrt(1 + (variables_values[0] - variables_values[1]) ** 2) - variables_values[0] + variables_values[1]) + d
    return f2

############################################################################

# Deb-Thiele-Zitzler F1 (3 Target Functions; N Decision Variables; Domain 0 <= xi <= 1)

# DTLZ1 - Target Function 1
def DTLZ_1_f1(variables_values = []):
    g = 0
    M = 3
    k = len(variables_values) - M + 1
    for i in range(len(variables_values)-k, len(variables_values)):
        g = g + (variables_values[i] - 0.5)**2 - np.cos(20*np.pi*(variables_values[i] - 0.5))
    g = 100*(k + g)
    f = (1 + g)*(1/2)
    for i in range(0, len(variables_values)-1):
        f = f*variables_values[i]
    return f

# DTLZ1 - Target Function 2
def DTLZ_1_f2(variables_values = []):
    g = 0
    M = 3
    k = len(variables_values) - M + 1
    for i in range(len(variables_values)-k, len(variables_values)):
        g = g + (variables_values[i] - 0.5)**2 - np.cos(20*np.pi*(variables_values[i] - 0.5))
    g = 100*(k + g)
    f = (1 + g)*(1/2)
    for i in range(0, len(variables_values)-2):
        f = f*variables_values[i]
    f = f*(1-variables_values[-2])
    return f

# DTLZ1 - Target Function 3
def DTLZ_1_f3(variables_values = []):
    g = 0
    M = 3
    k = len(variables_values) - M + 1
    for i in range(len(variables_values)-k, len(variables_values)):
        g = g + (variables_values[i] - 0.5)**2 - np.cos(20*np.pi*(variables_values[i] - 0.5))
    g = 100*(k + g)
    f = (1 + g)*(1/2)*(1 - variables_values[0])
    return f

############################################################################

# Deb-Thiele-Zitzler F2 (3 Target Functions; N Decision Variables; Domain 0 <= xi <= 1)

# DTLZ2 - Target Function 1
def DTLZ_2_f1(variables_values = []):
    g = 0
    M = 3
    k = len(variables_values) - M + 1
    for i in range(len(variables_values)-k, len(variables_values)):
        g = g + (variables_values[i] - 0.5)**2 
    f = (1 + g)
    for i in range(0, len(variables_values)-1):
        f = f*(np.cos(variables_values[i]*np.pi*(1/2)))
    return f

# DTLZ2 - Target Function 2
def DTLZ_2_f2(variables_values = []):
    g = 0
    M = 3
    k = len(variables_values) - M + 1
    for i in range(len(variables_values)-k, len(variables_values)):
        g = g + (variables_values[i] - 0.5)**2 
    f = (1 + g)
    for i in range(0, len(variables_values)-2):
        f = f*(np.cos(variables_values[i]*np.pi*(1/2)))
    f = f*(np.sin(variables_values[-2]*np.pi*(1/2)))
    return f

# DTLZ2 - Target Function 3
def DTLZ_2_f3(variables_values = []):
    g = 0
    M = 3
    k = len(variables_values) - M + 1
    for i in range(len(variables_values)-k, len(variables_values)):
        g = g + (variables_values[i] - 0.5)**2 
    f = (1 + g)*(np.sin(variables_values[0]*np.pi*(1/2)))
    return f

############################################################################

# Deb-Thiele-Zitzler F3 (3 Target Functions; N Decision Variables; Domain 0 <= xi <= 1)

# DTLZ3 - Target Function 1
def DTLZ_3_f1(variables_values = []):
    g = 0
    M = 3
    k = len(variables_values) - M + 1
    for i in range(len(variables_values)-k, len(variables_values)):
        g = g + (variables_values[i] - 0.5)**2 - np.cos(20*np.pi*(variables_values[i] - 0.5))
    g = 100*(k + g)
    f = (1 + g)
    for i in range(0, len(variables_values)-1):
        f = f*(np.cos(variables_values[i]*np.pi*(1/2)))
    return f

# DTLZ3 - Target Function 2
def DTLZ_3_f2(variables_values = []):
    g = 0
    M = 3
    k = len(variables_values) - M + 1
    for i in range(len(variables_values)-k, len(variables_values)):
        g = g + (variables_values[i] - 0.5)**2 - np.cos(20*np.pi*(variables_values[i] - 0.5))
    g = 100*(k + g)
    f = (1 + g)
    for i in range(0, len(variables_values)-2):
        f = f*(np.cos(variables_values[i]*np.pi*(1/2)))
    f = f*(np.sin(variables_values[-2]*np.pi*(1/2)))
    return f

# DTLZ3 - Target Function 3
def DTLZ_3_f3(variables_values = []):
    g = 0
    M = 3
    k = len(variables_values) - M + 1
    for i in range(len(variables_values)-k, len(variables_values)):
        g = g + (variables_values[i] - 0.5)**2 - np.cos(20*np.pi*(variables_values[i] - 0.5))
    g = 100*(k + g)
    f = (1 + g)*(np.sin(variables_values[0]*np.pi*(1/2)))
    return f

############################################################################

# Deb-Thiele-Zitzler F4 (3 Target Functions; N Decision Variables; Domain 0 <= xi <= 1)

# DTLZ4 - Target Function 1
def DTLZ_4_f1(variables_values = []):
    g = 0
    M = 3
    a = 100
    k = len(variables_values) - M + 1
    variables_values = list(variables_values)
    for i in range(len(variables_values)-k, len(variables_values)):
        variables_values[i] = variables_values[i]**a
    for i in range(len(variables_values)-k, len(variables_values)):
        g = g + (variables_values[i] - 0.5)**2 
    f = (1 + g)
    for i in range(0, len(variables_values)-1):
        f = f*(np.cos(variables_values[i]*np.pi*(1/2)))
    return f

# DTLZ4 - Target Function 2
def DTLZ_4_f2(variables_values = []):
    g = 0
    M = 3
    a = 100
    k = len(variables_values) - M + 1
    variables_values = list(variables_values)
    for i in range(len(variables_values)-k, len(variables_values)):
        variables_values[i] = variables_values[i]**a
    for i in range(len(variables_values)-k, len(variables_values)):
        g = g + (variables_values[i] - 0.5)**2 
    f = (1 + g)
    for i in range(0, len(variables_values)-2):
        f = f*(np.cos(variables_values[i]*np.pi*(1/2)))
    f = f*(np.sin(variables_values[-2]*np.pi*(1/2)))
    return f

# DTLZ4 - Target Function 3
def DTLZ_4_f3(variables_values = []):
    g = 0
    M = 3
    a = 100
    k = len(variables_values) - M + 1
    variables_values = list(variables_values)
    for i in range(len(variables_values)-k, len(variables_values)):
        variables_values[i] = variables_values[i]**a
    for i in range(len(variables_values)-k, len(variables_values)):
        g = g + (variables_values[i] - 0.5)**2 
    f = (1 + g)*(np.sin(variables_values[0]*np.pi*(1/2)))
    return f

############################################################################

# Deb-Thiele-Zitzler F5 (3 Target Functions; N Decision Variables; Domain 0 <= xi <= 1)

# DTLZ5 - Target Function 1
def DTLZ_5_f1(variables_values = []):
    g = 0
    M = 3
    k = len(variables_values) - M + 1
    for i in range(len(variables_values)-k, len(variables_values)):
        g = g + (variables_values[i] - 0.5)**2 
    f         = (1 + g)
    theta     = [0.0] * (M- 1)
    theta[0]  = variables_values[0] * np.pi / 2.0
    theta[1:] = [ (np.pi / (4.0 * (1.0 + g))) * (1.0 + 2.0 * g * variables_values[i]) for i in range(1, M - 1)]
    for i in range(0, len(theta)):
        f = f*(np.cos(theta[i]))
    return f

# DTLZ5 - Target Function 2
def DTLZ_5_f2(variables_values = []):
    g = 0
    M = 3
    k = len(variables_values) - M + 1
    for i in range(len(variables_values)-k, len(variables_values)):
        g = g + (variables_values[i] - 0.5)**2 
    f         = (1 + g)
    theta     = [0.0] * (M- 1)
    theta[0]  = variables_values[0] * np.pi / 2.0
    theta[1:] = [ (np.pi / (4.0 * (1.0 + g))) * (1.0 + 2.0 * g * variables_values[i]) for i in range(1, M - 1)]
    for i in range(0, len(theta)-1):
        f = f*(np.cos(theta[i]))
    f = f*(np.sin(theta[-1]))
    return f

# DTLZ5 - Target Function 3
def DTLZ_5_f3(variables_values = []):
    g = 0
    M = 3
    k = len(variables_values) - M + 1
    for i in range(len(variables_values)-k, len(variables_values)):
        g = g + (variables_values[i] - 0.5)**2 
    theta     = [0.0] * (M- 1)
    theta[0]  = variables_values[0] * np.pi / 2.0
    theta[1:] = [ (np.pi / (4.0 * (1.0 + g))) * (1.0 + 2.0 * g * variables_values[i]) for i in range(1, M - 1)]
    f         = (1+g)*(np.sin(theta[0]))
    return f

############################################################################

# Deb-Thiele-Zitzler F6 (3 Target Functions; N Decision Variables; Domain 0 <= xi <= 1)

# DTLZ6 - Target Function 1
def DTLZ_6_f1(variables_values = []):
    g = 0
    M = 3
    k = len(variables_values) - M + 1
    for i in range(len(variables_values)-k, len(variables_values)):
        g = g + (variables_values[i])**0.1 
    f         = (1 + g)
    theta     = [0.0] * (M- 1)
    theta[0]  = variables_values[0] * np.pi / 2.0
    theta[1:] = [ (np.pi / (4.0 * (1.0 + g))) * (1.0 + 2.0 * g * variables_values[i]) for i in range(1, M - 1)]
    for i in range(0, len(theta)):
        f = f*(np.cos(theta[i]))
    return f

# DTLZ6 - Target Function 2
def DTLZ_6_f2(variables_values = []):
    g = 0
    M = 3
    k = len(variables_values) - M + 1
    for i in range(len(variables_values)-k, len(variables_values)):
        g = g + (variables_values[i])**0.1
    f         = (1 + g)
    theta     = [0.0] * (M- 1)
    theta[0]  = variables_values[0] * np.pi / 2.0
    theta[1:] = [ (np.pi / (4.0 * (1.0 + g))) * (1.0 + 2.0 * g * variables_values[i]) for i in range(1, M - 1)]
    for i in range(0, len(theta)-1):
        f = f*(np.cos(theta[i]))
    f = f*(np.sin(theta[-1]))
    return f

# DTLZ6 - Target Function 3
def DTLZ_6_f3(variables_values = []):
    g = 0
    M = 3
    k = len(variables_values) - M + 1
    for i in range(len(variables_values)-k, len(variables_values)):
        g = g + (variables_values[i])**0.1
    theta     = [0.0] * (M- 1)
    theta[0]  = variables_values[0] * np.pi / 2.0
    theta[1:] = [ (np.pi / (4.0 * (1.0 + g))) * (1.0 + 2.0 * g * variables_values[i]) for i in range(1, M - 1)]
    f         = (1+g)*(np.sin(theta[0]))
    return f

############################################################################

# Deb-Thiele-Zitzler F7 (3 Target Functions; N Decision Variables; Domain 0 <= xi <= 1)

# DTLZ7 - Target Function 1
def DTLZ_7_f1(variables_values = []):
    f = variables_values[0]
    return f

# DTLZ7 - Target Function 2
def DTLZ_7_f2(variables_values = []):
    f = variables_values[1]
    return f

# DTLZ7 - Target Function 3
def DTLZ_7_f3(variables_values = []):
    g = 0
    M = 3
    k = len(variables_values) - M + 1
    for i in range(len(variables_values)-k, len(variables_values)):
        g = g + variables_values[i]
    g = 1 + (9/k)*g 
    h = 0
    for i in range(0, M - 1):
        h = h + (variables_values[i]/(1 + g))*np.sin(3*np.pi*variables_values[i])
    h = M - h
    f = (1 + g)*h
    return f

############################################################################

# Fonseca-Fleming (2 Target Functions; N Decision Variables; Domain -4 <= xi <= 4)

# Fonseca - Target Function 1
def fonseca_fleming_f1(variables_values = []):
    d  = 0
    for i in range(0, len(variables_values)):
        d = d + (variables_values[i] - 1/np.sqrt(len(variables_values)))**2
    f1 = 1 - np.exp(-d)
    return f1

# Fonseca - Target Function 2
def fonseca_fleming_f2(variables_values = []):
    d  = 0
    for i in range(0, len(variables_values)):
        d = d + (variables_values[i] + 1/np.sqrt(len(variables_values)))**2
    f2 = 1 - np.exp(-d)
    return f2

############################################################################

# Kursawe (2 Target Functions; 3 Decision Variables; Domain -5 <= x1, x2 <= 5)

# Kursawe - Target Function 1
def kursawe_f1(variables_values = [0, 0, 0]):
    f1 = 0
    if (len(variables_values) == 1):
        f1 = f1 - 10 * np.exp(-0.2 * np.sqrt(variables_values[0]**2 + variables_values[0]**2))
    else:
        for i in range(0, len(variables_values)-1):
            f1 = f1 - 10 * np.exp(-0.2 * np.sqrt(variables_values[i]**2 + variables_values[i + 1]**2))
    return f1

# Kursawe - Target Function 2
def kursawe_f2(variables_values = [0, 0, 0]):
    f2 = 0
    for i in range(0, len(variables_values)):
        f2 = f2 + abs(variables_values[i])**0.8 + 5 * np.sin(variables_values[i]**3)
    return f2

############################################################################

# Poloni (2 Target Functions; 2 Decision Variables; Domain -pi <= x1, x2 <= pi)

# Poloni - Target Function 1
def poloni_f1(variables_values = [0, 0]):
    A1 = 0.5*np.sin(1) - 2*np.cos(1) + 1*np.sin(2) - 1.5*np.cos(2)
    A2 = 1.5*np.sin(1) - 1*np.cos(1) + 2*np.sin(2) - 0.5*np.cos(2)
    B1 = 0.5*np.sin(variables_values[0]) - 2*np.cos(variables_values[0]) + 1*np.sin(variables_values[1]) - 1.5*np.cos(variables_values[1])
    B2 = 1.5*np.sin(variables_values[0]) - 1*np.cos(variables_values[0]) + 2*np.sin(variables_values[1]) - 0.5*np.cos(variables_values[1])
    f1 = 1 + (A1 - B1)**2 + (A2 - B2)**2
    return f1

# Poloni - Target Function 2
def poloni_f2(variables_values = [0, 0]):
    f2 = (variables_values[0] + 3)**2 + (variables_values[1] + 1)**2
    return f2

############################################################################

# Schaffer F1 (2 Target Functions; 1 Decision Variable; Domain -A <= x1 <= A, with 10 <= A <= 100000)

# Schaffer1 - Target Function 1
def schaffer_1_f1(variables_values = [0]):
    f1 = variables_values[0]**2
    return f1

# Schaffer1 - Target Function 2
def schaffer_1_f2(variables_values = [0]):
    f2 = (variables_values[0] - 2)**2
    return f2

############################################################################

# Schaffer F2 (2 Target Functions; 1 Decision Variable; Domain -5 <= xi <= 10)

# Schaffer2 - Target Function 1
def schaffer_2_f1(variables_values = [0]):
    f1 = 0
    if   (variables_values[0] <= 1):
        f1 = -variables_values[0]
    elif (variables_values[0] > 1 and variables_values[0] <= 3):
        f1 = variables_values[0] - 2
    elif (variables_values[0] > 3 and variables_values[0] <= 4):
        f1 = 4 - variables_values[0]
    elif (variables_values[0] > 4):
        f1 = variables_values[0] - 4
    return f1

# Schaffer2 - Target Function 2
def schaffer_2_f2(variables_values = [0]):
    f2 = (variables_values[0] - 5)**2
    return f2

############################################################################

# Zitzler–Deb–Thiele F1 (2 Target Functions; 30 Decision Variables; Domain 0 <= xi <= 1)

# ZDT1 - Target Function 1
def ZDT_1_f1(variables_values = []):
    f1 = variables_values[0]
    return f1

# ZDT1 - Target Function 2
def ZDT_1_f2(variables_values = []):
    d = 0
    for i in range(1, len(variables_values)):
        d = d + variables_values[i]
    g  = 1 + (9/29)*d
    h  = 1 - np.sqrt(variables_values[0]/g)
    f2 = g*h
    return f2

############################################################################

# Zitzler–Deb–Thiele F2 (2 Target Functions; 30 Decision Variables; Domain 0 <= xi <= 1)

# ZDT2 - Target Function 1
def ZDT_2_f1(variables_values = []):
    f1 = variables_values[0]
    return f1

# ZDT2 - Target Function 2
def ZDT_2_f2(variables_values = []):
    d = 0
    for i in range(1, len(variables_values)):
        d = d + variables_values[i]
    g  = 1 + (9/29)*d
    h  = 1 - (variables_values[0]/g)**2
    f2 = g*h
    return f2

############################################################################

 # Zitzler–Deb–Thiele F3 (2 Target Functions; 30 Decision Variables; Domain 0 <= xi <= 1)

# ZDT3 - Target Function 1
def ZDT_3_f1(variables_values = []):
    f1 = variables_values[0]
    return f1

# ZDT3 - Target Function 2
def ZDT_3_f2(variables_values = []):
    d = 0
    for i in range(1, len(variables_values)):
        d = d + variables_values[i]
    g  = 1 + (9/29)*d
    h  = 1 - np.sqrt(variables_values[0]/g) - (variables_values[0]/g)*np.sin(10*np.pi*variables_values[0])
    f2 = g*h
    return f2

############################################################################

# Zitzler–Deb–Thiele F4 (2 Target Functions; 10 Decision Variables; Domain 0 <= x1 <= 1; -5 <= xi <= 5)

# ZDT4 - Target Function 1
def ZDT_4_f1(variables_values = []):
    f1 = variables_values[0]
    return f1

# ZDT4 - Target Function 2
def ZDT_4_f2(variables_values = []):
    d = 0
    for i in range(1, len(variables_values)):
        d = d + variables_values[i]**2 - 10*np.cos(4*np.pi*variables_values[i])
    g  = 91 + d
    h  = 1 - np.sqrt(variables_values[0]/g)
    f2 = g*h
    return f2

############################################################################

# Zitzler–Deb–Thiele F6 (2 Target Functions; 10 Decision Variables; Domain 0 <= xi <= 1)

# ZDT6 - Target Function 1
def ZDT_6_f1(variables_values = []):
    f1 = 1 - np.exp(4*variables_values[0])*( (np.sin(6*np.pi*variables_values[0])**6) )
    return f1

# ZDT6 - Target Function 2
def ZDT_6_f2(variables_values = []):
    d = 0
    for i in range(1, len(variables_values)):
        d = d + variables_values[i]/9
    g  = 1 + 9*(d)**(0.25)
    h  = 1 - ((1 - np.exp(4*variables_values[0])*( (np.sin(6*np.pi*variables_values[0])**6) ))/g)**2
    f2 = g*h
    return f2

############################################################################

# Viennet 1 (3 Target Functions; 2 Decision Variables; Domain -2 <= x1, x2 <= 2)

# Viennet 1 - Target Function 1
def viennet_1_f1(variables_values = [0, 0]):
    f1 = variables_values[0]**2 + (variables_values[1] - 1)**2
    return f1

# Viennet 1 - Target Function 2
def viennet_1_f2(variables_values = [0, 0]):
    f2 = variables_values[0]**2 + (variables_values[1] + 1)**2 + 1
    return f2

# Viennet 1 - Target Function 3
def viennet_1_f3(variables_values = [0, 0]):
    f3 = (variables_values[0] - 1)**2 + variables_values[1]**2 + 2
    return f3

############################################################################

# Viennet 2 (3 Target Functions; 2 Decision Variables; Domain -4 <= x1, x2 <= 4)

# Viennet 2 - Target Function 1
def viennet_2_f1(variables_values = [0, 0]):
    f1 = (((variables_values[0] - 2)**2)/2) + (((variables_values[1] + 1)**2)/13) + 3
    return f1

# Viennet 2 - Target Function 2
def viennet_2_f2(variables_values = [0, 0]):
    f2 = (((variables_values[0] + variables_values[1] - 3)**2)/36) + (((-variables_values[0] + variables_values[1] + 2)**2)/8) - 17
    return f2

# Viennet 2 - Target Function 3
def viennet_2_f3(variables_values = [0, 0]):
    f3 = (((variables_values[0] + 2*variables_values[1] - 1)**2)/175) + (((-variables_values[0] + 2*variables_values[1])**2)/17) - 13
    return f3

############################################################################

# Viennet 3 (3 Target Functions; 2 Decision Variables; Domain -3 <= x1, x2 <= 3)

# Viennet 3 - Target Function 1
def viennet_3_f1(variables_values = [0, 0]):
    f1 = 0.5*(variables_values[0]**2 + variables_values[1]**2) + np.sin(variables_values[0]**2 + variables_values[1]**2)
    return f1

# Viennet 3 - Target Function 2
def viennet_3_f2(variables_values = [0, 0]):
    f2 = ((3*variables_values[0] - 2*variables_values[1] + 4)**2)/8 + ((variables_values[0] - variables_values[1] + 1)**2)/27 + 15
    return f2

# Viennet 3 - Target Function 3
def viennet_3_f3(variables_values = [0, 0]):
    f3 = 1/(variables_values[0]**2 + variables_values[1]**2 + 1) - 1.1*np.exp(-(variables_values[0]**2 + variables_values[1]**2))
    return f3

############################################################################
