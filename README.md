# pyMultiobjective

## Introduction

A python library for the following Multiobjective Optimization Algorithms or Many Objectives Optimization Algorithms: **C-NSGA II** (Clustered Non-Dominated Sorting Genetic Algorithm II); **CTAEA** (Constrained Two Archive Evolutionary Algorithm); **GrEA** (Grid-based Evolutionary Algorithm); **IBEA** (Indicator-Based Evolutionary Algorithm) with Fast Comparison; **MOEA/D** (Multiobjective Evolutionary Algorithm Based on Decomposition); **NAEMO** (Neighborhood-sensitive Archived Evolutionary Many-objective Optimization); **NSGA II** (Non-Dominated Sorting Genetic Algorithm II);  **NSGA III** (Non-Dominated Sorting Genetic Algorithm III); **OMOPSO** (Optimized Multiobjective Particle Swarm Optimization); **PAES** (Pareto Archived Evolution Strategy) with Fast Non-Dominance Sorting); **RVEA** (Reference Vector Guided Evolutionary Algorithm); **SMPSO** (Speed-Constrained Multiobjective Particle Swarm Optimization); **SPEA2** (Strength Pareto Evolutionary Algorithm 2);  **U-NSGA III** (Unified Non-Dominated Sorting Genetic Algorithm III).

1. Install
```bash
pip install pyMultiobjective
```

2. Import

```py3

# Import NSGA III
from pyMultiobjective.algorithm import non_dominated_sorting_genetic_algorithm_III

# Import Test Functions. Available Test Functions: Dent, DTLZ2, Fonseca-Fleming, Kursawe, Poloni, Schaffer1, Schaffer2, ZDT1, ZDT2, ZDT3, ZDT4, Viennet 
from pyMultiobjective.test_functions import dent_f1, dent_f2

# OR Define your Own Custom Function. The function input should be a list of values, 
# each value represents a dimenstion (x1, x2, ...xn) of the problem.

# Run NSGA III
parameters = {
	'references': 5,
	'min_values': (-5, -5),
	'max_values': (5, 5),
	'mutation_rate': 0.1,
	'generations': 1500,
	'mu': 1,
	'eta': 1,
	'k': 2, 
	'verbose': True
}
sol = non_dominated_sorting_genetic_algorithm_III(list_of_functions = [dent_f1, dent_f2], **parameters)

# Import Graphs
from pyMultiobjective.util import graphs

# Plot Solution - Scatter Plot
parameters = {
	'min_values': (-5, -5),
	'max_values': (5, 5),
	'step': (0.1, 0.1),
	'solution': sol, 
	'show_pf': True,
	'show_pts': True,
	'show_sol': True,
	'pf_min': True,  # True = Minimum Pareto Front; False = Maximum Pareto Front
	'custom_pf': [], # Input a custom Pareto Front(numpy array where each column is an Objective Function)
	'view': 'browser'
}
graphs.plot_mooa_function(list_of_functions = [dent_f1, dent_f2], **parameters)

# Plot Solution - Parallel Plot
parameters = {
	'min_values': (-5, -5), 
	'max_values': (5, 5), 
	'step': (0.1, 0.1), 
	'solution': sol, 
	'show_pf': True,
	'pf_min': True,  # True = Minimum Pareto Front; False = Maximum Pareto Front
	'custom_pf': [], # Input a custom Pareto Front(numpy array where each column is an Objective Function)
	'view': 'browser'
}
graphs.parallel_plot(list_of_functions = [dent_f1, dent_f2], **parameters)

# Plot Solution - Andrews Plot
parameters = {
	'min_values': (-5, -5), 
	'max_values': (5, 5), 
	'step': (0.1, 0.1), 
	'solution': sol, 
	'normalize': True,
	'size_x': 15,
	'size_y': 15,
	'show_pf': True, 
	'pf_min': True, # True = Minimum Pareto Front; False = Maximum Pareto Front
	'custom_pf': [] # Input a custom Pareto Front(numpy array where each column is an Objective Function)
}
graphs.andrews_plot(list_of_functions = [dent_f1, dent_f2], **parameters)

# Import Performance Indicators. Available Performance Indicators: GD, GD+, IGD, IGD+, Hypervolume
from pyMultiobjective.utils import indicators

parameters = {
	'min_values': (-5, -5), 
	'max_values': (5, 5), 
	'step': (0.1, 0.1), 
	'solution': sol, 
	'pf_min': True, # True = Minimum Pareto Front; False = Maximum Pareto Front
	'custom_pf': [] # Input a custom Pareto Front(numpy array where each column is an Objective Function)
}
gd   = indicators.gd_indicator(list_of_functions = [dent_f1, dent_f2], **parameters)
gdp  = indicators.gd_plus_indicator(list_of_functions = [dent_f1, dent_f2], **parameters)
igd  = indicators.igd_indicator(list_of_functions = [dent_f1, dent_f2], **parameters)
igdp = indicators.igd_plus_indicator(list_of_functions = [dent_f1, dent_f2], **parameters)

print('GD   = ', gd)
print('GDP  = ', gdp)
print('IGD  = ', igd)
print('IGDP = ', igdp)


parameters = {
	'solution': sol, 
	'n_objs': 2,
	'ref_point': [], # A Reference Point. If empty, an arbitrary Reference Point will be Used
}
hypervolume = indicators.hv_indicator(**parameters)
print('Hypervolume = ', hypervolume)

```

3. Colab Demo

Try it in **Colab**:

- C-NSGA II ([ Colab Demo ](https://colab.research.google.com/drive/1sXxCWV6dDmNXmes7RDka4OqKOtM0t9YX?usp=sharing)) ( [ Original Paper ]())
- CTAEA ([ Colab Demo ](https://colab.research.google.com/drive/1IC5m7JfmhT0ihWBhziQdfyq1PAHrmW1p?usp=sharing)) ( [ Original Paper ](https://doi.org/10.48550/arXiv.2103.06382))
- GrEA ([ Colab Demo ](https://colab.research.google.com/drive/1H2w77kCGUj33qI7uIE-e68999zy1L8tf?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1109/TEVC.2012.2227145))
- IBEA ([ Colab Demo ](https://colab.research.google.com/drive/1BBD0nWaE5SqL5n2Jpa_fDYgkWGSpy8xu?usp=sharing)) ( [ Original Paper ](https://www.simonkuenzli.ch/docs/ZK04.pdf))
- MOEA/D ([ Colab Demo ](https://colab.research.google.com/drive/1BP2qM9coiOTq28ZYeQEqxHSCHBeh3-Io?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1109/TEVC.2007.892759))
- NAEMO ([ Colab Demo ](https://colab.research.google.com/drive/1ctVjjOKhLQ1DqQJ0ozcvp2pClmbwBg8O?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.swevo.2018.12.002))
- NSGA II ([ Colab Demo ](https://colab.research.google.com/drive/1aD1uiJOCezCG6lotMAQENGas4abEO3_6?usp=sharing)) ( [ Original Paper ](http://dx.doi.org/10.1109/4235.996017))
- NSGA III ([ Colab Demo ](https://colab.research.google.com/drive/18zcEdU3NNplFiXAqH8g-oSrEhWB-uqQN?usp=sharing)) ( [ Original Paper ](http://dx.doi.org/10.1109/TEVC.2013.2281535))
- OMOPSO ([ Colab Demo ](https://colab.research.google.com/drive/1cvSZllLYhU6UvuFM7KgDvb1YaNLZVU32?usp=sharing)) ( [ Original Paper ](http://dx.doi.org/10.1007/978-3-540-31880-4_35))
- PAES ([ Colab Demo ](https://colab.research.google.com/drive/1iz5Q9CYiLpyYEKJzd0KwQrGrZykr49TX?usp=sharing))  ( [ Original Paper ](https://doi.org/10.1109/CEC.1999.781913))
- RVEA ([ Colab Demo ](https://colab.research.google.com/drive/1KYYAsMM52P6lxHRk5a9P8yrnRhwCgT5i?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1109/TEVC.2016.2519378))
- SMPSO ([ Colab Demo ](https://colab.research.google.com/drive/17m9AT9ORHvVqeqaRjBga1XCEuyG1EPzz?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1109/MCDM.2009.4938830))
- SPEA2 ([ Colab Demo ](https://colab.research.google.com/drive/1OrxJxxAMSpKu_xSWc9UQlPOeM_mmVHmW?usp=sharing)) ( [ Original Paper ](https://kdd.cs.ksu.edu/Courses/CIS830/Handouts/P8.pdf))
- U-NSGA III ([ Colab Demo ](https://colab.research.google.com/drive/1-AO_S6OlqzbA54DlMFBDGEL-wHh9hayH?usp=sharing)) ( [ Original Paper ](https://www.egr.msu.edu/~kdeb/papers/c2014022.pdf))


4. Test Functions

- Test Functions with various types of visualizations: Scatter (2D, 3D or ND), Parallel (2D, 3D or ND), Andrews (2D, 3D or ND), Radar (3D or ND) and Complex Radar Plots (3D or ND) ([ Colab Demo ](https://colab.research.google.com/drive/1ALVZp333yO6rPEcR0fhVQn-PJeH5PmGP?usp=sharing)) 

# Acknowledgement 
This section is dedicated to all the people that helped to improve or correct the code. Thank you very much!

* Wei Chen (07.AUGUST.2019) - AFRL Summer Intern/Rising Senior at Stony Brook University.
