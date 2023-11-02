# pyMultiobjective

## Introduction

A python library for the following Multiobjective Optimization Algorithms or Many Objectives Optimization Algorithms: **C-NSGA II** (Clustered Non-Dominated Sorting Genetic Algorithm II); **CTAEA** (Constrained Two Archive Evolutionary Algorithm); **GrEA** (Grid-based Evolutionary Algorithm); **HypE** (Hypervolume Estimation Multiobjective Optimization Algorithm); **IBEA-FC** (Indicator-Based Evolutionary Algorithm with Fast Comparison Indicator); **IBEA-HV** (Indicator-Based Evolutionary Algorithm with Hypervolume Indicator); **MOEA/D** (Multiobjective Evolutionary Algorithm Based on Decomposition); **NAEMO** (Neighborhood-sensitive Archived Evolutionary Many-objective Optimization); **NSGA II** (Non-Dominated Sorting Genetic Algorithm II);  **NSGA III** (Non-Dominated Sorting Genetic Algorithm III); **OMOPSO** (Optimized Multiobjective Particle Swarm Optimization); **PAES** (Pareto Archived Evolution Strategy) with Fast Non-Dominance Sorting); **RVEA** (Reference Vector Guided Evolutionary Algorithm); **SMPSO** (Speed-Constrained Multiobjective Particle Swarm Optimization); **SMS-EMOA** (S-Metric Selection Evolutionary Multiobjective Optimization Algorithm); **SPEA2** (Strength Pareto Evolutionary Algorithm 2);  **U-NSGA III** (Unified Non-Dominated Sorting Genetic Algorithm III).

## Usage

1. Install
```bash
pip install pyMultiobjective
```

2. Import
```py3

# Import NSGA III
from pyMultiobjective.algorithm import non_dominated_sorting_genetic_algorithm_III

# Import Test Functions. Available Test Functions: Dent, DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7, Fonseca-Fleming, Kursawe, Poloni, Schaffer1, Schaffer2, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, Viennet1, Viennet2, Viennet3 
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

# Import Performance Indicators. Available Performance Indicators: GD, GD+, IGD, IGD+, Maximum Spread, Spacing and Hypervolume
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
ms   = indicators.ms_indicator(list_of_functions = [dent_f1, dent_f2], **parameters)
sp   = indicators.sp_indicator(list_of_functions = [dent_f1, dent_f2], **parameters)

print('GD   = ', gd)
print('GDP  = ', gdp)
print('IGD  = ', igd)
print('IGDP = ', igdp)
print('MS   = ', ms)
print('SP   = ', sp)


parameters = {
	'solution': sol, 
	'n_objs': 2,
	'ref_point': [], # A Reference Point. If empty, an arbitrary Reference Point will be Used
}
hypervolume = indicators.hv_indicator(**parameters)
print('Hypervolume = ', hypervolume)

```

3. Try it in **Colab**
- C-NSGA II ([ Colab Demo ](https://colab.research.google.com/drive/1sXxCWV6dDmNXmes7RDka4OqKOtM0t9YX?usp=sharing)) ( [ Original Paper ](https://open.metu.edu.tr/bitstream/handle/11511/69040/12625931.pdf))
- CTAEA ([ Colab Demo ](https://colab.research.google.com/drive/1IC5m7JfmhT0ihWBhziQdfyq1PAHrmW1p?usp=sharing)) ( [ Original Paper ](https://doi.org/10.48550/arXiv.2103.06382))
- GrEA ([ Colab Demo ](https://colab.research.google.com/drive/1H2w77kCGUj33qI7uIE-e68999zy1L8tf?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1109/TEVC.2012.2227145))
- HypE ([ Colab Demo ](https://colab.research.google.com/drive/1cpIWZTECKfyf9jp_iiSuOJaWcWjy_NCr?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1162/EVCO_a_00009))
- IBEA-FC ([ Colab Demo ](https://colab.research.google.com/drive/1BBD0nWaE5SqL5n2Jpa_fDYgkWGSpy8xu?usp=sharing)) ( [ Original Paper ](https://www.simonkuenzli.ch/docs/ZK04.pdf))
- IBEA-HV ([ Colab Demo ](https://colab.research.google.com/drive/1XoiEAR3xpx0DbivrSp_QEFA32xm_R1lk?usp=sharing)) ( [ Original Paper ](https://www.simonkuenzli.ch/docs/ZK04.pdf))
- MOEA/D ([ Colab Demo ](https://colab.research.google.com/drive/1BP2qM9coiOTq28ZYeQEqxHSCHBeh3-Io?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1109/TEVC.2007.892759))
- NAEMO ([ Colab Demo ](https://colab.research.google.com/drive/1ctVjjOKhLQ1DqQJ0ozcvp2pClmbwBg8O?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.swevo.2018.12.002))
- NSGA II ([ Colab Demo ](https://colab.research.google.com/drive/1aD1uiJOCezCG6lotMAQENGas4abEO3_6?usp=sharing)) ( [ Original Paper ](http://dx.doi.org/10.1109/4235.996017))
- NSGA III ([ Colab Demo ](https://colab.research.google.com/drive/18zcEdU3NNplFiXAqH8g-oSrEhWB-uqQN?usp=sharing)) ( [ Original Paper ](http://dx.doi.org/10.1109/TEVC.2013.2281535))
- OMOPSO ([ Colab Demo ](https://colab.research.google.com/drive/1cvSZllLYhU6UvuFM7KgDvb1YaNLZVU32?usp=sharing)) ( [ Original Paper ](http://dx.doi.org/10.1007/978-3-540-31880-4_35))
- PAES ([ Colab Demo ](https://colab.research.google.com/drive/1iz5Q9CYiLpyYEKJzd0KwQrGrZykr49TX?usp=sharing))  ( [ Original Paper ](https://doi.org/10.1109/CEC.1999.781913))
- RVEA ([ Colab Demo ](https://colab.research.google.com/drive/1KYYAsMM52P6lxHRk5a9P8yrnRhwCgT5i?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1109/TEVC.2016.2519378))
- SMPSO ([ Colab Demo ](https://colab.research.google.com/drive/17m9AT9ORHvVqeqaRjBga1XCEuyG1EPzz?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1109/MCDM.2009.4938830))
- SMS-EMOA ([ Colab Demo ](https://colab.research.google.com/drive/1hCAW70vVRC-NXmkHPUdX_gK2aADyliQS?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.ejor.2006.08.008))
- SPEA2 ([ Colab Demo ](https://colab.research.google.com/drive/1OrxJxxAMSpKu_xSWc9UQlPOeM_mmVHmW?usp=sharing)) ( [ Original Paper ](https://kdd.cs.ksu.edu/Courses/CIS830/Handouts/P8.pdf))
- U-NSGA III ([ Colab Demo ](https://colab.research.google.com/drive/1-AO_S6OlqzbA54DlMFBDGEL-wHh9hayH?usp=sharing)) ( [ Original Paper ](https://www.egr.msu.edu/~kdeb/papers/c2014022.pdf))

4. Test Functions
- Dent ( [ Paper ](https://doi.org/10.1007/978-3-319-44003-3_12)) ( [Pareto Front](https://github.com/Valdecy/Datasets/raw/master/Pareto%20Front/Dent.txt) ) ( [ Plot ](https://colab.research.google.com/drive/1VTHJMmFUHw97tLu5jkTQCkupzj7VA2yp?usp=sharing)) 
- DTLZ1 ( [ Paper ](https://doi.org/10.1109/CEC.2002.1007032)) ( [Pareto Front](https://github.com/Valdecy/Datasets/raw/master/Pareto%20Front/DTLZ1.txt) ) ( [ Plot ](https://colab.research.google.com/drive/1ENkr2yDACfRwX1ZIidwC15T_YWJrDMtk?usp=sharing)) 
- DTLZ2 ( [ Paper ](https://doi.org/10.1109/CEC.2002.1007032)) ( [Pareto Front](https://github.com/Valdecy/Datasets/raw/master/Pareto%20Front/DTLZ2.txt) ) ( [ Plot ](https://colab.research.google.com/drive/1pmh6-4jWfQ2eXOzkUm2oydvKc5c0NALz?usp=sharing)) 
- DTLZ3 ( [ Paper ](https://doi.org/10.1109/CEC.2002.1007032)) ( [Pareto Front](https://github.com/Valdecy/Datasets/raw/master/Pareto%20Front/DTLZ3.txt) ) ( [ Plot ](https://colab.research.google.com/drive/10pmPlqgrkwAcjA15jWSimgIH5rTWraZN?usp=sharing)) 
- DTLZ4 ( [ Paper ](https://doi.org/10.1109/CEC.2002.1007032)) ( [Pareto Front](https://github.com/Valdecy/Datasets/raw/master/Pareto%20Front/DTLZ4.txt) ) ( [ Plot ](https://colab.research.google.com/drive/1PbGRqbLI-wFWg1Orvr0X7Reh5er0vNdW?usp=sharing)) 
- DTLZ5 ( [ Paper ](https://doi.org/10.1109/CEC.2002.1007032)) ( [Pareto Front](https://github.com/Valdecy/Datasets/raw/master/Pareto%20Front/DTLZ5.txt) ) ( [ Plot ](https://colab.research.google.com/drive/1mF6UR_yXBSC3E4vHbgfAHFxA5z3rVCWx?usp=sharing)) 
- DTLZ6 ( [ Paper ](https://doi.org/10.1109/CEC.2002.1007032)) ( [Pareto Front](https://github.com/Valdecy/Datasets/raw/master/Pareto%20Front/DTLZ6.txt) ) ( [ Plot ](https://colab.research.google.com/drive/1U4E__uPcb5zdztFr8EiQqdegn53Xq7oX?usp=sharing)) 
- DTLZ7 ( [ Paper ](https://doi.org/10.1109/CEC.2002.1007032)) ( [Pareto Front](https://github.com/Valdecy/Datasets/raw/master/Pareto%20Front/DTLZ7.txt) ) ( [ Plot ](https://colab.research.google.com/drive/1mNK9WXdYTKP8NWYACXBeTZwWml1M4woZ?usp=sharing)) 
- Fonseca-Fleming ( [ Paper ](https://doi.org/10.1162/evco.1995.3.1.1)) ( [Pareto Front](https://github.com/Valdecy/Datasets/raw/master/Pareto%20Front/Fonseca_Fleming.txt) ) ( [ Plot ](https://colab.research.google.com/drive/14LS3MNRwmgbq9ZqA6K7vfbBqMJE4hH4v?usp=sharing)) 
- Kursawe ( [ Paper ](https://doi.org/10.1007/BFb0029752)) ( [Pareto Front](https://github.com/Valdecy/Datasets/raw/master/Pareto%20Front/Kursawe.txt) ) ( [ Plot ](https://colab.research.google.com/drive/1bG15YQiRjVX9r4mtEpnt9D-IA2Cjx7Q_?usp=sharing)) 
- Poloni ( [ Paper ](https://www.researchgate.net/publication/243686783_Hybrid_GA_for_multi_objective_aerodynamic_shape_optimization)) ( [Pareto Front](https://github.com/Valdecy/Datasets/raw/master/Pareto%20Front/Poloni.txt) ) ( [ Plot ](https://colab.research.google.com/drive/1USBnNWf-UayqZERHq5PM6WWExcwOrOin?usp=sharing)) 
- Schaffer1 ( [ Paper ](https://www.researchgate.net/publication/236443691_Some_Experiments_in_Machine_Learning_Using_Vector_Evaluated_Genetic_Algorithms)) ( [Pareto Front](https://github.com/Valdecy/Datasets/raw/master/Pareto%20Front/Schaffer1.txt) ) ( [ Plot ](https://colab.research.google.com/drive/1XIpcbZOHJq1xneYiyxCeb-ZhK9QZ3-81?usp=sharing)) 
- Schaffer2 ( [ Paper ](https://www.researchgate.net/publication/236443691_Some_Experiments_in_Machine_Learning_Using_Vector_Evaluated_Genetic_Algorithms)) ( [Pareto Front](https://github.com/Valdecy/Datasets/raw/master/Pareto%20Front/Schaffer2.txt) ) ( [ Plot ](https://colab.research.google.com/drive/1X5-c9e5wAss9pZE6xOUGt-cVhZTeQ50K?usp=sharing)) 
- ZDT1 ( [ Paper ](https://doi.org/10.1162/106365600568202)) ( [Pareto Front](https://github.com/Valdecy/Datasets/raw/master/Pareto%20Front/ZDT1.txt) ) ( [ Plot ](https://colab.research.google.com/drive/1Cvzz5o1KWT9vNEyeirUa_pH8jsyOTp8d?usp=sharing)) 
- ZDT2 ( [ Paper ](https://doi.org/10.1162/106365600568202)) ( [Pareto Front](https://github.com/Valdecy/Datasets/raw/master/Pareto%20Front/ZDT2.txt) ) ( [ Plot ](https://colab.research.google.com/drive/1AvXi0fV7PEdcByDDaPCyHP1CyARf3hQV?usp=sharing)) 
- ZDT3 ( [ Paper ](https://doi.org/10.1162/106365600568202)) ( [Pareto Front](https://github.com/Valdecy/Datasets/raw/master/Pareto%20Front/ZDT3.txt) ) ( [ Plot ](https://colab.research.google.com/drive/1XTq9CRYKDu5KPKMAY7w9edF05wJHXhFB?usp=sharing)) 
- ZDT4 ( [ Paper ](https://doi.org/10.1162/106365600568202)) ( [Pareto Front](https://github.com/Valdecy/Datasets/raw/master/Pareto%20Front/ZDT4.txt) ) ( [ Plot ](https://colab.research.google.com/drive/1pEQxM18HD2ZnwU4E7hXY3IYIvq7QrJ0U?usp=sharing)) 
- ZDT6 ( [ Paper ](https://doi.org/10.1162/106365600568202)) ( [Pareto Front](https://github.com/Valdecy/Datasets/raw/master/Pareto%20Front/ZDT6.txt) ) ( [ Plot ](https://colab.research.google.com/drive/1xAG_8N-K-X83DNj24tWDNbb15q_KDi7V?usp=sharing))
- Viennet1 ( [ Paper ](https://doi.org/10.1080/00207729608929211)) ( [Pareto Front](https://github.com/Valdecy/Datasets/raw/master/Pareto%20Front/Viennet1.txt) ) ( [ Plot ](https://colab.research.google.com/drive/1JqtvAZw1Mh3VmEURZANtZBgYmd07j9vN?usp=sharing)) 
- Viennet2 ( [ Paper ](https://doi.org/10.1080/00207729608929211)) ( [Pareto Front](https://github.com/Valdecy/Datasets/raw/master/Pareto%20Front/Viennet2.txt) ) ( [ Plot ](https://colab.research.google.com/drive/1tUFYPj9A3herBXg4rSylfBCP2jpDXIm8?usp=sharing)) 
- Viennet3 ( [ Paper ](https://doi.org/10.1080/00207729608929211)) ( [Pareto Front](https://github.com/Valdecy/Datasets/raw/master/Pareto%20Front/Viennet3.txt) ) ( [ Plot ](https://colab.research.google.com/drive/1okdsDM01DTuvg6VG_hozU42_KqScBcz7?usp=sharing)) 

5. Peformance Indicators
- GD ( [ Paper ](https://apps.dtic.mil/sti/pdfs/ADA364478.pdf))
- GD+ ( [ Paper ](https://doi.org/10.1007/978-3-319-15892-1_8))
- IGD ( [ Paper ](https://doi.org/10.1007/978-3-540-24694-7_71))
- IGD+ ( [ Paper ](https://doi.org/10.1007/978-3-319-15892-1_8))
- Maximum Spread ( [ Paper ](https://doi.org/10.1162/106365600568202))
- Spacing ( [ Paper ](https://doi.org/10.1109/TEVC.2006.882428))
- Hypervolume ( [ Paper ](https://scholar.afit.edu/cgi/viewcontent.cgi?article=6130&context=etd))

# Single Objective Optimization
For Single Objective Optimization try [pyMetaheuristic](https://github.com/Valdecy/pyMetaheuristic)

# TSP (Travelling Salesman Problem)
For Travelling Salesman Problems try [pyCombinatorial](https://github.com/Valdecy/pyCombinatorial)

# Acknowledgement 
This section is dedicated to all the people that helped to improve or correct the code. Thank you very much!

* Wei Chen (07.AUGUST.2019) - AFRL Summer Intern/Rising Senior at Stony Brook University.
