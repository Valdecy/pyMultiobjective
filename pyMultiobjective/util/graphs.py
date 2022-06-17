############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Multivariate Plots

# Citation: 
# PEREIRA, V. (2022). GitHub repository: <https://github.com/Valdecy/pyMultiojective>

############################################################################

# Required Libraries
import itertools
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go

############################################################################

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

# Function: Solution Plot 
def plot_mooa_function(min_values = [-5, -5], max_values = [5, 5], list_of_functions = [], step = [0.1, 0.1], solution = [ ], view = 'browser', show_pf = True, show_pts = True, show_sol = True, pf_min = True, custom_pf = []):
    if (view == 'browser' ):
        pio.renderers.default = 'browser'
    data  = []
    if (show_pts == True or (show_pf == True and len(custom_pf) == 0)):
        front = generate_points(min_values, max_values, list_of_functions, step, pf_min)
        if (show_pf == True and len(custom_pf) == 0):
            pf_idx = pareto_front_points(pts = front[:,len(min_values):], pf_min = pf_min)
            pf     = front[pf_idx, :]
    if (show_pf == True and len(custom_pf) > 0):
        pf = np.copy(custom_pf)
    if  (len(list_of_functions) == 2):
        if (show_pts == True):
            n_trace = go.Scatter(x         = front[:, -2],
                                 y         = front[:, -1],
                                 opacity   = 0.5,
                                 mode      = 'markers',
                                 marker    = dict(symbol = 'circle-dot', size = 2, color = 'blue'),
                                 name      = ''
                                 )
            data.append(n_trace)
        if (show_pf == True):
            p_trace = go.Scatter(x         = pf[:, -2],
                                 y         = pf[:, -1],
                                 opacity   = 1,
                                 mode      = 'markers',
                                 marker    = dict(symbol = 'circle-dot', size = 4, color = 'red'),
                                 name      = ''
                                 )
            data.append(p_trace)
        if (len(solution) > 0 and show_sol == True):
            s_trace = go.Scatter(x         = solution[:, -2],
                                 y         = solution[:, -1],
                                 opacity   = 1,
                                 mode      = 'markers',
                                 marker    = dict(symbol = 'circle-dot', size = 8, color = 'black'),
                                 name      = ''
                                 )
            data.append(s_trace)
        layout  = go.Layout(showlegend   = False,
                            hovermode    = 'closest',
                            margin       = dict(b = 10, l = 5, r = 5, t = 10),
                            plot_bgcolor = 'rgb(235, 235, 235)',
                            xaxis        = dict(  showgrid       = True, 
                                                  zeroline       = True, 
                                                  showticklabels = True, 
                                                  tickmode       = 'array', 
                                               ),
                            yaxis        = dict(  showgrid       = True, 
                                                  zeroline       = True, 
                                                  showticklabels = True,
                                                  tickmode       = 'array', 
                                                )
                            )
        fig = go.Figure(data = data, layout = layout)
        fig.update_traces(textfont_size = 10, textfont_color = 'rgb(235, 235, 235)') 
        fig.show() 
    elif (len(list_of_functions) == 3):
        if (show_pts == True):
            n_trace = go.Scatter3d(x         = front[:, -3],
                                   y         = front[:, -2],
                                   z         = front[:, -1],
                                   opacity   = 0.5,
                                   mode      = 'markers',
                                   marker    = dict(size = 2, color = 'blue'),
                                   name      = ''
                                   )
            data.append(n_trace)
        if (show_pf == True ):
            p_trace = go.Scatter3d(x         = pf[:, -3],
                                   y         = pf[:, -2],
                                   z         = pf[:, -1],
                                   opacity   = 1,
                                   mode      = 'markers',
                                   marker    = dict(size = 4, color = 'red'),
                                   name      = ''
                                   )
            data.append(p_trace)
        if (len(solution) > 0 and show_sol == True):
            s_trace = go.Scatter3d(x         = solution[:, -3],
                                   y         = solution[:, -2],
                                   z         = solution[:, -1],
                                   opacity   = 1,
                                   mode      = 'markers',
                                   marker    = dict(size = 8, color = 'black'),
                                   name      = ''
                                   )
            data.append(s_trace)
        layout  = go.Layout(showlegend   = False,
                            hovermode    = 'closest',
                            margin       = dict(b = 10, l = 5, r = 5, t = 10),
                            plot_bgcolor = 'white',
                            )
        fig = go.Figure(data = data, layout = layout)
        fig.update_traces(textfont_size = 10, textfont_color = 'rgb(235, 235, 235)') 
        fig.update_scenes(xaxis_visible = True, yaxis_visible = True, zaxis_visible = True)
        fig.show() 
    elif (len(list_of_functions) > 3):
        dim_lst = []
        color_c = []
        color   = []
        val     = []
        if (show_pts == True):
            val = np.copy(front)
            color_c.append(('blue', front.shape[0]))
        if (show_pf == True ):
            if (len(val) == 0):
                val = np.copy(pf)
            else:
                val = np.r_[val, pf]
            color_c.append(('red', front[pf,:].shape[0]))
        if (len(solution) > 0 and show_sol == True):
            if (len(val) == 0):
                val = np.copy(solution)
            else:
                val = np.r_[val, solution]
            color_c.append(('black', solution.shape[0]))
        for tup in color_c:
            c, q = tup
            for _ in range(0, q):
                color.append(c)
        for i in range(0, len(list_of_functions)):
            d = dict(label = 'f'+str(i+1), values = val[:, -(i+1)])
            dim_lst.append(d)
        c_trace = go.Splom(dimensions = dim_lst,
                           marker     = dict(color = color, size = 5, line = dict(width = 0.5, color = 'rgb(230,230,230)')),
                           diagonal   = dict(visible = False)
                           )
        data.append(c_trace)
        layout  = go.Layout(hovermode = 'closest',
                            margin    = dict(b = 10, l = 5, r = 5, t = 10)
                            )
        fig = go.Figure(data = data, layout = layout)
        fig.show() 
    return

############################################################################

# Function: Parallel Plot
def parallel_plot(solution = [], show_pf = True, min_values = [], max_values = [], list_of_functions = [], step = [], pf_min = True, custom_pf = [], view = 'browser'):
    if (view == 'browser' ):
        pio.renderers.default = 'browser'
    names = ['f'+str(i+1) for i in range(0, len(list_of_functions))]
    if (show_pf == True):
        if (len(custom_pf) == 0):
            data_ = generate_points(min_values, max_values, list_of_functions, step, pf_min)
            data_ = data_[:, len(min_values):]
            pf    = pareto_front_points(pts = data_, pf_min = True)
            data_ = data_[pf, :]
        else:
            data_ = np.copy(custom_pf)
    if (show_pf == True and len(solution) > 0):
        cluster  = np.r_[np.zeros(data_.shape[0]), np.ones(solution.shape[0])]
        solution = solution[:, len(min_values):]
        data_    = np.r_[data_, solution]
    elif (show_pf == True and len(solution) == 0):
        cluster  = np.zeros(data_.shape[0], dtype = 'int8')
    elif (show_pf == False and len(solution) > 0):
        cluster  = np.ones(solution.shape[0], dtype = 'int8')
        data_    = solution[:, len(min_values):]
    data_  = pd.DataFrame(data_, columns = names)
    dims   = []
    traces = []
    for i in range(0, len(names)):
        dims.append(dict(range = [data_[names[i]].min()*1.00, data_[names[i]].max()*1.00], label = names[i], values = data_[names[i]]))
    if (show_pf == True and len(solution) > 0):
        n_trace = go.Parcoords(line       = dict(color = cluster, colorscale =  [ [0, 'rgba(245, 0, 0, 0.5)'], [1, 'rgba(0, 0, 0, 0.5)'] ]),          
                               dimensions = dims
                          )
    elif (show_pf == True and len(solution) == 0):
        n_trace = go.Parcoords(line       = dict(color = 'rgba(245, 0, 0, 0.5)'),          
                               dimensions = dims
                          )
    elif (show_pf == False and len(solution) > 0):
        n_trace = go.Parcoords(line       = dict(color = 'rgba(0, 0, 0, 0.5)'),          
                               dimensions = dims
                          )
    traces.append(n_trace)
    par_plot = go.Figure(data = traces)
    par_plot.update_layout(font = dict(family = 'Arial Black', size = 15, color = 'black'))
    par_plot.show()
    return

# Function: Andrews Plot
def andrews_plot(solution = [], normalize = True, size_x = 15, size_y = 15, show_pf = True, min_values = [], max_values = [], list_of_functions = [], step = [], pf_min = True, custom_pf = []):
    names = ['f'+str(i+1) for i in range(0, len(list_of_functions))]
    if (show_pf == True):
        if (len(custom_pf) == 0):
            data_ = generate_points(min_values, max_values, list_of_functions, step, pf_min)
            data_ = data_[:, len(min_values):]
            pf    = pareto_front_points(pts = data_, pf_min = True)
            data_ = data_[pf, :]
        else:
            data_ = np.copy(custom_pf)
    if (show_pf == True and len(solution) > 0):
        cluster  = np.r_[np.zeros(data_.shape[0]), np.ones(solution.shape[0])]
        solution = solution[:, len(min_values):]
        data_    = np.r_[data_, solution]
    elif (show_pf == True and len(solution) == 0):
        cluster  = np.zeros(data_.shape[0], dtype = 'int8')
    elif (show_pf == False and len(solution) > 0):
        cluster  = np.ones(solution.shape[0], dtype = 'int8')
        data_    = solution[:, len(min_values):]
    if (normalize == True):
        z_min  = np.min(data_, axis = 0)
        z_max  = np.max(data_, axis = 0)
        f      = (data_ - z_min)/(z_max - z_min)
    else:
        f      = data_
    df = pd.DataFrame(f, columns = names)
    if (len(cluster) == 0 or len(cluster) < df.shape[0]):
        df['T'] = 0
    else:
        df['T'] = cluster
    fig = plt.figure(figsize = (size_x, size_y))
    ax  = fig.add_axes( [.05, .05, .9, .9])
    if (show_pf == True and len(solution) > 0):
        pd.plotting.andrews_curves(df, 'T', color = ['#f50000', '#000000'])
    elif (show_pf == True and len(solution) == 0):
        pd.plotting.andrews_curves(df, 'T', color = '#f50000')
    elif (show_pf == False and len(solution) > 0):
        pd.plotting.andrews_curves(df, 'T', color = '#000000')
    ax.get_legend().remove()
    return

# Function: Radial Plot
def radial_plot(solution = [], size_x = 15, size_y = 15, show_pf = True, min_values = [], max_values = [], list_of_functions = [], step = [], pf_min = True, custom_pf = []):   
    names = ['f'+str(i+1) for i in range(0, len(list_of_functions))]
    if (show_pf == True):
        if (len(custom_pf) == 0):
            data_ = generate_points(min_values, max_values, list_of_functions, step, pf_min)
            data_ = data_[:, len(min_values):]
            pf    = pareto_front_points(pts = data_, pf_min = True)
            data_ = data_[pf, :]
        else:
            data_ = np.copy(custom_pf)
    if (show_pf == True and len(solution) > 0):
        cluster  = np.r_[np.zeros(data_.shape[0]), np.ones(solution.shape[0])]
        solution = solution[:, len(min_values):]
        data_    = np.r_[data_, solution]
    elif (show_pf == True and len(solution) == 0):
        cluster  = np.zeros(data_.shape[0], dtype = 'int8')
    elif (show_pf == False and len(solution) > 0):
        cluster  = np.ones(solution.shape[0], dtype = 'int8')
        data_    = solution[:, len(min_values):]
    dim    = data_.shape[1]
    radius = 1 
    points = []
    angles = np.arange(0, 360, 360/dim)
    xs     = radius*np.cos(angles*np.pi/180) + radius
    ys     = radius*np.sin(angles*np.pi/180) + radius
    points = np.column_stack((xs, ys))
    data_  = pd.DataFrame(data_, columns = names)
    if (len(cluster) == 0 or len(cluster) < data_.shape[0]):
        data_['T'] = 0
    else:
        data_['T'] = cluster
    fig = plt.figure(figsize = (size_x, size_y))
    ax  = fig.add_axes( [.05, .05, .9, .9])
    for i in range(0, points.shape[0]):
        ax.plot([points[i,0] - 1 , 0], [points[i,1] - 1 , 0], color = 'b', linewidth = 0.5, linestyle = 'dashed')
    if (show_pf == True and len(solution) > 0):
        pd.plotting.radviz(data_, 'T', ax = ax, color = ['#f50000', '#000000']) 
    elif (show_pf == True and len(solution) == 0):
        pd.plotting.radviz(data_, 'T', ax = ax, color = '#f50000') 
    elif (show_pf == False and len(solution) > 0):
        pd.plotting.radviz(data_, 'T', ax = ax, color = '#000000') 
    if (len(cluster) == 0 or len(cluster) < data_.shape[0]):
        ax.get_legend().remove()
    ax.add_patch(plt.Circle((0, 0), radius, color = 'k', fill = None))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.get_legend().remove()
    return

# Function: Complex Radar Plot
def complex_radar_plot(solution = [], idx = [], size_x = 15, size_y = 15, show_pf = True, min_values = [], max_values = [], list_of_functions = [], step = [], pf_min = True, custom_pf = []):
    names = ['f'+str(i+1) for i in range(0, len(list_of_functions))]
    cycol = itertools.cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
                             '#bcbd22', '#17becf', '#bf77f6', '#ff9408', '#d1ffbd', '#c85a53', '#3a18b1', '#ff796c', 
                             '#04d8b2', '#ffb07c', '#aaa662', '#0485d1', '#fffe7a', '#b0dd16', '#d85679', '#12e193', 
                             '#82cafc', '#ac9362', '#f8481c', '#c292a1', '#c0fa8b', '#ca7b80', '#f4d054', '#fbdd7e', 
                             '#ffff7e', '#cd7584', '#f9bc08', '#c7c10c'])
    def scale_data(data, ranges):
        def invert(x, limits):
            return limits[1] - (x - limits[0])
        x1, x2 = ranges[0]
        d      = data[0]
        if x1 > x2:
            d      = invert(d, (x1, x2))
            x1, x2 = x2, x1
        sdata = [d]
        for d, (y1, y2) in zip(data[1:], ranges[1:]):
            if y1 > y2:
                d      = invert(d, (y1, y2))
                y1, y2 = y2, y1
            sdata.append((d-y1) / (y2-y1) * (x2 - x1) + x1)
        return sdata
    if (show_pf == True):
        if (len(custom_pf) == 0):
            data_ = generate_points(min_values, max_values, list_of_functions, step, pf_min)
            data_ = data_[:, len(min_values):]
            pf    = pareto_front_points(pts = data_, pf_min = True)
            data_ = data_[pf, :]
        else:
            data_ = np.copy(custom_pf)
    if (show_pf == True and len(solution) > 0):
        solution = solution[:, len(min_values):]
        r_       = np.r_[data_, solution]
    elif (show_pf == True and len(solution) == 0):
        r_       = np.copy(data_)
    elif (show_pf == False and len(solution) > 0):
        solution = solution[:, len(min_values):]
        r_       = np.copy(solution)
    ranges = []
    for j in range(0, r_.shape[1]):
        ranges.append((np.min(r_[:,j]), np.max(r_[:,j])))
    angles  = np.arange(0, 360, 360./r_.shape[1])
    fig     = plt.figure(figsize = (size_x, size_y))
    axes    = [fig.add_axes([0.1, 0.1, 0.9, 0.9], polar = True, label = 'axes{}'.format(i)) for i in range(0, r_.shape[1])]
    _, text = axes[0].set_thetagrids(angles, labels = names)
    axes[0].xaxis.set_tick_params(pad = 20)
    for ax in axes[1:]:
        ax.patch.set_visible(False)
        ax.grid('off')
        ax.xaxis.set_visible(False)
    for i, ax in enumerate(axes):
        grid      = np.linspace(*ranges[i], num = 7)
        gridlabel = ['{}'.format(round(x, 2)) for x in grid]
        if ranges[i][0] > ranges[i][1]:
            grid     = grid[::-1] 
        gridlabel[0] = ''
        ax.set_rgrids(grid, labels = gridlabel, angle = angles[i])
        ax.set_ylim(*ranges[i])
    angle  = np.deg2rad(np.r_[angles, angles[0]])
    ranges = ranges
    ax     = axes[0]
    if (show_pf == True):
        for i in range(0, data_.shape[0]):
            sdata = scale_data(data_[i,:], ranges)
            ax.fill(angle, np.r_[sdata, sdata[0]], color = '#ed7e7e', alpha = 0.05)
    if (len(idx) > 0 and len(solution) > 0):
        for i in idx:
            sdata = scale_data(solution[i,:], ranges)
            ax.plot(angle, np.r_[sdata, sdata[0]], color = '#000000', alpha = 1, linewidth = 1)
            ax.fill(angle, np.r_[sdata, sdata[0]], color = next(cycol),  alpha = 0.45)
    return

############################################################################

