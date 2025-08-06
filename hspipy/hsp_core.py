from scipy.optimize import differential_evolution

def split_grid(grid,inside_limit):
    '''Splits the solvent grid in inside and outside solvents taking into account the inside limit'''
    inside = []
    outside = []
    for solvent, D, P, H, score in grid:
        if score <= inside_limit:
            inside.append([solvent, D, P, H, score])
        if score > inside_limit:
            outside.append([solvent, D, P, H, score])
    return inside,outside

def get_hsp_error(HSP,grid,inside_limit):
    errors = []
    for solvent, D, P, H, score in grid:
        # Calculate the Euclidean distance between (D, P, H) and (HSP[0], HSP[1], HSP[2]).
        distance = (
            4*(D-HSP[0])**2+
            (P-HSP[1])**2+
            (H-HSP[2])**2)**0.5
        
        # Calculate the Relative Euclidean Distance (RED).
        RED = distance/HSP[3]
        
        
        if ( (RED <= 1 and score <= inside_limit) or (RED > 1 and score > inside_limit) ):
            errors.append(0)
        else:
            errors.append(abs(RED - 1))
            
    return sum(errors)/len(errors)

def build_minimization_function(grid,inside_limit):  
    def fun(array):
        return get_hsp_error(array, grid,inside_limit)
    return fun

def get_hsp(grid,inside_limit=1):
    minimization_function = build_minimization_function(grid,inside_limit)
    bounds= [(0,30), (0, 30), (0, 30), (0.1, 20)]
    optimization_result = differential_evolution(func=minimization_function, bounds=bounds, tol=1e-8)
    hsp = [optimization_result.x[0],optimization_result.x[1],optimization_result.x[2]]
    radius = optimization_result.x[3]
    error = 1-optimization_result.fun

    return hsp,radius,error