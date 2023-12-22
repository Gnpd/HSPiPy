from scipy.optimize import differential_evolution

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
        
        
        if ( (RED <= 1 and score == inside_limit) or (RED > 1 and score != inside_limit) ):
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
    bounds= [(1,30), (1, 30), (1, 30), (1, 20)]
    optimization_result = differential_evolution(func=minimization_function, bounds=bounds, tol=1e-8)
    return optimization_result 