import pandas as pd
import numpy as np
from hspcore import get_hsp, split_grid
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def WireframeSphere(centre=[0.,0.,0.], radius=1.,
                    n_meridians=40, n_circles_latitude=None):
    """
    Create the arrays of values to plot the wireframe of a sphere.

    Parameters
    ----------
    centre: array like
        A point, defined as an iterable of three numerical values.
    radius: number
        The radius of the sphere.
    n_meridians: int
        The number of meridians to display (circles that pass on both poles).
    n_circles_latitude: int
        The number of horizontal circles (akin to the Equator) to display.
        Notice this includes one for each pole, and defaults to 4 or half
        of the *n_meridians* if the latter is larger.

    Returns
    -------
    sphere_x, sphere_y, sphere_z: arrays
        The arrays with the coordinates of the points to make the wireframe.
        Their shape is (n_meridians, n_circles_latitude).
    """
    if n_circles_latitude is None:
        n_circles_latitude = max(n_meridians/2, 4)
    u, v = np.mgrid[0:2*np.pi:n_meridians*1j, 0:np.pi:n_circles_latitude*1j]
    sphere_x = centre[0] + radius * np.cos(u) * np.sin(v)
    sphere_y = centre[1] + radius * np.sin(u) * np.sin(v)
    sphere_z = centre[2] + radius * np.cos(v)
    return sphere_x, sphere_y, sphere_z

def get_solvent_points(grid):
    x=[]
    y=[]
    z=[]
    for solvent, D, P, H, score in grid:
        x.append(H)
        y.append(D)
        z.append(P)
    return x,y,z

class HSP:
    def read(self,path):
        self.grid = pd.read_csv(path)
    def get(self,inside_limit=1):
        if hasattr(self, 'grid'):
            hsp_grid = self.grid[['Solvent','D','P','H','Score']].to_numpy()
            hsp,radius,error = get_hsp(hsp_grid,inside_limit)
            inside,outside = split_grid(hsp_grid,inside_limit)
            self.d = hsp[0]
            self.p = hsp[1]
            self.h = hsp[2]
            self.hsp=hsp
            self.radius = radius
            self.error = error
            self.inside = inside
            self.outside = outside
            formatted_hsp = [ '%.2f' % elem for elem in hsp ]
            print('HSP: ' + ', '.join(map(str, formatted_hsp)) + '\n'+
                  'Radius: ' + str('%.3f' % radius)+'\n'+
                  'error: ' + str('%.4f' % error))
            return
        print('Your HSP has no grid, you can use the read function to import an HSP grid from a file')
    def plot_3d(self):
        fig = plt.figure()
        fig.suptitle('3D HSP Plot')
        ax = plt.axes(projection='3d')
        ax.invert_yaxis()
        ax.set_xlabel('H')
        ax.set_ylabel('D')
        ax.set_zlabel('P')
        ax.zaxis.labelpad=-2 
        
        # draw sphere
        x,y,z = WireframeSphere([self.h,self.d,self.p],self.radius)
        #ax.contour3D(x, y, z, 50, cmap='binary')
        ax.plot_wireframe(x, y, z, color="g", linewidth=0.5)
        # draw a points
        ax.scatter([self.h], [self.d], [self.p], color="g", s=50)
        good_x,good_y,good_z = get_solvent_points(self.inside)
        ax.scatter(good_x, good_y, good_z, color="b", s=50)
        bad_x,bad_y,bad_z = get_solvent_points(self.outside)
        ax.scatter(bad_x, bad_y, bad_z, color="r", s=50)
    def plot_2d(self):
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3,figsize=(10,3))
        fig.suptitle('2D HSP Subplots')
        good_x,good_y,good_z = get_solvent_points(self.inside)
        bad_x,bad_y,bad_z = get_solvent_points(self.outside)
        
        # P vs H
        circle1 = plt.Circle((self.p, self.h), self.radius, color='g', fill = False)
        ax1.scatter(good_z, good_x, color="b")
        ax1.scatter(bad_z, bad_x,color="r")
        ax1.scatter(self.p, self.h,color="g")
        ax1.add_patch(circle1)
        # H vs D
        circle2 = plt.Circle((self.h,self.d), self.radius, color='g', fill = False)
        ax2.scatter(good_z, good_x, color="b")
        ax2.scatter(bad_z, bad_x,color="r")
        ax2.scatter(self.h, self.d,color="g")
        ax2.add_patch(circle2)
        # P vs D
        circle3 = plt.Circle((self.p,self.d), self.radius, color='g', fill = False)
        ax3.scatter(good_z, good_x, color="b")
        ax3.scatter(bad_z, bad_x,color="r")
        ax3.scatter(self.p, self.d,color="g")
        ax3.add_patch(circle3)
    def plots(self):
        self.plot_3d()
        self.plot_2d()

