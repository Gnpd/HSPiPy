import numpy as np
from .core import HSPEstimator
from .readers import HSPDataReader
import matplotlib.pyplot as plt


def WireframeSphere(
    centre=[0.0, 0.0, 0.0], radius=1.0, n_meridians=40, n_circles_latitude=None
):
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
        n_circles_latitude = max(n_meridians // 2, 4)
    u, v = np.mgrid[
        0 : 2 * np.pi : n_meridians * 1j, 0 : np.pi : n_circles_latitude * 1j
    ]
    sphere_x = centre[0] + radius * np.cos(u) * np.sin(v)
    sphere_y = centre[1] + radius * np.sin(u) * np.sin(v)
    sphere_z = centre[2] + radius * np.cos(v)
    return sphere_x, sphere_y, sphere_z


def get_solvent_points(grid):
    """Extract H, D, P coordinates from grid data for plotting."""
    x = []
    y = []
    z = []
    for solvent, D, P, H, score in grid:
        x.append(H)
        y.append(D)
        z.append(P)
    return x, y, z

def split_grid(grid, inside_limit=1):
    """Split grid into inside and outside solvents based on score."""
    inside = []
    outside = []
    for solvent, D, P, H, score in grid:
        if score != 0 and score <= inside_limit:
            inside.append([solvent, D, P, H, score])
        else:
            outside.append([solvent, D, P, H, score])
    return inside, outside

class HSP(HSPEstimator):
    '''
    Hansen Solubility Parameters with plotting and convenience methods.
    
    This class extends HSPEstimator with data loading, plotting capabilities,
    and a simplified interface for typical HSP workflows. It maintains full
    sklearn compatibility while providing domain-specific functionality.
    '''
    def __init__(self, inside_limit=1, n_spheres=1):
        super().__init__(inside_limit=inside_limit, n_spheres=n_spheres)
        self._reader = HSPDataReader()
        self.grid = None
        self.inside = None
        self.outside = None

    def read(self, path):
        """
        Read HSP data from various file formats.
        
        Supports CSV, HSD, and HSDX formats with automatic format detection.
        
        Parameters
        ----------
        path : str or Path
            Path to the HSP data file
        """
        self.grid = self._reader.read(path)

    def get(self, inside_limit=1, n_spheres=1):
        """
        Fit HSP spheres to the loaded data and prepare for plotting.
        
        Parameters
        ----------
        inside_limit : float, optional
            Threshold for inside vs outside classification.
            If None, uses the value from initialization.
        n_spheres : int, optional
            Number of spheres to fit. If None, uses value from initialization.
        """
        if not hasattr(self, "grid") or self.grid is None:
            raise ValueError("No data loaded. Call read() first.")
        
        # Update parameters if provided
        if inside_limit is not None:
            self.inside_limit = inside_limit
        if n_spheres is not None:
            self.n_spheres = n_spheres
        
        # Convert grid to numpy array for fitting
        hsp_grid = self.grid[["Solvent", "D", "P", "H", "Score"]].to_numpy()

        # Fit the model using the grid
        X = hsp_grid[:, 1:4]  # D, P, H columns
        y = hsp_grid[:, 4]    # Score column
        self.fit(X, y)
        self.score(X, y)

        # Split grid for plotting
        self.inside, self.outside = split_grid(hsp_grid, self.inside_limit)

        # Handle single or multi-sphere
        if self.n_spheres == 1:
            hsp = np.array(self.hsp_[0][:3], dtype=float)
            radius = float(self.hsp_[0][3])
            self.d, self.p, self.h = hsp
            self.hsp = hsp
            self.radius = radius
        else:
            # For multi-sphere, store all centers/radii
            self.hsp = np.array([s[:3] for s in self.hsp_], dtype=float)
            self.radius = np.array([s[3] for s in self.hsp_], dtype=float)

        self.error = self.error_
        self.accuracy = self.accuracy_
        formatted_hsp = ["%.2f" % elem for elem in np.ravel(self.hsp)]
        formatted_radius = (
            ", ".join("%.3f" % r for r in np.ravel(self.radius))
            if isinstance(self.radius, (np.ndarray, list)) and len(np.ravel(self.radius)) > 1
            else "%.3f" % self.radius
        )
        print(
            "HSP: "
            + ", ".join(map(str, formatted_hsp))
            + "\n"
            + "Radius: "
            + str(formatted_radius)
            + "\n"
            + "error: "
            + "{:.2e}".format(self.error_)
            + "\n"
            + "accuracy: "
            + str("%.4f" % self.accuracy_)
        )
        return self.hsp, self.radius, self.error, self.accuracy  
           
    def plot_3d(self):
        """Create a 3D plot of the HSP space with spheres and solvents."""
        if not hasattr(self, 'hsp_') or self.hsp_ is None:
            raise ValueError("Model not fitted. Call get() first.")
        
        fig = plt.figure()
        fig.suptitle("3D HSP Plot")
        ax = plt.axes(projection="3d")
        ax.invert_yaxis()
        ax.set_xlabel("H")
        ax.set_ylabel("D")
        ax.set_zlabel("P")
        ax.zaxis.labelpad = -2

        # Draw spheres and centers
        if self.n_spheres == 1:
            # hsp_[0] = [D, P, H, radius], we need [H, D, P] for 3D plot
            center = np.array([self.hsp_[0][2], self.hsp_[0][0], self.hsp_[0][1]])  # [H, D, P]
            centers = [center]
            radii = [self.hsp_[0][3]]
        else:
            # Multiple spheres: convert [D, P, H, radius] to [H, D, P] for each sphere
            centers = []
            radii = []
            for sphere in self.hsp_:
                center = np.array([sphere[2], sphere[0], sphere[1]])  # [H, D, P]
                centers.append(center)
                radii.append(sphere[3])

        for center, radius in zip(centers, radii):
            x, y, z = WireframeSphere(center, radius)
            ax.plot_wireframe(x, y, z, color="g", linewidth=0.5)
            ax.scatter(center[0], center[1], center[2], color="g", s=50)

        # Draw solvent points
        good_x, good_y, good_z = get_solvent_points(self.inside)
        ax.scatter(good_x, good_y, good_z, color="b", s=50)
        bad_x, bad_y, bad_z = get_solvent_points(self.outside)
        ax.scatter(bad_x, bad_y, bad_z, color="r", s=50)

    def plot_2d(self):
        """Create 2D projections of the HSP space."""
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 3.5))
        fig.suptitle("2D HSP Subplots")
        good_x, good_y, good_z = get_solvent_points(self.inside)
        bad_x, bad_y, bad_z = get_solvent_points(self.outside)

        # Prepare centers and radii
        if self.n_spheres == 1:
            # hsp_[0] = [D, P, H, radius]
            d_center, p_center, h_center = self.hsp_[0][0], self.hsp_[0][1], self.hsp_[0][2]
            centers_d = [d_center]
            centers_p = [p_center]
            centers_h = [h_center]
            radii = [self.hsp_[0][3]]
        else:
            # Multiple spheres
            centers_d = [sphere[0] for sphere in self.hsp_]  # D values
            centers_p = [sphere[1] for sphere in self.hsp_]  # P values
            centers_h = [sphere[2] for sphere in self.hsp_]  # H values
            radii = [sphere[3] for sphere in self.hsp_]      # radii


       # P vs H
        ax1.scatter(good_z, good_x, color="b", label="Good solvents")
        ax1.scatter(bad_z, bad_x, color="r", label="Bad solvents")
        for p_center, h_center, radius in zip(centers_p, centers_h, radii):
            ax1.scatter(p_center, h_center, color="g", s=100)
            circle = plt.Circle((p_center, h_center), radius, color="g", fill=False)
            ax1.add_patch(circle)
        ax1.set_xlabel("P")
        ax1.set_ylabel("H")
        ax1.legend()

        # H vs D
        ax2.scatter(good_x, good_y, color="b", label="Good solvents")
        ax2.scatter(bad_x, bad_y, color="r", label="Bad solvents")
        for h_center, d_center, radius in zip(centers_h, centers_d, radii):
            ax2.scatter(h_center, d_center, color="g", s=100)
            circle = plt.Circle((h_center, d_center), radius, color="g", fill=False)
            ax2.add_patch(circle)
        ax2.set_xlabel("H")
        ax2.set_ylabel("D")
        ax2.legend()

        # P vs D
        ax3.scatter(good_z, good_y, color="b", label="Good solvents")
        ax3.scatter(bad_z, bad_y, color="r", label="Bad solvents")
        for p_center, d_center, radius in zip(centers_p, centers_d, radii):
            ax3.scatter(p_center, d_center, color="g", s=100)
            circle = plt.Circle((p_center, d_center), radius, color="g", fill=False)
            ax3.add_patch(circle)
        ax3.set_xlabel("P")
        ax3.set_ylabel("D")
        ax3.legend()

    def plots(self):
        """Show both 3D and 2D plots."""
        self.plot_3d()
        self.plot_2d()
