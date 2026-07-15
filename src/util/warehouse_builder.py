import numpy as np
import hj_reachability as hj
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

plt.rcParams["text.usetex"] = False
plt.rcParams["mathtext.fontset"] = "cm"
font = {"size": 12}
plt.rc("font", **font)

class Wall:

    def __init__(self, x0: float, y0: float, x1: float, y1: float):

        """
        Creates a 2D wall of the maze. Note that walls have thickness.
        (x0, y0): bottom left corner of wall.
        (x1, y1): top right corner of wall.
        """
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

        self.cx = (x0 + x1) / 2
        self.cy = (y0 + y1) / 2

        self.hx = (x1 - x0) / 2 
        self.hy = (y1 - y0) / 2


    def __call__(self, grid_states):
        X = grid_states[...,0]
        Y = grid_states[...,1]
        qx = np.abs(X - self.cx) - self.hx
        qy = np.abs(Y - self.cy) - self.hy

        sdf = np.sqrt(np.maximum(qx, 0.0 * qx)**2 + np.maximum(qy, 0.0 * qy)**2) + np.minimum(np.maximum(qx, qy), 0.0 * qx)
        return sdf / min(self.hx, self.hy)

class Shelf:

    def __init__(self, x0: float, y0: float, x1: float, y1: float):

        """
        Creates a 2D goal in the maze.
        (x0, y0): bottom left corner of wall.
        (x1, y1): top right corner of wall.
        """
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

        self.cx = (x0 + x1) / 2
        self.cy = (y0 + y1) / 2

        self.hx = (x1 - x0) / 2 
        self.hy = (y1 - y0) / 2


    def __call__(self, grid_states):
        X = grid_states[...,0]
        Y = grid_states[...,1]
        qx = np.abs(X - self.cx) - self.hx
        qy = np.abs(Y - self.cy) - self.hy

        sdf = np.sqrt(np.maximum(qx, 0.0 * qx)**2 + np.maximum(qy, 0.0 * qy)**2) + np.minimum(np.maximum(qx, qy), 0.0 * qx)
        return -sdf / min(self.hx, self.hy)

class Warehouse:

    def __init__(self):

        self.walls = []
        self.shelves = []

    def wall_sdf(self, grid_states):

        sdf = np.inf * np.ones(grid_states.shape[:-1])
        for wall in self.walls:
            sdf = np.minimum(sdf, wall(grid_states))

        return sdf

    def shelf_sdf(self, grid_states):
        sdf = -np.inf * np.ones(grid_states.shape[:-1])
        for shelf in self.shelves:
            sdf = np.maximum(sdf, shelf(grid_states))

        return sdf

    def add_wall(self, wall):
        self.walls.append(wall)

    def add_shelf(self, shelf):
        self.shelves.append(shelf)

    def plot_value_functions(self, grid_states, ax):

        """
        Visualizes the shelf (target) SDF in green and the wall (constraint)
        SDF in red as level sets over the grid, with the zero level set drawn
        bold and the interior of each object shaded.
        Returns the axis so callers can overlay trajectories, etc.
        """
        X = grid_states[..., 0]
        Y = grid_states[..., 1]
        wall_sdf = self.wall_sdf(grid_states)
        shelf_sdf = self.shelf_sdf(grid_states)

        # zero level sets: shelf SDF is positive inside, wall SDF negative inside
        ax.contour(X, Y, shelf_sdf, levels=[0.0], colors="green", linewidths=2)
        ax.contour(X, Y, wall_sdf, levels=[0.0], colors="red", linewidths=2)
        ax.contourf(X, Y, shelf_sdf, levels=[0.0, np.inf], colors=["green"], alpha=0.2)
        ax.contourf(X, Y, wall_sdf, levels=[-np.inf, 0.0], colors=["red"], alpha=0.2)

        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_{1,A}, x_{2,B}$")
        ax.set_ylabel("$x_{2,A}, x_{2,B}$")
        ax.set_title("Warehouse floor")
        return ax
