import numpy as np
import hj_reachability as hj
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

plt.rcParams["text.usetex"] = False
plt.rcParams["mathtext.fontset"] = "cm"
font = {"size": 12}
plt.rc("font", **font)


class Zone:

    def __init__(self, x0: float, y0: float, R: float):

        """
        Creates a 2D landing.
        (x0, y0): bottom left corner of wall.
        """
        self.x0 = x0
        self.y0 = y0
        self.R = R


    def __call__(self, grid_states):
        return self.R - np.linalg.norm(grid_states[...,:] - np.array([self.x0, self.y0]), axis = -1)


class Pad:

    def __init__(self, x0: float, y0: float, x1: float, y1: float):

        """
        Creates a 2D landing.
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
        return -sdf

class Warf:

    def __init__(self):

        self.zones = []
        self.pads = []

    def zone_sdf(self, grid_states):

        sdf = np.inf * np.ones(grid_states.shape[:-1])
        for zone in self.zones:
            sdf = np.minimum(sdf, zone(grid_states))

        return sdf

    def pad_sdf(self, grid_states):
        sdf = -np.inf * np.ones(grid_states.shape[:-1])
        for pad in self.pads:
            sdf = np.maximum(sdf, pad(grid_states))

        return sdf

    def add_zone(self, zone):
        self.zones.append(zone)

    def add_pad(self, pad):
        self.pads.append(pad)

    def plot_value_functions(self, grid_states, ax, show_pads=True):

        """
        Visualizes the shelf (target) SDF in green and the wall (constraint)
        SDF in red as level sets over the grid, with the zero level set drawn
        bold and the interior of each object shaded.
        Set show_pads=False for problems in which the pads do not exist
        (e.g. the pure avoid tube).
        Returns the axis so callers can overlay trajectories, etc.
        """
        X = grid_states[..., 0]
        Y = grid_states[..., 1]
        zone_sdf = self.zone_sdf(grid_states)

        # zero level sets: shelf SDF is positive inside, wall SDF negative inside
        if show_pads:
            pad_sdf = self.pad_sdf(grid_states)
            ax.contour(X, Y, pad_sdf, levels=[0.0], colors="green", linewidths=2, label = 'landing pad')
            ax.contourf(X, Y, pad_sdf, levels=[0.0, np.inf], colors=["green"], alpha=0.2)
        ax.contour(X, Y, -zone_sdf, levels=[0.0], colors="red", linewidths=2, label = 'dead zone')
        ax.contourf(X, Y, -zone_sdf, levels=[0.0, np.inf], colors=["red"], alpha=0.2)
        ax.scatter([0.], [0.], color = 'orange', label = 'antenna')

        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_title("Drone environment")
        return ax
