import numpy as np
import hj_reachability as hj
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

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


    def __call__(self, grid):
        X = grid.states[...,0]
        Y = grid.states[...,1]
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


    def __call__(self, grid):
        X = grid.states[...,0]
        Y = grid.states[...,1]
        qx = np.abs(X - self.cx) - self.hx
        qy = np.abs(Y - self.cy) - self.hy

        sdf = np.sqrt(np.maximum(qx, 0.0 * qx)**2 + np.maximum(qy, 0.0 * qy)**2) + np.minimum(np.maximum(qx, qy), 0.0 * qx)
        return -sdf / min(self.hx, self.hy)

class Warehouse:

    def __init__(self):

        self.walls = []
        self.shelves = []

    def constraint_sdf(self, grid):

        sdf = np.inf * np.ones(grid.shape[:-1])
        for wall in self.walls:
            sdf = np.minimum(sdf, wall(grid))

        return sdf

    def target_sdf(self, grid):
        sdf = -np.inf * np.ones(grid.shape[:-1])
        for shelfs in self.shelves:
            sdf = np.maximum(sdf, shelves(grid))

        return sdf

    def add_wall(self, wall):
        self.walls.append(wall)

    def add_shelf(self, shelf):
        self.shelves.append(shelf)
