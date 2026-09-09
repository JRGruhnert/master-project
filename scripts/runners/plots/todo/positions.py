import matplotlib.pyplot as plt
import numpy as np
from src.plotting.helper import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


class Positions3DPlot:
    def __init__(self):
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection="3d")

    def print_object(self, pos: list[float], quat: list[float], color):
        x, y, z = pos
        # Convert quaternion to rotation matrix
        r = R.from_quat(quat)
        # Direction vector (e.g., x-axis of the object)
        direction = r.apply([1, 0, 0])
        # Plot position
        self.ax.scatter(x, y, z, color=color)  # type: ignore
        # Plot direction as arrow
        self.ax.quiver(
            x, y, z, direction[0], direction[1], direction[2], length=1, color="r"
        )

    def connect_objects(self, pos1: list[float], pos2: list[float], color, alpha):
        x_values = [pos1[0], pos2[0]]
        y_values = [pos1[1], pos2[1]]
        z_values = [pos1[2], pos2[2]]
        self.ax.plot(
            x_values,
            y_values,
            z_values,
            color=color,
            linestyle="--",
            alpha=alpha,
        )

    def make_ellipsoid(self, pos: list[float], color, alpha=0.2):
        # mean [x, y, z]
        # var [var_x, var_y, var_z]
        mean = np.mean(pos, axis=0)
        var = np.var(pos, axis=0)
        # Standard deviations (axis lengths)
        rx, ry, rz = np.sqrt(var)
        # Create a grid for the ellipsoid
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = rx * np.outer(np.cos(u), np.sin(v)) + mean[0]
        y = ry * np.outer(np.sin(u), np.sin(v)) + mean[1]
        z = rz * np.outer(np.ones_like(u), np.cos(v)) + mean[2]
        self.ax.plot_surface(x, y, z, color=color, alpha=alpha)

    def create(self, show: bool, save: bool):
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        if show:
            plt.show()
        if save:
            plt.savefig("plot.png")
