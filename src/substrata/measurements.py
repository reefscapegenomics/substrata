# Standard Library
import sys
import random
import copy

# Third-Party Libraries
import numpy as np
from scipy.spatial import KDTree
import cv2
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt  # TO REMOVE? (if you no longer need it, you can comment it out or remove it)
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed
from open3d import geometry, utility, cpu
import open3d as o3d
from scipy.signal import convolve2d

try:  # TO DO
    import alphashape
    from scipy.spatial import distance_matrix
    from alphashape import optimizealpha
    from shapely.geometry import Point
    from shapely.geometry import Polygon, MultiPolygon
    from matplotlib.patches import Polygon as MplPolygon
except ImportError:
    pass

# Local Modules
from unicorn import (
    annotations,
    cameras,
    pointclouds,
    visualizations,
    settings,
    transforms,
    utils,
)


def conduct_PCA(pcd, sort=True):
    """
    Calculate eigenvalues/eigenvectors for pointcloud
    """

    matrix = np.cov(pcd.points.T)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def best_fit_plane_normal(pcd) -> geometry.Vector:
    """Return a unit normal (Vector) of the PCA best-fit plane."""
    a, b, c, _ = conduct_PCA(pcd)[:4]
    return geometry.Vector([a, b, c, 0])  # promote to 4-vector


def conduct_xy_PCA(pcd, sort=True, visualize=False):
    """
    Calculate eigenvalues/eigenvectors for x/y plane only.
    Optionally print and plot the results.
    """
    xy_points = pcd.points[:, :2]
    mean_xy = np.mean(xy_points, axis=0)
    centered_xy = xy_points - mean_xy  # shape (N, 2)

    cov = np.cov(centered_xy, rowvar=False)  # shape (2, 2)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    if sort:
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

    if visualize:
        visualizations.plot_xy_pca(
            points=xy_points, mean=mean_xy, eig_vecs=eigenvectors, eig_vals=eigenvalues
        )

    return eigenvalues, eigenvectors
