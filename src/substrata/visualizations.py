   # visualizations.py
   import matplotlib.pyplot as plt
   from numpy.typing import NDArray
   import numpy as np

   def plot_xy_pca(points: NDArray, mean: NDArray,
                   eig_vecs: NDArray, eig_vals: NDArray) -> None:
       """Scatter the points and show the first two eigen-vectors."""
       plt.figure(figsize=(6, 6))
       plt.scatter(points[:, 0], points[:, 1], s=5, alpha=0.4)
       plt.plot(mean[0], mean[1], "ro")
       scale = 2.0 * np.sqrt(eig_vals)
       colors = ["r", "g"]
       for i in range(2):
           dx, dy = scale[i] * eig_vecs[:, i]
           plt.arrow(mean[0], mean[1], dx, dy,
                     width=0.01, color=colors[i],
                     length_includes_head=True)
       plt.gca().set_aspect("equal")
       plt.xlabel("X"); plt.ylabel("Y"); plt.title("XY PCA"); plt.show()