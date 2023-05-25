from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def calc_u(X, Y):
    Z = np.zeros_like(X)

    mask1 = (np.abs(X) + np.abs(Y)) < 0.5
    Z[mask1] = X[mask1] - np.exp(2 * Y[mask1])

    mask2 = (np.abs(X) + np.abs(Y)) >= 0.5
    Z[mask2] = 2 * X[mask2] ** 2 - np.exp(Y[mask2])

    mask3 = (np.abs(X) + np.abs(Y)) >= 1
    Z[mask3] = np.exp(5 * X[mask3] - 3) - Y[mask3]

    return Z


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

X = np.arange(-4, 4, 0.3)
Y = np.arange(-4, 4, 0.3)
X, Y = np.meshgrid(X, Y)
U = calc_u(X, Y)

np.savetxt('Task_2.txt', np.c_[X.ravel(), Y.ravel(), U.ravel()], header='X Y Z')

surf = ax.plot_surface(X, Y, U, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('U')

ax.zaxis.set_major_locator(LinearLocator(10))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
