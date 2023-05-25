import matplotlib.pyplot as plt
import numpy as np

def func_z(x):
    y = np.zeros_like(x)
    mask = x <= 0
    y[mask] = (1 + 5 * x[mask] ** 2 - np.sin(x[mask]) ** 2) ** (1 / 2)
    y[~mask] = (7+x[~mask]) ** 2 / (4 + np.exp(-0.7 * x[~mask])) ** (1 / 3)
    return y

def func_y(x):
    return (4 + x ** 2 * np.exp(-3 * x)) / (4 + np.sqrt(x**4 + np.sin(x) ** 2))


x = np.arange(0, 2, 0.01)
y = func_y(x)
z = func_z(x)

fig_y, plt_y = plt.subplots()
plt_y.plot(x, y)
plt_y.grid(True, linestyle='-.')
plt_y.set_xlabel('X')
plt_y.set_ylabel('Y')
plt_y.set_title('Y-func')
plt_y.tick_params(labelcolor='r', labelsize='medium', width=4)

fig_z, plt_z = plt.subplots()
plt_z.plot(x, z)
plt_z.grid(True, linestyle='-.')
plt_z.set_xlabel('X')
plt_z.set_ylabel('Y')
plt_z.set_title('Z-func')

plt_z.tick_params(labelcolor='r', labelsize='medium', width=4)
plt.show()

data = np.column_stack((x, y))
np.savetxt('Task_1.1.txt', data, delimiter='\t', header='X\tY')

data = np.column_stack((x, z))
np.savetxt('Task_1.2.txt', data, delimiter='\t', header='X\tZ')
