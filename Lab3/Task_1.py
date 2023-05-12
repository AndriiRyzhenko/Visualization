import matplotlib.pyplot as plt
import numpy as np

def func_z(x, y):
    return x * np.sqrt(y) + y * np.sqrt(x)

x = np.linspace(0, 5, 300)
y = np.linspace(0, 5, 300)

X, Y = np.meshgrid(x, y)

Z = func_z(X, Y)

plt.pcolormesh(X, Y, Z)

plt.title('Скалярне поле')
plt.show()

