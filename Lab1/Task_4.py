import numpy as np
import matplotlib.pyplot as plt

# Визначення параметрів a, b та c - які константи
a = 1
b = 2
c = 3

# Створення масивів x, y та z
x = np.linspace(-a, a, 100)
y = np.linspace(-b, b, 100)
x, y = np.meshgrid(x, y)
z = c * np.sqrt((x**2/a**2) + (y**2/b**2) - 1)

# Створення 3D графіка
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# об'єднати x, y, u в один масив
combined = np.array([x.flatten(), y.flatten(), z.flatten()])

# зберегти combined в txt формат
np.savetxt('Task_4.txt', combined.T)