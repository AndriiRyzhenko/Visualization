import matplotlib.pyplot as plt
import numpy as np

x, y, z = np.meshgrid(np.arange(-3, 4, 0.6),
                      np.arange(-3, 4, 0.6),
                      np.arange(-3, 4, 2.4))

Fx = (y * z) / (x ** 2 + y ** 2 + z ** 2)
Fy = (x * z) / (x ** 2 + y ** 2 + z ** 2)
Fz = (x * y) / (x ** 2 + y ** 2 + z ** 2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.quiver(x, y, z, Fx, Fy, Fz, length=0.5, normalize=True, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Тривимірна візуалізація векторного поля')

plt.show()