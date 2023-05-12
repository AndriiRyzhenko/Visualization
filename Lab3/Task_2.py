import numpy as np
import matplotlib.pyplot as plt

def u(x, y):
    return x ** 2 + 2 * y

def v(x, y):
    return y ** 2 + 2 * x

xx, yy = np.meshgrid(np.linspace(-4, 4, 10), np.linspace(-4, 4, 10))
u_val = u(xx, yy)
v_val = v(xx, yy)

plt.quiver(xx, yy, u_val, v_val)
plt.title('Векторне поле (вектори)')
plt.show()

fig, ax = plt.subplots()
ax.set_aspect('equal', 'box')
ax.streamplot(xx, yy, u_val, v_val, color=u_val, cmap='viridis')
plt.title('Вектор поле (лінії потоку)')
plt.show()