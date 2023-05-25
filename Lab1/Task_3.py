import numpy as np
import matplotlib.pyplot as plt

alfa = 1.0
theta = np.linspace(0, 2*np.pi, 1000)
x = alfa * np.cos(theta) ** 3
y = alfa * np.sin(theta) ** 3

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta, x, linewidth=2, label='x = a cos^3 t')
ax.plot(theta, y, linewidth=1, label='x = a sin^3 t')

ax.set_rlabel_position(-22.5)
ax.grid(True)
ax.legend(loc='right')
plt.title('astroid')
plt.show()

theta2 = np.degrees(theta)

# об'єднати x, y, u в один масив
combined = np.array([theta2.flatten(), x.flatten(), y.flatten()])

# зберегти combined в txt формат
np.savetxt('Task_3.txt', combined.T)