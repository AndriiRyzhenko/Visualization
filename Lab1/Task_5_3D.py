import matplotlib.pyplot as plt
import numpy as np

Z = np.array([[76.4, 97.6, 122.2, 130.5, 153, 176, 200.5, 227, 247, 277],
              [45.7, 54.7, 58.7, 62.3, 67, 72, 77, 78.5, 79, 82],
              [40.8, 41.8, 42, 42, 42, 46, 50.5, 54, 56.5, 59],
              [44, 51.6, 63.2, 71.8, 83, 93, 104, 116.8, 123.5, 127],
              [123, 158, 171.5, 186.5, 205.5, 226.5, 247, 258.5, 290, 290]])

y_labels = np.array(['US', 'De', 'Fr', 'Jp', 'USSR'])
x_labels = np.array(['1900', '1913', '1929', '1938', '1950', '1960', '1970', '1980', '1990', '2000'])


# координати барів
xpos, ypos = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)


# розміри барів
dx = 0.75 * np.ones_like(zpos)
dy = dx.copy()
dz = Z.flatten()

#додано зміщення що б ось була по центру
ypos = np.add(ypos, -0.5 * dy)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')


# задаємо кожному бару свій колір
colors = ['blue', 'yellow', 'green', 'gray', 'red']
for i, country in enumerate(y_labels):
    start = i * len(x_labels)
    end = (i + 1) * len(x_labels)
    ax.bar3d(xpos[start:end], ypos[start:end], zpos[start:end], dx[start:end], dy[start:end], dz[start:end],
             color=colors[i])


ax.set_xticks(np.arange(len(x_labels)))
ax.set_yticks(np.arange(len(y_labels)))
ax.set_xticklabels(x_labels)
ax.set_yticklabels(y_labels)
ax.set_title('Чисельність населення, млн. чол.')
ax.set_ylabel('Country')
ax.set_xlabel('Years')
ax.set_zlabel('Billions $')

plt.show()