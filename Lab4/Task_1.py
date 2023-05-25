import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
from matplotlib.patches import Rectangle

fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7, 6.5)

ax = plt.axes(xlim=(-5, 15), ylim=(-5, 15))
patch = Rectangle((0, 0), 1, 0.75, fc='r')

def init():
    patch.set_xy([4, 4.625])
    ax.add_patch(patch)
    return patch,

def animate(i):
    x = 5 + 3 * np.sin(np.radians(i))
    y = 5 + 5 * np.cos(np.radians(i))
    patch.set_xy([x, y])
    return patch,

anim = animation.FuncAnimation(fig, animate,
                               init_func=init,
                               frames=360,
                               interval=20,
                               blit=True)

writer = PillowWriter(fps=240)
anim.save("animation.gif", writer=writer)

plt.show()
