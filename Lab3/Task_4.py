import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
import glyph_visualization_lib as gvl

x = np.linspace(0, np.pi, 8, dtype=float, endpoint=True)
y = np.linspace(-np.pi, -1, 8, dtype=float, endpoint=True)
z = np.linspace(0, 2 * np.pi, 8, dtype=float, endpoint=True)
X, Y, Z = np.meshgrid(x, y, z)

stress_tensor = np.array(([np.sin(X), X + Y, X + Z],
                          [X + Y, np.cos(Y), Y + Z],
                          [X + Z, Y + Z, np.cos(Z)]))

print(stress_tensor.shape)
vm_stress = gvl.get_von_Mises_stress(stress_tensor)
glyph_radius = 0.25
limits = [np.min(vm_stress), np.max(vm_stress)]
colormap = plt.get_cmap('rainbow', 120)
fig = mlab.figure(bgcolor=(1, 1, 1))
fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
for i in range(x.size):
    for j in range(y.size):
        for k in range(z.size):
            center = [x[i], y[j], z[k]]
            data = stress_tensor[:, :, i, j, k]
            color = colormap(gvl.get_colormap_ratio_on_stress(vm_stress[i, j, k], limits))[:3]

            x_g, y_g, z_g = gvl.get_glyph_data(center, data, limits, glyph_points=12, glyph_radius=glyph_radius,
                                               glyph_type=2)

            mlab.mesh(x_g, y_g, z_g, color=color)

mlab.move(forward=1.8)
mlab.savefig("superquadric-Kindlmann_modified-viz.png", size=(100, 100))
mlab.show()
