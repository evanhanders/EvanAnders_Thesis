import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig = plt.figure(frameon=False)
fig.set_size_inches(6.875, 3.4375)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

img = mpimg.imread('./snapshots_fig.png')
ax.imshow(img, rasterized=True, aspect='auto')


fig.savefig('./snapshots_fig_raster.png', dpi=300, pad_inches=0)
