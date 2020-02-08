import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

images = glob.glob('*.png')

for fname in images:
    stem = fname.split('.png')[0]

    img = mpimg.imread(fname)
    size = img.shape[:2]
    print(stem, size)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(size[1]/300, size[0]/300)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, aspect='auto')


    fig.savefig('./{:s}.pdf'.format(stem), dpi=300, pad_inches=0)
