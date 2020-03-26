import numpy as np
import matplotlib.pyplot as plt

data = np.array(\
    (   (10,    3e-3,   0.042,  20),
        (10,    1e-3,   0.039,  9.6),
        (10,    3e-4,   0.12,   6.8),
        (20,    1e-2,   0.16,   20),
        (20,    3e-3,   0.13,   20),
        (20,    1e-3,   0.097,  20),
        (20,    8e-4,   0.088,  4.6),
        (20,    5e-4,   0.079,  2.3),
        (40,    1e-2,   0.34,   20),
        (40,    3e-3,   0.26,   20),
        (40,    1e-3,   0.19,   3.3),
        (80,    1e-2,   0.61,   20),
        (80,    3e-3,   0.45,   20),
        (80,    1e-3,   0.34,   10),
        (80,    3e-4,   0.26,   0.41)
    ))

fig = plt.figure(figsize=(7.5, 3))
ax1 = fig.add_subplot(1,2,1)


for rop, c, m in ((10, 'green', 's'), (20, 'orange', 'o'), (40, 'blue', 'd'), (80, 'black', '+')):
    good = (data[:,0] == rop)*(data[:,-1] > 5)
    plt.scatter(data[:,1][good], data[:,2][good], color=c, marker=m, label='Ro$_p^2$ = {}'.format(rop))
    if rop > 10:
        deltaRo = np.log10(data[:,2][good][-1]) - np.log10(data[:,2][good][0])
        deltaEk = np.log10(data[:,1][good][-1]) - np.log10(data[:,1][good][0])
        print(deltaRo, deltaEk, deltaRo/deltaEk)


plt.legend(loc='best', fontsize=8)
plt.xscale('log')
plt.yscale('log')
plt.xlim(1.5e-2, 1e-4)
plt.ylabel('Ro')
plt.xlabel('Ek')


ax2 = fig.add_subplot(1,2,2)
power = 0.22
for rop, c, m in ((10, 'green', 's'), (20, 'orange', 'o'), (40, 'blue', 'd'), (80, 'black', '+')):
    good = (data[:,0] == rop)*(data[:,-1] > 5)
    norm = data[:,2][good][data[:,1][good] == 3e-3]
    plt.scatter(data[:,1][good], data[:,2][good]/data[:,1][good]**(power)/norm, color=c, marker=m)


plt.xscale('log')
plt.xlim(1.5e-2, 3e-4)
plt.ylabel(r'Ro/[Ek$^{0.22}\cdot$Ro(Ek$= 3\times10^{-3}$)]')
plt.xlabel('Ek')
plt.ylim(3, 4)

ax2.yaxis.tick_right()
ax2.yaxis.set_label_position('right')


plt.savefig('mdwarfs_rop.png', dpi=300, bbox_inches='tight')
plt.savefig('mdwarfs_rop.pdf', dpi=300, bbox_inches='tight')
