import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
#matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
#matplotlib settings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import h5py
import scipy.optimize as scop
from scipy.optimize import fmin
from scipy.interpolate import interp1d

ORANGE   = [1.        , 0.5, 0]
GREEN    = [0, 0.398, 0]
BLUE     = [0, 0.5, 1]
kwargs = {'lw' : 0, 'ms' : 3, 'markeredgewidth' : 0.5}


Lz = np.exp(3/(1.5-1e-4)) - 1

cmap = 'viridis_r'
norm = matplotlib.colors.Normalize(vmin=5, vmax=9)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

rho   = dict()
z   = dict()
lines = []
for fname in glob.glob('../profile_data/*.h5'):
    key = ''
    pieces = fname.split('.h5')[0].split('/')[-1].split('_')
    ind_num = -2
    for s in pieces:
        if 'rop' in s:
            key += s + '_'
        if 'ta' in s:
            key += s
    f = h5py.File(fname, 'r')
    for k in f.keys():
        print(k)
    n_profiles = f['grad_s_tot'].value.shape[0]
    print(n_profiles)
    rho[key] = np.mean(np.log(f['rho_full'].value[-4:-2,:]) - (1.5-1e-4)*np.log(1+Lz-f['z'].value), axis=0)
    z[key]  = f['z'].value
keys = rho.keys()
ta = [-float(k.split('_ta')[-1]) for k in keys]
tas, sorted_keys = zip(*sorted(zip(ta, keys)))
tas = -np.array(tas)

maxminta = dict()
for ta, k in zip(tas, sorted_keys):
    key = k.split('_')[0]
    if key in maxminta.keys():
        if ta < maxminta[key][0]:
            maxminta[key][0] = ta
        if ta > maxminta[key][1]:
            maxminta[key][1] = ta
    else:
        maxminta[key] = [ta, ta]


rops = []
tas = []
for key in sorted_keys:
    rop, ta = key.split('_')
    rops.append(float(rop[3:]))
    tas.append(float(ta[2:]))
rops = np.array(rops)
tas = np.array(tas)
ras = rops**2 * tas**(3/4)

fig = plt.figure(figsize=(8.5, 2.5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

zs = np.linspace(0, Lz, 100)
rho0 = (1 + Lz - zs)**(1.5-1e-4)
ax1.plot(zs, np.log(rho0), c='k', lw=3, label=r'$\ln\rho_0$')
ax1.legend(loc='best')

#low rop
s_bls = []
ro_bls = []
tylors = []
rayleighs = []
for key in sorted_keys:
    rop, ta = key.split('_')
    rop = float(rop[3:])
    ta = float(ta[2:])
    ra = rop**2 * ta**(3/4)

    if rop != 0.6:
        continue

    print('rop {:.2f}, ra {:.4e}, ta {:.4e}'.format(rop,ra, ta))
    rho0 = (1 + Lz - z[key])**(1.5-1e-4)
    ax1.plot(z[key], np.log(rho0) + rho[key], c=sm.to_rgba(np.log10(ra)))
    ax2.plot(z[key], rho[key], c=sm.to_rgba(np.log10(ra)))

ax1.set_xlim(0, Lz)
ax2.set_xlim(0, Lz)

ax2.yaxis.tick_right()
ax2.yaxis.set_label_position('right')


ax1.set_ylabel(r'$\ln\rho$')
ax1.set_xlabel(r'$z$')

ax2.set_ylabel(r'$\ln\rho_1$')
ax2.set_xlabel(r'$z$')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label(r'log$_{10}$(Ra)')




fig.savefig('rot_density.png', dpi=300, bbox_inches='tight')
fig.savefig('rot_density.pdf', dpi=300, bbox_inches='tight')
