from matplotlib import ticker, cm
import numpy as np
import matplotlib.pyplot as plt
import h5py

from scipy.optimize import brentq
from scipy.interpolate import interp1d


f = h5py.File('FC_poly_onsets_eps1e-4_nrho3_Ta1e10_fixedTBC_2.5D.h5', 'r')

print(f.keys())

grid = f['grid'][()].real
ra   = f['xyz_0'][()]
kx   = f['xyz_1'][()]


kx_vals = np.unique(kx)
ra_crit = np.zeros_like(kx_vals)
for i, k in enumerate(kx_vals):
    ra_vals, growth = ra[kx == k], grid[kx == k]
    if growth.min() < 0 and growth.max() > 0:
        ra_crit[i] = brentq(interp1d(ra_vals, growth), ra_vals.min(), ra_vals.max())    
    else:
        continue
crit_found = ra_crit != 0

kx_c = kx_vals[crit_found][np.argmin(ra_crit[crit_found])]
ra_c = np.min(ra_crit[crit_found])
    

pos = grid > 0
neg = grid < 0

fig = plt.figure(figsize=(8, 4))
plt.pcolormesh(kx, ra, grid, cmap=cm.RdBu_r, vmin=-5e-3, vmax=5e-3)
plt.plot(kx_vals[crit_found], ra_crit[crit_found], c='k', lw=3)
plt.scatter(kx_c, ra_c, c='k', marker='*', s=200)
plt.axvline(kx_c/2, c='k')

Lx = 2*np.pi / (kx_c/2)
Lperp = np.sqrt(2)*Lx
kperp = 2*np.pi / Lperp
plt.axvline(kperp, c='k')
plt.xlim(3, 60)
plt.ylim(ra.min(), 8e7)

plt.text(kx_c/2.7, 7.6e7, r'Minimum simulation $k_\perp$', rotation=-90, fontsize=11)
plt.text(kx_c/1.9, 7.6e7, r'Minimum simulation $k_x$', rotation=-90, fontsize=11)

plt.xlabel(r'$k_\perp$')
plt.ylabel('Ra')

plt.savefig('crit_curve_ta1e10.png', dpi=300)
plt.savefig('crit_curve_ta1e10.pdf', dpi=300)
