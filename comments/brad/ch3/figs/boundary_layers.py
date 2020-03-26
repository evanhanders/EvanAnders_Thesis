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

scale_law = -1/3



Lz = np.exp(3/(1.5-1e-4)) - 1

cmap = 'viridis_r'

entropy_gradients = dict()
entropy_stds = dict()
rossby_profiles   = dict()
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
    entropy_gradients[key] = f['grad_s_tot'].value[-int(n_profiles/3)-1:-1,:]
    entropy_stds[key] = f['s_fluc_std'].value[-int(n_profiles/3)-1:-1,:]
    rossby_profiles[key]  = f['Rossby'].value[-int(n_profiles/3)-1:-1,:]
    z[key]  = f['z'].value
keys = rossby_profiles.keys()
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

print(maxminta)
        

rops = []
tas = []
for key in sorted_keys:
    rop, ta = key.split('_')
    rops.append(float(rop[3:]))
    tas.append(float(ta[2:]))
rops = np.array(rops)
tas = np.array(tas)
ras = rops**2 * tas**(3/4)

fig = plt.figure(figsize=(8.5, 3))

#COLUMN 2
norm = matplotlib.colors.Normalize(vmin=np.min(np.log10(maxminta['rop1.58'][0])), vmax=np.max(np.log10(maxminta['rop1.58'][1])))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

s_bls = []
ro_bls = []
s_bls_bot = []
ro_bls_bot = []
taylors = []
rayleighs = []
print('_________________')
for key in sorted_keys:
    rop, ta = key.split('_')
    rop = float(rop[3:])
    ta = float(ta[2:])

    if rop != 1.58:
        continue
    
    ra = rop**2 * ta**(3/4)
    print('rop {:.2f}, ra {:.4e}, ta {:.4e}'.format(rop,ra, ta))
    n_calcs = 0
    mean_s_bl = 0
    mean_ro_bl = 0

    s_std = np.mean(entropy_stds[key], axis=0)
    ro    = np.mean(rossby_profiles[key], axis=0)

    half_z = z[key][int(len(z[key])/2):]
    half_s = s_std[int(len(s_std)/2):]
    half_ro = ro[int(len(ro)/2):]
    ro = interp1d(half_z, half_ro, fill_value='extrapolate')
    s = interp1d(half_z, half_s, fill_value='extrapolate')
    big_z = np.linspace(Lz/2, Lz, 1e4)
    big_ro = ro(big_z)
    big_s  = s(big_z)


    ro_bl = Lz - big_z[np.argmax(big_ro)]
    max_place = big_z[np.argmax(big_s[:-100])]
    s_bl  = Lz - max_place
    mean_s_bl += s_bl
    mean_ro_bl += ro_bl
    n_calcs += 1

    ro_bls.append(mean_ro_bl/n_calcs)
    s_bls.append(mean_s_bl/n_calcs)

    taylors.append(ta)
    rayleighs.append(ra)

ro_bls = np.array(ro_bls)
s_bls = np.array(s_bls)
ro_bls_bot = np.array(ro_bls_bot)
s_bls_bot = np.array(s_bls_bot)
rayleighs = np.array(rayleighs)

s = rayleighs/(10**(1.39))
plt.plot(s, s_bls/s**(scale_law), lw=0, marker='o', c='orange' , label='Ro$_{\mathrm{p}}$ = 1.58', zorder=1)


#COLUMN 1
s_bls = []
ro_bls = []
s_bls_bot = []
ro_bls_bot = []
taylors = []
rayleighs = []
for key in sorted_keys:
    rop, ta = key.split('_')
    rop = float(rop[3:])
    ta = float(ta[2:])
    ra = rop**2 * ta**(3/4)

    if rop != 0.957:
        continue

    print('rop {:.2f}, ra {:.4e}, ta {:.4e}'.format(rop,ra, ta))
    n_calcs = 0
    mean_s_bl = 0
    mean_ro_bl = 0

    s_std = np.mean(entropy_stds[key], axis=0)
    ro    = np.mean(rossby_profiles[key], axis=0)

    half_z = z[key][int(len(z[key])/2):]
    half_s = s_std[int(len(s_std)/2):]
    half_ro = ro[int(len(ro)/2):]
    ro = interp1d(half_z, half_ro, fill_value='extrapolate')
    s = interp1d(half_z, half_s, fill_value='extrapolate')
    big_z = np.linspace(Lz/2, Lz, 1e4)
    big_ro = ro(big_z)
    big_s  = s(big_z)


    ro_bl = Lz - big_z[np.argmax(big_ro)]
    max_place = big_z[np.argmax(big_s[:-100])]
    s_bl  = Lz - max_place
    mean_s_bl += s_bl
    mean_ro_bl += ro_bl
    n_calcs += 1

    ro_bls.append(mean_ro_bl/n_calcs)
    s_bls.append(mean_s_bl/n_calcs)

    taylors.append(ta)
    rayleighs.append(ra)



  
ro_bls = np.array(ro_bls)
s_bls = np.array(s_bls)
ro_bls_bot = np.array(ro_bls_bot)
s_bls_bot = np.array(s_bls_bot)
rayleighs = np.array(rayleighs)
s = rayleighs/(10**(2.44))
plt.plot(s, s_bls/s**(scale_law), lw=0, marker='o', c='blue', label='Ro$_{\mathrm{p}}$ = 0.96', zorder=2)


 
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
    n_calcs = 0
    mean_s_bl = 0
    mean_ro_bl = 0

    s_std = np.mean(entropy_stds[key], axis=0)
    ro    = np.mean(rossby_profiles[key], axis=0)

    half_z = z[key][int(len(z[key])/2):]
    half_s = s_std[int(len(s_std)/2):]
    half_ro = ro[int(len(ro)/2):]
    ro = interp1d(half_z, half_ro, fill_value='extrapolate')
    s = interp1d(half_z, half_s, fill_value='extrapolate')
    big_z = np.linspace(Lz/2, Lz, 1e4)
    big_ro = ro(big_z)
    big_s  = s(big_z)


    ro_bl = Lz - big_z[np.argmax(big_ro)]
    max_place = big_z[np.argmax(big_s[:-100])]
    s_bl  = Lz - max_place
    mean_s_bl += s_bl
    mean_ro_bl += ro_bl
    n_calcs += 1

    ro_bls.append(mean_ro_bl/n_calcs)
    s_bls.append(mean_s_bl/n_calcs)

    taylors.append(ta)
    rayleighs.append(ra)

rayleighs = np.array(rayleighs)

  
ro_bls = np.array(ro_bls)
s_bls = np.array(s_bls)

s = rayleighs/(10**(4.88))
plt.plot(s, s_bls/s**(scale_law), lw=0, marker='o', c='green', label='Ro$_{\mathrm{p}}$ = 0.6', zorder=3)
plt.xscale('log')

plt.legend(loc='lower right')

plt.axhline(4,   c='blue',   lw=0.5)
plt.axhline(5.1, c='orange', lw=0.5)
plt.axhline(3.3, c='green',  lw=0.5)

plt.ylabel(r'$\delta_s/\mathcal{S}^{-1/3}$')
plt.xlabel(r'$\mathcal{S} \equiv$ Ra/Ra$_{\mathrm{crit}}(\mathrm{Ro}_{\mathrm{p}})$')

fig.savefig('ro_p_boundary_layers.png', dpi=300, bbox_inches='tight')
fig.savefig('ro_p_boundary_layers.pdf', dpi=300, bbox_inches='tight')
