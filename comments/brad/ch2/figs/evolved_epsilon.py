import numpy as np
import matplotlib.pyplot as plt

dfile = 'simulation_inputs_and_outputs.csv'
data = np.genfromtxt(dfile, delimiter=',', usecols=(0, 1, 2, 14), skip_header=1)
n_rho_init    = 3
n_rho_evolved = data[:,-1]
epsilon       = data[:,0]
ra            = data[:,1]
threeD        = data[:,2]
gamma = 5./3


m = 1/(gamma-1) - epsilon
Lz = np.exp(n_rho_init/m) - 1
initial_deltaS_over_cp = -(gamma-1) * epsilon / gamma * np.log(1 + Lz)
evolved_deltaS_over_cp = -(1/gamma)*n_rho_init/m + (gamma-1)/gamma * n_rho_evolved

print(evolved_deltaS_over_cp/initial_deltaS_over_cp)

eps_vals = [1e-4, 0.5, 1]
colors   = ['green', 'indigo', 'red']

print(n_rho_evolved[epsilon == 1e-7])

fig = plt.figure(figsize=(8, 3))
ax1 = fig.add_subplot(1,2,1)

plt.axhline(1, c='k')

for i, eps in enumerate(eps_vals):
    good = (epsilon == eps)*(threeD == 0)
    plt.plot(ra[good], np.abs(evolved_deltaS_over_cp[good]/initial_deltaS_over_cp[good]), label=r'$\epsilon={}$'.format(eps), lw=0, marker='o', c=colors[i])
    good = (epsilon == eps)*(threeD == 1)
    plt.plot(ra[good], np.abs(evolved_deltaS_over_cp[good]/initial_deltaS_over_cp[good]),  lw=0, marker='o', markersize=10, alpha=0.3, c=colors[i])

plt.xscale('log')
plt.xlabel('Ra')
plt.ylabel(r'Evolved $\Delta S$ / (Initial $\Delta S$)')

ax2 = fig.add_subplot(1,2,2)

for i, eps in enumerate(eps_vals):
    good = (epsilon == eps)*(threeD == 0)
    plt.plot(ra[good], ra[good]*np.abs(evolved_deltaS_over_cp[good]/initial_deltaS_over_cp[good]), label=r'$\epsilon={}$'.format(eps), lw=0, marker='o', c=colors[i])
    good = (epsilon == eps)*(threeD == 1)
    plt.plot(ra[good], ra[good]*np.abs(evolved_deltaS_over_cp[good]/initial_deltaS_over_cp[good]),  lw=0, marker='o', markersize=10, alpha=0.3, c=colors[i])

ras = np.logspace(1, 8, 100)
plt.plot(ras, ras, c='k')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Ra$_{\mathrm{in}}$')
plt.ylabel(r'Ra$_{\mathrm{out}}$')

plt.savefig('delta_S_vs_ra.png', dpi=300, bbox_inches='tight')
plt.savefig('delta_S_vs_ra.pdf', dpi=300, bbox_inches='tight')
