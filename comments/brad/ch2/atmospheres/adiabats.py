from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

import dedalus.public as de

from scipy.special import erf

def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x-x0)/width))/2

def zero_to_one(*args, **kwargs):
    return - (one_to_zero(*args, **kwargs) - 1)

def find_rho(domain, rho0, T0, T0z, mass, m, tol=1e-10, n_rho0=3, epsilon=1e-4, gamma=5./3):
    problem = de.NLBVP(domain, variables=['rho', 'ln_rho', 'M'])
    problem.parameters['T']      = T0
    problem.parameters['Tz']     = T0z
    problem.parameters['tot_M']  = mass
    problem.parameters['g']      = 1 + m

    problem.add_equation('dz(M) - rho = 0')
    problem.add_equation('ln_rho = log(rho)')
    problem.add_equation('dz(ln_rho) = -(Tz + g)/T')
    
    problem.add_bc(' left(M) = 0')
    problem.add_bc('right(M) = tot_M')

    solver = problem.build_solver()
    rho    = solver.state['rho']
    ln_rho = solver.state['ln_rho']
    M      = solver.state['M']
    rho0.antidifferentiate('z', ('left', 0), out=M)
    ln_rho['g'] = np.copy(np.log(rho0['g']))
    rho['g']    = np.copy(rho0['g'])

    pert = solver.perturbations.data
    pert.fill(1 + tol)
    while np.sum(np.abs(pert)) > tol:
        solver.newton_iteration()

    ln_rho_bot = np.mean(ln_rho.interpolate(z=0)['g'])
    ln_rho_top = np.mean(ln_rho.interpolate(z=Lz)['g'])
    n_rho = ln_rho_bot - ln_rho_top

    d_nrho = n_rho - n_rho0
    dS_0   = -epsilon*n_rho0 * (gamma-1) / m / gamma
    delta_dS = (gamma-1)*d_nrho / gamma
    dS     = dS_0 + delta_dS
    print('epsilon {} / BL {} / Delta n_rho: {} / dS_0 {}, delta_dS {}, dS/dS_0 {}'.format(epsilon, bl_thickness, d_nrho, dS_0, delta_dS, dS/dS_0))
    
    return rho['g'], ln_rho['g'], d_nrho, dS/dS_0


n_rho0   = 3
gamma   = 5/3
m_ad    = 1/(gamma-1)
Cp      = gamma/(gamma-1)

upper_profiles = OrderedDict()
lower_profiles = OrderedDict()
z_profs        = OrderedDict()

for epsilon in [0.5, 1e-1, 1e-2, 1e-3, 1e-4, 1e-7]:

    m       = m_ad - epsilon
    Lz      = np.exp(n_rho0/m) - 1

    z_basis = de.Chebyshev('z', 256, [0, Lz], dealias=1)
    domain  = de.Domain([z_basis,], grid_dtype=np.float64)


    z = domain.grid(-1)
    z_profs['eps{:.4e}'.format(epsilon)] = np.copy(z)

    T0 = domain.new_field()
    rho0 = domain.new_field()
    T_ad = domain.new_field()
    rho_ad = domain.new_field()

    T0['g'] = 1 + Lz - z
    rho0['g'] = T0['g']**m
    mass = np.mean(rho0.integrate('z')['g'])


    T_ad['g'] = 1 + Lz + (-1 + epsilon/Cp)*z
    rho_ad['g'] = T_ad['g']**m_ad
    rho_ad_bot =np.mean(rho_ad.interpolate(z=0)['g'])
    rho_ad['g'] /= rho_ad_bot
    rho_ad['g'] *= np.mean(rho0.interpolate(z=0)['g'])
    mass_ad2 = np.mean(rho_ad.integrate('z')['g'])

    Tz = domain.new_field()
    T  = domain.new_field()

    print('UPPER BL')
    for bl_thickness in [0.2, 0.02]:
        d_BL = Lz*bl_thickness
        T_bot_BL = np.mean(T_ad.interpolate(z=Lz*(1-bl_thickness))['g'])
        dTdz_BL  = (1 - T_bot_BL)/d_BL
        delta_dTdz = dTdz_BL - (-1 + epsilon/Cp)

        Tz['g'] = (-1 + epsilon/Cp) + delta_dTdz*zero_to_one(z, Lz*(1 - bl_thickness), width=bl_thickness*Lz/5)
        Tz.antidifferentiate('z', ('left', 1 + Lz), out=T)

        rho, ln_rho, d_nrho, dS_frac = find_rho(domain, rho0, T, Tz, mass, m, n_rho0=n_rho0, epsilon=epsilon, gamma=gamma)
        lower_profiles['eps{:.4e}_bl{}'.format(epsilon, bl_thickness)] = (np.copy(T0['g']), np.copy(T['g']), rho, np.copy(np.log(rho0['g'])), ln_rho, d_nrho, dS_frac)

    T_ad['g'] = 1 + (-1 + epsilon/Cp)*(z - Lz)
    rho_ad['g'] = T_ad['g']**m_ad
    mass_ad1 = np.mean(rho_ad.integrate('z')['g'])

    print('LOWER BL')
    for bl_thickness in [0.2, 0.02]:
        d_BL = Lz*bl_thickness
        T_top_BL = np.mean(T_ad.interpolate(z=Lz*bl_thickness)['g'])
        dTdz_BL  = (T_top_BL - (1 + Lz))/d_BL
        delta_dTdz = dTdz_BL - (-1 + epsilon/Cp)

        Tz['g'] = (-1 + epsilon/Cp) + delta_dTdz*one_to_zero(z, Lz*bl_thickness, width=bl_thickness*Lz/5)
        Tz.antidifferentiate('z', ('right', 1), out=T)

        rho, ln_rho, d_nrho, dS_frac = find_rho(domain, rho0, T, Tz, mass, m, n_rho0=n_rho0, epsilon=epsilon, gamma=gamma)
        upper_profiles['eps{:.4e}_bl{}'.format(epsilon, bl_thickness)] = (np.copy(T0['g']), np.copy(T['g']), rho, np.copy(np.log(rho0['g'])), ln_rho, d_nrho, dS_frac)

epsilons = []
bls      = []
uppers   = []
lowers   = []
for k in lower_profiles.keys():
    epsilons.append(float(k.split('eps')[-1].split('_')[0]))
    bls.append(float(k.split('bl')[-1]))
    uppers.append(upper_profiles[k][-2:])
    lowers.append(lower_profiles[k][-2:])
epsilons = np.array(epsilons)
bls    = np.array(bls)
uppers = np.array(uppers)
lowers = np.array(lowers)

fig = plt.figure(figsize=(8, 4.5))
ax1a = fig.add_subplot(2,2,1)

z = z_profs['eps5.0000e-01']

k = 'eps5.0000e-01_bl0.2'
plt.plot(z, upper_profiles[k][0], c='k', lw=1.5)
plt.plot(z, upper_profiles[k][1], c='indigo', lw=0.75)
plt.plot(z, lower_profiles[k][1], c='orange', lw=0.75)

k = 'eps5.0000e-01_bl0.02'
plt.plot(z, upper_profiles[k][1], c='indigo', lw=0.75)
plt.plot(z, lower_profiles[k][1], c='orange', lw=0.75)

plt.xlabel('z')
plt.ylabel('T')
plt.xlim(z.min(), z.max())

ax1b = fig.add_subplot(2,2,3)
k = 'eps5.0000e-01_bl0.2'
plt.plot(z, upper_profiles[k][3], c='k', lw=1.5)
plt.plot(z, upper_profiles[k][4], c='indigo', lw=0.75)
plt.plot(z, lower_profiles[k][4], c='orange', lw=0.75)

k = 'eps5.0000e-01_bl0.02'
plt.plot(z, upper_profiles[k][4], c='indigo', lw=0.75)
plt.plot(z, lower_profiles[k][4], c='orange', lw=0.75)


plt.xlim(z.min(), z.max())
plt.xlabel('z')
plt.ylabel(r'$\ln(\rho)$')


ax2 = fig.add_subplot(2,2,2)

for bl, marker in [(0.2, '+'), (0.02, 'x')]:
    plt.plot(epsilons[bls == bl],  uppers[bls == bl,0], c='indigo', label='BL = {}'.format(bl),    lw=0, marker=marker)
    plt.plot(epsilons[bls == bl], -lowers[bls == bl,0], c='orange', lw=0, marker=marker)
plt.yscale('log')
plt.xscale('log')
plt.legend(loc='best')
plt.ylabel(r'$|\Delta n_\rho|$', labelpad=-1)
plt.xlabel(r'$\epsilon$')

ax3 = fig.add_subplot(2,2,4)

for bl, marker in [(0.2, '+'), (0.02, 'x')]:
    plt.plot(epsilons[bls == bl], uppers[bls == bl,1], c='indigo', label='BL = {}'.format(bl),    lw=0, marker=marker)
    plt.plot(epsilons[bls == bl], lowers[bls == bl,1], c='orange', lw=0, marker=marker)
plt.xscale('log')

plt.ylabel(r'$\Delta s(t) / \Delta s_0$')
plt.xlabel(r'$\epsilon$')

plt.savefig('limiting_adiabats.png', dpi=300, bbox_inches='tight')
plt.savefig('limiting_adiabats.pdf', dpi=300, bbox_inches='tight')
