import numpy as np
import matplotlib.pyplot as plt

import dedalus.public as de

from scipy.special import erf

def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x-x0)/width))/2

def zero_to_one(*args, **kwargs):
    return - (one_to_zero(*args, **kwargs) - 1)

n_rho0   = 3
gamma   = 5/3
m_ad    = 1/(gamma-1)
Cp      = gamma/(gamma-1)

for epsilon in [0.5, 1e-1, 1e-2, 1e-3, 1e-4, 1e-7]:

    m       = m_ad - epsilon
    Lz      = np.exp(n_rho0/m) - 1

    z_basis = de.Chebyshev('z', 256, [0, Lz], dealias=1)
    domain  = de.Domain([z_basis,], grid_dtype=np.float64)


    z = domain.grid(-1)

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
    for bl_thickness in [0.1]:
        d_BL = Lz*bl_thickness
        T_bot_BL = np.mean(T_ad.interpolate(z=Lz*(1-bl_thickness))['g'])
        dTdz_BL  = (1 - T_bot_BL)/d_BL
        delta_dTdz = dTdz_BL - (-1 + epsilon/Cp)

        Tz['g'] = (-1 + epsilon/Cp) + delta_dTdz*zero_to_one(z, Lz*(1 - bl_thickness), width=bl_thickness*Lz/5)
        Tz.antidifferentiate('z', ('left', 1 + Lz), out=T)

        problemt = de.NLBVP(domain, variables=['rho', 'ln_rho', 'M'])
        problemt.parameters['T'] = T
        problemt.parameters['Tz'] = Tz
        problemt.parameters['tot_M']  = mass
        problemt.parameters['g']      = 1 + m

        problemt.add_equation('dz(M) - rho = 0')
        problemt.add_equation('ln_rho = log(rho)')
        problemt.add_equation('dz(ln_rho) = -(Tz + g)/T')
        
        problemt.add_bc(' left(M) = 0')
        problemt.add_bc('right(M) = tot_M')

        solver = problemt.build_solver()
        rho    = solver.state['rho']
        ln_rho = solver.state['ln_rho']
        M      = solver.state['M']
        rho0.antidifferentiate('z', ('left', 0), out=M)
        ln_rho['g'] = np.copy(np.log(rho0['g']))
        rho['g']    = np.copy(rho0['g'])

        pert = solver.perturbations.data
        pert.fill(1 + 1e-10)
        while np.sum(np.abs(pert)) > 1e-10:
            solver.newton_iteration()
#            print('pert norm: {}'.format(np.sum(np.abs(pert))))
        plt.plot(z, ln_rho['g'])
        plt.show()


        ln_rho_bot = np.mean(ln_rho.interpolate(z=0)['g'])
        ln_rho_top = np.mean(ln_rho.interpolate(z=Lz)['g'])
        n_rho = ln_rho_bot - ln_rho_top

        d_nrho = n_rho - n_rho0
        dS_0   = -epsilon*n_rho0 * (gamma-1) / m / gamma
        delta_dS = (gamma-1)*d_nrho / gamma
        dS     = dS_0 + delta_dS
        print('epsilon {} / BL {} / Delta n_rho: {} / dS_0 {}, delta_dS {}, dS/dS_0 {}'.format(epsilon, bl_thickness, d_nrho, dS_0, delta_dS, dS/dS_0))


    T_ad['g'] = 1 + (-1 + epsilon/Cp)*(z - Lz)
    rho_ad['g'] = T_ad['g']**m_ad
    mass_ad1 = np.mean(rho_ad.integrate('z')['g'])

    print('LOWER BL')
    for bl_thickness in [0.1]:
        d_BL = Lz*bl_thickness
        T_top_BL = np.mean(T_ad.interpolate(z=Lz*bl_thickness)['g'])
        dTdz_BL  = (T_top_BL - (1 + Lz))/d_BL
        delta_dTdz = dTdz_BL - (-1 + epsilon/Cp)

        Tz['g'] = (-1 + epsilon/Cp) + delta_dTdz*one_to_zero(z, Lz*bl_thickness, width=bl_thickness*Lz/5)
        Tz.antidifferentiate('z', ('right', 1), out=T)

        problemb = de.NLBVP(domain, variables=['rho', 'ln_rho', 'M'])
        problemb.parameters['T'] = T
        problemb.parameters['Tz'] = Tz
        problemb.parameters['tot_M']  = mass
        problemb.parameters['g']      = 1 + m

        problemb.add_equation('dz(M) - rho = 0')
        problemb.add_equation('ln_rho = log(rho)')
        problemb.add_equation('dz(ln_rho) = -(Tz + g)/T')
        
        problemb.add_bc(' left(M) = 0')
        problemb.add_bc('right(M) = tot_M')

        solver = problemb.build_solver()
        rho    = solver.state['rho']
        ln_rho = solver.state['ln_rho']
        M      = solver.state['M']
        rho0.antidifferentiate('z', ('left', 0), out=M)
        ln_rho['g'] = np.copy(np.log(rho0['g']))
        rho['g']    = np.copy(rho0['g'])

        pert = solver.perturbations.data
        pert.fill(1 + 1e-10)
        while np.sum(np.abs(pert)) > 1e-10:
            solver.newton_iteration()

        ln_rho_bot = np.mean(ln_rho.interpolate(z=0)['g'])
        ln_rho_top = np.mean(ln_rho.interpolate(z=Lz)['g'])
        n_rho = ln_rho_bot - ln_rho_top
#        print(ln_rho['g'])

        d_nrho = n_rho - n_rho0
        dS_0   = -epsilon*n_rho0 * (gamma-1) / m / gamma
        delta_dS = (gamma-1)*d_nrho / gamma
        dS     = dS_0 + delta_dS
        print('epsilon {} / BL {} / Delta n_rho: {} / dS_0 {}, delta_dS {}, dS/dS_0 {}'.format(epsilon, bl_thickness, d_nrho, dS_0, delta_dS, dS/dS_0))



#    print(mass, mass_ad1, mass_ad2)
#    print('dMt/M: {:.4e}, dMb/M: {:.4e}'.format(1 - mass_ad1/mass, 1 - mass_ad2/mass))


