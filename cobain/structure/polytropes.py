import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.integrate import odeint
from constants import *
import sys
import os
import logging
logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

def return_MS_factors(mass):

    if mass <= 1.:
        zeta = 0.8
    else:
        zeta = 0.57

    if mass <= 0.25:
        nu = 2.5
    elif mass <= 10.:
        nu = 4.
    else:
        nu = 3.

    return zeta, nu

def opal_sun(T, rho):
    # load opal table and prepare for interpolation
    opaldir_local = os.path.dirname(os.path.abspath(__file__))+'/'
    opal = np.loadtxt(opaldir_local+'opal_sun.csv', delimiter=',')
    x = opal.T[:, 0]
    y = opal[:, 0]
    x = np.delete(x, 0)
    y = np.delete(y, 0)
    opac = np.delete(opal, 0, 0)
    opac = np.delete(opac.T, 0, 0).T
    opac_interp = interp2d(x, y, opac)

    logT = np.log10(T)
    logR = np.log10(rho / ((T * 1e-6) ** 3))

    logk = opac_interp(logR, logT)
    if logk == 0 or logk > 9.99:
        return 0.0
    else:
        return 0.1 * (10 ** logk)  # in m^2/kg (multiplied by 0.1)

def opal_sun_points(Ts,rhos):
    # load opal table and prepare for interpolation
    opaldir_local = os.path.dirname(os.path.abspath(__file__))+'/'
    opal = np.loadtxt(opaldir_local+'opal_sun.csv', delimiter=',')
    x = opal.T[:, 0]
    y = opal[:, 0]
    x = np.delete(x, 0)
    y = np.delete(y, 0)
    opac = np.delete(opal, 0, 0)
    opac = np.delete(opac.T, 0, 0).T
    opac_interp = interp2d(x, y, opac)

    logTs = np.log10(Ts)
    logRs = np.log10(rhos / ((Ts * 1e-6) ** 3))

    logk = np.zeros(len(logTs))
    for i,logT in enumerate(logTs):
        logk[i] = opac_interp(logRs[i], logTs[i])[0]

    logk[(logk==0.) & (logk > 9.99)] = -np.inf

    return 0.1 * (10 ** logk)  # in m^2/kg (multiplied by 0.1)

def lane_emden(n):
    ''' solves the Lane-Emden equation for polytropic index n as a system of
    first order odes: dtheta/dxi = u, du/dxi = -theta^n - 2u/xi
    y = [theta, u]'''

    def f(y, t):
        return [y[1], -y[0] ** n - 2 * y[1] / t]

    y0 = [1., 0.]
    if n <= 1:
        tmax = 3.5
    elif n <= 2:
        tmax = 5.
    else:
        tmax = 10.
    ts = np.arange(1e-120, tmax, 1e-4)
    soln = odeint(f, y0, ts)

    if np.isnan(soln[:,0]).any():
        nonans = np.argwhere(~np.isnan(soln[:,0])).flatten()
        t_surface = float(ts[nonans[-1]])
        dtheta_surface = float(soln[:,1][nonans[-1]])
        theta_interp = interp1d(ts[:nonans[-1]+1], soln[:,0][:nonans[-1]+1])
    else:
        theta_interp = interp1d(ts, soln[:,0])
        dtheta_interp = interp1d(ts, soln[:,1])

        # compute the value of t and dthetadt where theta falls to zero
        ts_theta_interp = interp1d(soln[:,0], ts)
        t_surface = float(ts_theta_interp(0.))
        dtheta_surface = float(dtheta_interp(t_surface))

    return t_surface, dtheta_surface, theta_interp

def lane_emden_tidal(n, q, xiu, k):
    def f(y, t):
        def A(t):
            return t ** 2 * (1 + (2. / 5. * q ** 2 + 4. / 15. * 0.5 * (q + 1) * q + 16. / 15. * (
                0.5 * (q + 1.)) ** 2) * t ** 6 + 9. / 14. * q ** 2 * t ** 8 + 8. / 9. * q ** 2 * t ** 10)

        def dA(t):
            return 2 * t * (1 + (2. / 5. * q ** 2 + 4. / 15. * 0.5 * (q + 1) * q + 16. / 15. * (
                0.5 * (q + 1.)) ** 2) * t ** 6 + 9. / 14. * q ** 2 * t ** 8 + 8. / 9. * q ** 2 * t ** 10) + \
                   t ** 2 * (6 * (2. / 5. * q ** 2 + 4. / 15. * 0.5 * (q + 1) * q + 16. / 15. * (0.5 * (
                       q + 1.)) ** 2) * t ** 5 + 8. * 9. / 14. * q ** 2 * t ** 7 + 10. * 8. / 9. * q ** 2 * t ** 9)

        def B(t):
            return 1 + 4 * (0.5 * (q + 1)) * t ** 3 + (36. / 5. * q ** 2 + 24. / 5. * 0.5 * (
                q + 1) * q + 96. / 5. * (0.5 * (
                q + 1)) ** 2) * t ** 6 + 55. / 7 * q ** 2 * t ** 8 + 26. / 3 * q ** 2 * t ** 10

        return [y[1], (-(xiu / k) ** 2 * t ** 2 * B(t) * y[0] ** n - dA(t) * y[1]) / A(t)]

    y0 = [1., 0.]
    ts = np.arange(1e-120, 10., 1e-4)
    soln = odeint(f, y0, ts)

    if np.isnan(soln[:, 0]).any():
        nonans = np.argwhere(~np.isnan(soln[:, 0])).flatten()

        ts_new = np.linspace(ts[nonans[-1]],ts[nonans[-1]+1], 1e4)
        soln_new = odeint(f, y0, ts_new)
        nonans_new = np.argwhere(~np.isnan(soln_new[:, 0])).flatten()

        t_surface = float(ts_new[nonans_new[-1]])
        dtheta_surface = float(soln_new[:, 1][nonans_new[-1]])

        soln[:, 0][np.isnan(soln[:, 0])] = 0.0
        soln_new[:, 0][np.isnan(soln_new[:, 0])] = 0.0

        ts_stack = np.hstack((ts[:nonans[-1]+1], ts_new, ts[nonans[-1]+1:]))
        soln_stack = np.hstack((soln[:,0][:nonans[-1]+1], soln_new[:,0], soln[:,0][nonans[-1]+1:]))
        theta_interp_return = np.array([ts_stack, soln_stack])
    else:

        # func has the form (xi, theta, dtheta/dxi)
        theta_interp = interp1d(ts, soln[:, 0], fill_value='extrapolate')
        dtheta_interp = interp1d(ts, soln[:, 1])

        negtheta = int(np.argwhere(theta_interp(ts) < 0)[1])
        # compute the value of t and dthetadt where theta falls to zero
        ts_theta_interp = interp1d(soln[:, 0][:negtheta], ts[:negtheta])
        t_surface = float(ts_theta_interp(0.))
        dtheta_surface = float(dtheta_interp(t_surface))
        theta_interp_return = np.array([ts, soln[:, 0]])

    return t_surface, dtheta_surface, theta_interp_return


def LE_tidal(R, M, T, k, q, n=3., mu=0.6):
    xi_s, dthetadxi_s, theta_interp_xi = lane_emden(n)

    rn = R / xi_s
    rhoc = (-1) * M / (4 * np.pi * rn ** 3 * xi_s ** 2 * dthetadxi_s)
    Tc = 1. / ((n + 1) * xi_s * ((-1) * dthetadxi_s)) * GG_sol * M * mu * mp_kB_sol / R

    # k = xi_s * rn / sma

    # compute the polytrope of the contact star
    r0_s, dthetadr0_s, theta_interp = lane_emden_tidal(n, q, xi_s, k)
    #
    # Tc = T/theta_interp(r0_pot)*((2./3.)**(3./4.)+0.5)**0.25
    # R = 1. / ((n + 1) * xi_s * ((-1) * dthetadxi_s)) * GG_sol * M * mu * mp_kB_sol / Tc
    # rn = R / xi_s
    # rhoc = (-1) * M / (4 * np.pi * rn ** 3 * xi_s ** 2 * dthetadxi_s)

    le = {'n': n, 'xi_s': xi_s, 'dthetadxi_s': dthetadxi_s, 'r0_s': r0_s, 'theta_interp': theta_interp,
          'rhoc': rhoc, 'Tc': Tc}

    return le

def lane_emden_diffrot(n, bs, xiu):
    def f(y, t):
        def A(t):
            return t ** 2 * (1 - 1. / 15. * bs[0] ** 2 * t ** 6 - 8. / 105. * bs[0] * bs[1] * t ** 8 - (
                16. / 315. * bs[0] * bs[2] + 8. / 315. * bs[1] ** 2) * t ** 10 - 196. / 525. * bs[0] ** 2 * bs[
                                 1] * t ** 11 - 128. / 3405. * bs[1] * bs[2] * t ** 12)

        def dA(t):
            return 2 * t * (1 - 1. / 15. * bs[0] ** 2 * t ** 6 - 8. / 105. * bs[0] * bs[1] * t ** 8 - (
                16. / 315. * bs[0] * bs[2] + 8. / 315. * bs[1] ** 2) * t ** 10 - 196. / 525. * bs[0] ** 2 * bs[
                                1] * t ** 11 - 128. / 3405. * bs[1] * bs[2] * t ** 12) + t ** 2 * (
                -6. / 15. * bs[0] ** 2 * t ** 5 - 8. * 8. / 105. * bs[0] * bs[1] * t ** 7 - 10 * (
                    16. / 315. * bs[0] * bs[2] + 8. / 315. * bs[1] ** 2) * t ** 9 - 196. * 11. / 525. * bs[0] ** 2 * bs[
                    1] * t ** 10 - 128. * 12. / 3405. * bs[1] * bs[2] * t ** 11)

        def B(t):
            return 1 + 2 * bs[0] * t ** 3 + 16. / 15. * bs[1] * t ** 5 + 24. / 5. * bs[0] ** 2 * t ** 6 + \
                   16. / 21. * bs[2] * t ** 7 + 44. / 7. * bs[0] * bs[1] * t ** 8 + (1664./315. * bs[0] * bs[2] + 208./105. * bs[1]**2) * t ** 10 + \
                   2912./105. * bs[0]**2 * bs[1] * t ** 11 + 2240./693. * bs[1] * bs[2] * t ** 12

        return [y[1],(-xiu ** 2 * t ** 2 * B(t) * y[0] ** n - dA(t) * y[1])/A(t)]

    y0 = [1., 0.]
    ts = np.arange(1e-120, 8., 1e-4)
    soln = odeint(f, y0, ts)

    # func has the form (xi, theta, dtheta/dxi)
    theta_interp = interp1d(ts, soln[:,0], fill_value='extrapolate')
    dtheta_interp = interp1d(ts, soln[:,1], fill_value='extrapolate')

    negtheta = int(np.argwhere(theta_interp(ts) < 0)[1])
    # compute the value of t and dthetadt where theta falls to zero
    ts_theta_interp = interp1d(soln[:,0][:negtheta], ts[:negtheta])
    t_surface = float(ts_theta_interp(0.))
    dtheta_surface = float(dtheta_interp(t_surface))

    return t_surface, dtheta_surface, np.array([ts, soln[:, 0]])

def LE_diffrot(bs, R, M, T, n=3., mu=0.6):
    # R1 is the undistorted radius of the outermost potential surface
    xi_s, dthetadxi_s, theta_interp_xi = lane_emden(n)

    rn = R / xi_s
    rhoc = (-1) * M / (4 * np.pi * rn ** 3 * xi_s ** 2 * dthetadxi_s)
    Pc = (4 * np.pi * GG_sol * rhoc ** 2 * rn ** 2) / (n + 1)
    Tc = 1. / ((n + 1) * xi_s * ((-1) * dthetadxi_s)) * GG_sol * M * mu * mp_kB_sol / R

    # compute the polytrope of the distorted star
    r0_s, dthetadr0_s, theta_interp = lane_emden_diffrot(n, bs, xi_s)

    le = {'n': n, 'xi_s': xi_s, 'r0_s': r0_s, 'theta_interp': theta_interp, 'rhoc': rhoc, 'Tc': Tc, 'pot_max': 1e+120}

    return le


def find_opdepth_eq1_diffrot(le, Teff):

    pot = 1./le['r0_s']
    r0s = np.linspace(1./pot - 0.1, 1./pot, 10000)
    op_depths = np.zeros(len(r0s))

    theta_interp = interp1d(le['theta_interp'][0],le['theta_interp'][1])
    for i, r0 in enumerate(r0s):
        T = le['Tc'] * theta_interp(r0)
        if T > 0:
            op_depths[i] = ((T / Teff) ** 4 - 0.5) * 4.0 / 3.0
        else:
            op_depths[i] = 0.0
        # print 'diffrot pot = %.2f T = %.2f, op_depth = %.2f' % (1. / r0, T, op_depths[i])

    r0_intp = interp1d(op_depths, r0s)

    r0_tau1 = r0_intp([2.0 / 3.0])[0]

    return 1./r0_tau1

def find_opdepth_eq1_contact(le, Teff, q):

    pot = 1./le['r0_s'] + q
    r0s = np.linspace(1./(pot-q) - 0.1, 1./(pot-q), 10000)
    op_depths = np.zeros(len(r0s))
    theta_interp = interp1d(le['theta_interp'][0], le['theta_interp'][1])

    for i, r0 in enumerate(r0s):
        T = le['Tc'] * theta_interp(r0)
        if T > 0:
            op_depths[i] = ((T / Teff) ** 4 - 0.5) * 4.0 / 3.0
        else:
            op_depths[i] = 0.0

        # print 'contact pot = %.2f, r0 = %.2f, T = %.2f, op_depth = %.2f' % (1./r0 + q, r0, T, op_depths[i])

    r0_intp = interp1d(op_depths, r0s)

    r0_tau1 = r0_intp([2.0 / 3.0])[0]

    return 1./r0_tau1 + q