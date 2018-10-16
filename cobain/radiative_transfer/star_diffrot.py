import numpy as np
import quadpy
from cobain.structure import potentials
from cobain.structure import polytropes
from scipy.interpolate import UnivariateSpline
from cobain.structure.constants import *

def adjust_stepsize(r,theta,phi,scale,ndir,bs,pot_range,le,chi_interp,S_interp,I_interp,arg):

    exps = np.linspace(-10,-4,1000)
    stepsizes = 10**exps

    b = np.ones((len(stepsizes), 3)).T * stepsizes
    newpoints = r - ndir*b.T
    taus = np.zeros(len(newpoints))

    for i, r in enumerate(newpoints):
        pot, chi, S, I = return_structure(newpoints[i], theta, phi, bs, pot_range, le, chi_interp,
                                                      S_interp, I_interp, arg, scale)
        taus[i] = chi*stepsizes[i]/2.

    diffs = np.abs(taus-1.)
    # print 'taus, diffs: ', taus, diffs
    stepsize = stepsizes[np.argmin(diffs)]

    return stepsize

def rot_theta(theta):
    if theta <= np.pi:
        return np.pi - theta
    elif theta <= 1.5 * np.pi:
        return theta - np.pi
    else:
        return 2 * np.pi - theta

def return_structure(r,theta,phi,bs,pot_range,le,chi_interp,S_interp,I_interp,arg,scale):

    if r[0] == 0.0 and r[1] == 0.0 and r[2] == 0.0:
        pot = le['pot_max']
    else:
        pot = potentials.compute_diffrot_potential(r, pot_range[0], bs, scale)
        if pot > le['pot_max']:
            pot = le['pot_max']

    if np.round(pot,14) < np.round(pot_range[0],14):
        return pot, 0., 0., 0.
    elif pot <= pot_range[1]:
        if pot < pot_range[0]:
            pot = pot_range[0]
        return pot, float(chi_interp((pot,theta,phi))), float(S_interp((pot,theta,phi))), float(I_interp[arg]((pot,theta,phi)))
        # return pot, float(chi_interp(pot)), float(S_interp(pot)), float(I_interp[arg](pot))

    else:
        T = le['Tc'] * le['theta_interp_func'](1. / pot)
        rho = le['rhoc'] * le['theta_interp_func'](1. / pot) ** le['n']
        rho_cgs = (rho * const.M_sun / (const.R_sun) ** 3).value * 1e-3
        opac_si = polytropes.opal_sun(T, rho_cgs)[0]  # this opac is in m^2/kg

        opac = (opac_si * const.M_sun / const.R_sun ** 2).value
        S = (stefb / const.L_sun.value * const.R_sun.value ** 2) * T ** 4

        return pot,opac*rho, S, S


def compute_intensity(Mc, arg, thetas, phis, chi_interp, S_interp, I_interp, scale, bs, pot_range, le, R_z, R_xy):
    N = 20000
    # theta, phi, theta_u, phi_u = prep_angles(thetas[arg], phis[arg])
    vorig = np.array(
        [np.sin(thetas[arg]) * np.cos(phis[arg]), np.sin(thetas[arg]) * np.sin(phis[arg]), np.cos(thetas[arg])])

    vnew = np.dot(R_xy, np.dot(R_z, vorig))
    theta_new = np.arccos(vnew[2])
    phi_new = np.arctan2(vnew[1], vnew[0])

    ndir = np.array(
        [np.sin(theta_new) * np.cos(phi_new), np.sin(theta_new) * np.sin(phi_new), np.cos(theta_new)])

    # prepare arrays for interpolated values
    rs = np.zeros((N,3))
    paths = np.zeros(N)
    rs[0] = Mc
    pots = np.zeros(N)
    taus = np.zeros(N)
    chis = np.zeros(N)
    Ss = np.zeros(N)
    Is = np.zeros(N)
    Ss_exp = np.zeros(N)

    i = 0
    theta_i = np.arccos(rs[i][2] / np.sqrt(np.sum(rs[i] ** 2)))
    phi_i = np.abs(np.arctan2(rs[i][1], rs[i][0]))

    pots[i], chis[i], Ss[i], Is[i] = return_structure(rs[i], theta_i, phi_i, bs, pot_range, le, chi_interp,
                                             S_interp, I_interp, arg, scale)
    # print 'initial: ', pots[i], chis[i], Ss[i], Is[i]
    Ss_exp[i] = Ss[i] * np.exp(-taus[i])
    if np.isnan(Ss_exp[i]) or np.isinf(Ss_exp[i]) or Ss_exp[i]==-np.nan:
        Ss_exp[i] = 0.
    stepsize = adjust_stepsize(Mc,theta_i,phi_i,scale,ndir,bs,pot_range,le,chi_interp,S_interp,I_interp,arg)

    while i < 100 and Is[i] != 0:

        stepsize = stepsize / 2.
        i = 0

        while taus[i] < 740. and np.round(pots[i], 14) >= np.round(pot_range[0], 14) and i < N - 1:

            i += 1
            rs[i] = Mc - i * stepsize * ndir
            theta_i = np.arccos(rs[i][2]/np.sqrt(np.sum(rs[i]**2)))
            phi_i = np.abs(np.arctan2(rs[i][1],rs[i][0]))

            if theta_i > np.pi / 2:
                theta_i = rot_theta(theta_i)
            paths[i] = i*stepsize
            pots[i], chis[i], Ss[i], Is[i] = return_structure(rs[i],theta_i,phi_i,bs,pot_range,le,chi_interp,S_interp,I_interp,arg,scale)
            chis_sp = UnivariateSpline(paths[:i + 1], chis[:i + 1], k=1, s=0)
            taus[i] = chis_sp.integral(paths[0], paths[i])
            Ss_exp[i] = Ss[i] * np.exp(-taus[i])
            if np.isnan(Ss_exp[i]) or np.isinf(Ss_exp[i]) or Ss_exp[i]==-np.nan:
        	Ss_exp[i] = 0.
    Ss_exp[pots < pot_range[0]] = 0.0

    if i > 1:
        taus = taus[:i + 1]
        Ss_exp = Ss_exp[:i + 1]
        Sexp_sp = UnivariateSpline(taus, Ss_exp, k=1, s=0)

        I = Is[i] * np.exp(-taus[-1]) + Sexp_sp.integral(taus[0], taus[-1])

        if np.isnan(I) or I < 0.:
            I = 0.0
    else:
        I = 0.0
    # print 'Out of loop, stepsize = %.12f, i = %i, I0 = %.5f, I = %.5f' % (stepsize, i, Is[0], I)

    return I, theta_new, phi_new


def run(Mc, normal, pot, thetas, phis, scale, bs, pot_range, le, chi_interp, S_interp, I_interp):
    theta_n = np.arccos(normal[2])
    phi_n = np.arctan2(normal[1], normal[0])
    theta_z = 0.0
    phi_z = -np.pi / 2

    R_z = np.array([[1., 0., 0.],
                    [0., np.cos(theta_n - theta_z), -np.sin(theta_n - theta_z)],
                    [0., np.sin(theta_n - theta_z), np.cos(theta_n - theta_z)]])
    R_xy = np.array([[np.cos(phi_n - phi_z), -np.sin(phi_n - phi_z), 0.],
                     [np.sin(phi_n - phi_z), np.cos(phi_n - phi_z), 0.],
                     [0., 0., 1.]])

    Ic_j = np.zeros(thetas.shape)
    thetas_j = np.zeros(thetas.shape)
    phis_j = np.zeros(thetas.shape)

    args_sweep = np.arange(0, len(thetas), 1).astype(int)

    for arg in args_sweep:
        # print 'Mc_ind, angles_ind: ', j, arg
        Ic_j[arg], thetas_j[arg], phis_j[arg] = compute_intensity(Mc, arg, thetas, phis, chi_interp, S_interp,
                                                                        I_interp, scale, bs, pot_range, le, R_z, R_xy)

    return Ic_j, thetas_j, phis_j
