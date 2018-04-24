import numpy as np
from cobain.structure import potentials
from cobain.structure import polytropes
from scipy.interpolate import UnivariateSpline
from cobain.structure.constants import *
import astropy.constants as const

def rot_theta(theta):
    if theta <= np.pi:
        return np.pi - theta
    elif theta <= 1.5 * np.pi:
        return theta - np.pi
    else:
        return 2 * np.pi - theta

def rot_theta_points(thetas):
    thetas[thetas <= np.pi] = np.pi-thetas[thetas <= np.pi]
    thetas[(thetas > np.pi) & (thetas <= 1.5*np.pi)] = thetas[(thetas > np.pi) & (thetas <= 1.5*np.pi)] - np.pi
    thetas[thetas > 1.5*np.pi] = 2*np.pi - thetas[thetas > 1.5*np.pi]

    return thetas

def adjust_stepsize(r, theta, phi, scale, r0_scales, ndir, q, les, pot_range, nekmin, angle_breaks, chis_interp, Ss_interp, Is_interp, arg):

    exps = np.linspace(-10,-4,1000)
    stepsizes = 10**exps

    b = np.ones((len(stepsizes), 3)).T * stepsizes
    newpoints = r - ndir * b.T
    taus = np.zeros(len(newpoints))

    for i,r in enumerate(newpoints):
        pot, chi, S, I = return_structure(r, theta, phi, q, nekmin, angle_breaks, pot_range, les, chis_interp, Ss_interp, Is_interp, arg, scale, r0_scales)
        taus[i] = chi*stepsizes[i]/2.

    diffs = np.abs(taus - 1.)
    stepsize_final = stepsizes[np.argmin(diffs)]

    # print 'stepsize: ', stepsize_final
    return stepsize_final

def return_le(pot,le,r0_scale,q):
    T = le['Tc'] * le['theta_interp_func'](r0_scale / (pot - q))
    rho = le['rhoc'] * le['theta_interp_func'](r0_scale / (pot - q)) ** le['n']
    rho_cgs = (rho * const.M_sun / (const.R_sun) ** 3).value * 1e-3
    opac_si = polytropes.opal_sun(T, rho_cgs)[0]  # this opac is in m^2/kg

    opac = (opac_si * const.M_sun / const.R_sun ** 2).value
    S = (stefb / const.L_sun.value * const.R_sun.value ** 2) * T ** 4
    return opac * rho, S

def return_values(pot,theta,phi,arg,le,r0_scale,q,pot_range,breaks,chi_interp,S_interp,I_interp):

    # print 'pot %s, potrange %s, %s' % (pot, pot_range[0], pot_range[1])
    if pot > le['pot_max']:
        pot = le['pot_max']

    if np.round(pot, 8) < np.round(pot_range[0], 8):
        # print 'outside of star, returning zeros'
        chi = 0.
        S = 0.
        I = 0.

    elif pot <= pot_range[1]:
        if pot < pot_range[0]:
            pot = pot_range[0]

        # check if angles between neck breaks
        pot_diffs = breaks[:,0] - pot
        # print breaks[:,0]
        min_pot = np.argmax(pot_diffs[pot_diffs <= 0.])
        max_pot = np.argmin(pot_diffs[pot_diffs >= 0.])

        # if angles not in neck breaks interpolate from grids
        if theta <= breaks[:,1][min_pot] and theta <= breaks[:,1][max_pot] and phi >= breaks[:,2][min_pot] and phi >= breaks[:,2][max_pot]:
            # print 'interpolating in grid'
            chi = float(chi_interp((pot, theta, phi)))
            S = float(S_interp((pot, theta, phi)))
            I = float(I_interp[arg]((pot, theta, phi)))
        # otherwise use Lane-Emden solution
        else:
            # print 'in neck, using LE'
            chi, S = return_le(pot, le, r0_scale, q)
            I = S

    else:
        # print 'deeper in star, using LE'
        chi, S = return_le(pot,le,r0_scale,q)
        I = S

    return chi, S, I

def return_structure(r,theta,phi,q,nekmin,angle_breaks,pot_range,les,chis_interp,Ss_interp,Is_interp,arg,scale,r0_scales):

    if np.all(r == 0.0):
        pot = les[0]['pots_max']
        T = les[0]['Tc']
        rho = les[0]['rhoc']
        rho_cgs = (rho * const.M_sun / (const.R_sun) ** 3).value * 1e-3
        opac_si = polytropes.opal_sun(T, rho_cgs)[0]  # this opac is in m^2/kg

        opac = (opac_si * const.M_sun / const.R_sun ** 2).value
        S = (stefb / const.L_sun.value * const.R_sun.value ** 2) * T ** 4
        return pot, opac * rho, S, S

    else:
        # r_abs = np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        if r[0] <= nekmin:
            comp = 0
            pot = potentials.BinaryRoche(r / scale, q)
            le = les[0]
            r0_scale = r0_scales[0]
            chi_interp = chis_interp[0]
            S_interp = Ss_interp[0]
            I_interp = Is_interp[0]
            breaks = angle_breaks[0]
            pot_limits = np.array(pot_range).copy()
            q_compute=q

        else:
            comp = 1
            rnew = r.copy()
            rnew[0] = scale - rnew[0]
            pot = potentials.BinaryRoche(rnew / scale, 1. / q)
            theta = np.arccos(rnew[2] / np.sqrt(np.sum(rnew ** 2)))
            phi = np.abs(np.arctan2(rnew[1] / np.sqrt(np.sum(rnew ** 2)), rnew[0] / np.sqrt(np.sum(rnew ** 2))))

            if theta > np.pi / 2:
                theta = rot_theta(theta)

            le = les[1]
            r0_scale = r0_scales[1]
            chi_interp = chis_interp[1]
            S_interp = Ss_interp[1]
            I_interp = Is_interp[1]
            breaks = angle_breaks[1]
            # pot_limits = np.array([float(pot_range[0] / q + 0.5 * (q - 1) / q), float(pot_range[1] / q + 0.5 * (q - 1) / q)])
            pot_limits = np.array(pot_range).copy()/q + 0.5 * (q - 1) / q
            # pot_range[0] = pot_range[0] / q + 0.5 * (q - 1) / q
            # pot_range[1] = pot_range[1] / q + 0.5 * (q - 1) / q

            # print 'secondary pot_range:', pot_limits
            q_compute = 1./q

        chi, S, I = return_values(pot,theta,phi,arg,le,r0_scale,q_compute,pot_limits,breaks,chi_interp,S_interp,I_interp)

        if comp == 0:
            return pot, chi, S, I
        elif comp ==1:
            q = 1./q
            return pot * q - 0.5 * (q - 1), chi, S, I

def return_structure_grid(pots, thetas, phis, chi, S, I):
    return chi((pots, thetas, phis)), S((pots, thetas, phis)), I((pots, thetas, phis))

def return_structure_le(pots,q,le,r0_scale):

    Ts = le['Tc'] * le['theta_interp_func'](r0_scale / (pots - q))
    rhos = le['rhoc'] * le['theta_interp_func'](r0_scale / (pots - q)) ** le['n']
    rhos_cgs = (rhos * const.M_sun / (const.R_sun) ** 3).value * 1e-3
    opac_si = polytropes.opal_sun_points(Ts, rhos_cgs)  # this opac is in m^2/kg

    opac = (opac_si * const.M_sun / const.R_sun ** 2).value
    Ss = (stefb / const.L_sun.value * const.R_sun.value ** 2) * Ts ** 4

    return opac * rhos, Ss, Ss


def mask_neck_points(prim_cond, sec_cond, pot_cond, pots, thetas, phis, nekmin, angle_breaks, qs):

    pots_argmin_prim = np.searchsorted(angle_breaks[0][:,0], pots[prim_cond & pot_cond])
    pots_argmin_sec = np.searchsorted(angle_breaks[1][:,0], pots[sec_cond & pot_cond])
    thetas_min_prim = angle_breaks[0][:,1][pots_argmin_prim-1]
    thetas_max_prim = angle_breaks[0][:,1][pots_argmin_prim]
    phis_min_prim = angle_breaks[0][:,2][pots_argmin_prim-1]
    phis_max_prim = angle_breaks[0][:,2][pots_argmin_prim]
    thetas_min_sec = angle_breaks[1][:,1][pots_argmin_sec-1]
    thetas_max_sec = angle_breaks[1][:,1][pots_argmin_sec]
    phis_min_sec = angle_breaks[1][:,2][pots_argmin_sec-1]
    phis_max_sec = angle_breaks[1][:,2][pots_argmin_sec]

    neck_cond = np.zeros(len(pots))
    neck_cond[prim_cond & pot_cond][(thetas[prim_cond & pot_cond] > thetas_min_prim) | (thetas[prim_cond & pot_cond] > thetas_max_prim)
                                    | (phis[prim_cond & pot_cond] < phis_min_prim) | (phis[prim_cond & pot_cond] < phis_max_prim)] = 1
    neck_cond[sec_cond & pot_cond][
        (thetas[sec_cond & pot_cond] > thetas_min_sec) | (thetas[sec_cond & pot_cond] > thetas_max_sec) |
        (phis[sec_cond & pot_cond] < phis_min_sec) | (phis[sec_cond & pot_cond] < phis_max_sec)] = 1
    neck_cond[~pot_cond] = 1

    return neck_cond


def adjust_stepsize_points(r, theta, phi, scale, r0_scales, ndir, q, les, pot_range, nekmin, angle_breaks, chis_interp, Ss_interp, Is_interp, arg):

    exps = np.linspace(-10,-4,1000)
    stepsizes = 10**exps

    b = np.ones((len(stepsizes), 3)).T * stepsizes
    newpoints = r - ndir * b.T

    prim_cond = (newpoints[:, 0] <= nekmin)
    sec_cond = (newpoints[:, 0] > nekmin)

    pots = np.zeros(1000)
    invalid1 = np.all(newpoints == 0., axis=1)
    invalid2 = (newpoints[:, 0] == scale) & (newpoints[:, 1] == 0.) & (newpoints[:, 2] == 0.)
    pots[~invalid1 & ~invalid2] = potentials.BinaryRoche_points(newpoints[~invalid1 & ~invalid2] / scale, q)
    pots[invalid1] = les[0]['pot_max']
    pots[invalid2] = les[1]['pot_max']

    # take care of pots that rounded up belong to grid but otherwise don't
    pots[(np.round(pots, 8) >= np.round(pot_range[0], 8)) & (pots < pot_range[0])] = pot_range[0]

    rs = newpoints
    rs_sec = rs[sec_cond].copy()
    rs_sec[:,0] = scale - rs_sec[:,0]
    thetas = np.zeros(1000) 
    phis = np.zeros(1000)    

    thetas[prim_cond] = np.arccos(rs[prim_cond][:,2] / np.sqrt(np.sum(rs[prim_cond] ** 2, axis=1)))
    phis[prim_cond] = np.abs(np.arctan2(rs[prim_cond][:,1] / np.sqrt(np.sum(rs[prim_cond] ** 2, axis=1)), rs[prim_cond][:,0] / np.sqrt(np.sum(rs[prim_cond] ** 2, axis=1))))
    thetas[sec_cond] = np.arccos(rs_sec[:, 2] / np.sqrt(np.sum(rs_sec ** 2, axis=1)))
    phis[sec_cond] = np.abs(np.arctan2(rs_sec[:, 1] / np.sqrt(np.sum(rs_sec ** 2, axis=1)), rs_sec[:, 0] / np.sqrt(np.sum(rs_sec ** 2, axis=1))))

    thetas[thetas > np.pi / 2] = rot_theta_points(thetas[thetas > np.pi / 2])

    # find breaks in theta and phi for each potential
    pot_cond = (pots >= pot_range[0]) & (pots <= pot_range[1])
    neck_cond = mask_neck_points(prim_cond, sec_cond, pot_cond, pots, thetas, phis, nekmin, angle_breaks, q)

    chis = np.zeros(len(pots))
    Ss = np.zeros(len(pots))
    Is = np.zeros(len(pots))

    # find indices of point belonging to different computation regimes
    prim_grid = np.argwhere(prim_cond & pot_cond & (neck_cond == 0)).flatten()
    prim_le = np.argwhere(prim_cond & ((pots > pot_range[1]) | (pot_cond & (neck_cond==1)))).flatten()
    sec_grid = np.argwhere(sec_cond & pot_cond & (neck_cond == 0)).flatten()
    sec_le = np.argwhere(sec_cond & ((pots > pot_range[1]) | (pot_cond & (neck_cond==1)))).flatten()

    chis[prim_grid], Ss[prim_grid], Is[prim_grid] = return_structure_grid(pots[prim_grid], thetas[prim_grid],
                                                                                  phis[prim_grid], chis_interp[0],
                                                                                  Ss_interp[0], Is_interp[0][arg])
    chis[prim_le], Ss[prim_le], Is[prim_le] = return_structure_le(pots[prim_le],q,les[0],r0_scales[0])

    chis[sec_grid], Ss[sec_grid], Is[sec_grid] = return_structure_grid(pots[sec_grid],thetas[sec_grid], phis[sec_grid],
                                                                                 chis_interp[1], Ss_interp[1],
                                                                                 Is_interp[1][arg])
    chis[sec_le], Ss[sec_le], Is[sec_le] = return_structure_le(pots[sec_le]/q + 0.5 * (q - 1) / q, 1./q,
                                                                         les[1],r0_scales[1])

    taus = chis*stepsizes/2.

    diffs = np.abs(taus - 1.)
    stepsize_final = stepsizes[np.argmin(diffs)]

    # print 'stepsize: ', stepsize_final
    return stepsize_final/2.

def compute_intensity_points(Mc, arg, q, nekmin, angle_breaks, thetas, phis, chis_interp, Ss_interp, Is_interp, scale, r0_scales, pot_range,
                      les, R):

    # set up the direction (rotate with respect to normal)

    N = 5000
    vorig = np.array(
        [np.sin(thetas[arg]) * np.cos(phis[arg]), np.sin(thetas[arg]) * np.sin(phis[arg]), np.cos(thetas[arg])])

    vnew = np.dot(R, vorig)
    theta_new = np.arccos(vnew[2])
    phi_new = np.arctan2(vnew[1], vnew[0])
    ndir = np.array(
        [np.sin(theta_new) * np.cos(phi_new), np.sin(theta_new) * np.sin(phi_new), np.cos(theta_new)])

    theta_0 = np.arccos(Mc[2] / np.sqrt(np.sum(Mc ** 2)))
    phi_0 = np.abs(np.arctan2(Mc[1]/np.sqrt(np.sum(Mc ** 2)), Mc[0]/np.sqrt(np.sum(Mc ** 2))))

    # print 'theta_i, phi_i: ', theta_i, phi_i
    if theta_0 > np.pi / 2:
        theta_0 = rot_theta(theta_0)

    stepsize = adjust_stepsize_points(Mc, theta_0, phi_0, scale, r0_scales, ndir, q, les, pot_range, nekmin,
                                      angle_breaks, chis_interp, Ss_interp,
                                      Is_interp, arg)

    # nbreak = 0
    # Iinit = 100.
    subd = 'no'
    # div = 2.
    #
    # while nbreak < 200 and Iinit != 0.0:
    #
    #     subd += 1
    #     print 'div %s, new stepsize %s' % (div, stepsize)
    #     stepsize = stepsize/div

    rs = np.array([Mc - i * stepsize * ndir for i in range(N)])
    paths = np.array([i * stepsize for i in range(N)])

    prim_cond = (rs[:, 0] <= nekmin)
    sec_cond = (rs[:, 0] > nekmin)

    pots = np.zeros(N)
    invalid1 = np.all(rs==0.,axis=1)
    invalid2 = (rs[:,0]==scale) & (rs[:,1]==0.) & (rs[:,2]==0.)
    pots[~invalid1 & ~invalid2] = potentials.BinaryRoche_points(rs[~invalid1 & ~invalid2] / scale, q)
    pots[invalid1] = les[0]['pot_max']
    pots[invalid2] = les[1]['pot_max']

    # print pots

    # take care of pots that rounded up belong to grid but otherwise don't
    pots[(np.round(pots, 8) >= np.round(pot_range[0], 8)) & (pots < pot_range[0])] = pot_range[0]

    rs_sec = rs[sec_cond].copy()
    rs_sec[:,0] = scale - rs_sec[:,0]
    thetas = np.zeros(N) 
    phis = np.zeros(N)    
    thetas[prim_cond] = np.arccos(rs[prim_cond][:,2] / np.sqrt(np.sum(rs[prim_cond] ** 2, axis=1)))
    phis[prim_cond] = np.abs(np.arctan2(rs[prim_cond][:,1] / np.sqrt(np.sum(rs[prim_cond] ** 2, axis=1)), rs[prim_cond][:,0] / np.sqrt(np.sum(rs[prim_cond] ** 2, axis=1))))
    thetas[sec_cond] = np.arccos(rs_sec[:, 2] / np.sqrt(np.sum(rs_sec ** 2, axis=1)))
    phis[sec_cond] = np.abs(np.arctan2(rs_sec[:, 1] / np.sqrt(np.sum(rs_sec ** 2, axis=1)), rs_sec[:, 0] / np.sqrt(np.sum(rs_sec ** 2, axis=1))))

    thetas[thetas > np.pi/2] = rot_theta_points(thetas[thetas > np.pi/2])

    # find breaks in theta and phi for each potential
    pot_cond = (pots >= pot_range[0]) & (pots <= pot_range[1])
    neck_cond = mask_neck_points(prim_cond, sec_cond, pot_cond, pots, thetas, phis, nekmin, angle_breaks, q)

    chis = np.zeros(len(pots))
    Ss = np.zeros(len(pots))
    Is = np.zeros(len(pots))

    # find indices of point belonging to different computation regimes

    prim_grid = np.argwhere(prim_cond & pot_cond & (neck_cond == 0)).flatten()
    prim_le = np.argwhere(prim_cond & ((pots > pot_range[1]) | (pot_cond & (neck_cond==1)))).flatten()
    sec_grid = np.argwhere(sec_cond & pot_cond & (neck_cond == 0)).flatten()
    sec_le = np.argwhere(sec_cond & ((pots > pot_range[1]) | (pot_cond & (neck_cond==1)))).flatten()

    chis[prim_grid], Ss[prim_grid], Is[prim_grid] = return_structure_grid(pots[prim_grid], thetas[prim_grid],
                                                                                  phis[prim_grid], chis_interp[0],
                                                                                  Ss_interp[0], Is_interp[0][arg])
    chis[prim_le], Ss[prim_le], Is[prim_le] = return_structure_le(pots[prim_le],q,les[0],r0_scales[0])

    chis[sec_grid], Ss[sec_grid], Is[sec_grid] = return_structure_grid(pots[sec_grid], thetas[sec_grid], phis[sec_grid],
                                                                                 chis_interp[1], Ss_interp[1],
                                                                                 Is_interp[1][arg])
    chis[sec_le], Ss[sec_le], Is[sec_le] = return_structure_le(pots[sec_le]/ q + 0.5 * (q - 1) / q, 1./q,
                                                                         les[1],r0_scales[1])


    chis_sp = UnivariateSpline(paths, chis, k=1, s=0)
    taus = np.array([chis_sp.integral(paths[0], paths[i]) for i in range(N)])
    Ss_exp = Ss * np.exp(-taus)

    Iszero = np.argwhere(Is==0.0).flatten()
    Ssezero = np.argwhere(Ss_exp==0.0).flatten()

    if Iszero.size == 0 and Ssezero.size == 0:
        nbreak = N-1
    else:
        nbreak = np.min(np.hstack((Iszero,Ssezero)))

    if nbreak > 1:
        taus = taus[:nbreak + 1]
        Ss_exp = Ss_exp[:nbreak + 1]
        Ss_exp[(Ss_exp == np.nan) | (Ss_exp ==np.inf) | (Ss_exp == -np.nan) | (Ss_exp == -np.inf)] = 0.0

        taus_u, indices = np.unique(taus, return_index=True)

        # if len(taus_u) > 2 and len(taus_u) < 10:
        #     Sexp_sp = UnivariateSpline(taus[indices], Ss_exp[indices], k=2, s=0)
        #     I = Is[nbreak] * np.exp(-taus[-1]) + Sexp_sp.integral(taus[0], taus[-1])
        if len(taus_u) > 1:
            Sexp_sp = UnivariateSpline(taus[indices], Ss_exp[indices], k=1, s=0)
            I = Is[nbreak] * np.exp(-taus[-1]) + Sexp_sp.integral(taus[0], taus[-1])
        else:
            I = 0.0

        if np.isnan(I) or I < 0.:
            I = 0.0

    else:
        taus = taus[:nbreak + 1]
        I = 0.0

    Iinit = Is[nbreak]

    if (nbreak < 1000 and Iinit != 0.0) or nbreak==4999:
        if (nbreak < 1000 and Iinit != 0.0):
            subd = 'yes'
            div = 1000./nbreak
            stepsize = stepsize/div
        elif nbreak==N-1:
            subd = 'yes'
            N = N*2
        else:
            subd = 'yes'
            div = 1000./nbreak
            stepsize = stepsize/div
            N = N * 2

        rs = np.array([Mc - i * stepsize * ndir for i in range(N)])
        paths = np.array([i * stepsize for i in range(N)])

        prim_cond = (rs[:, 0] <= nekmin)
        sec_cond = (rs[:, 0] > nekmin)

        pots = np.zeros(N)
        invalid1 = np.all(rs == 0., axis=1)
        invalid2 = (rs[:, 0] == scale) & (rs[:, 1] == 0.) & (rs[:, 2] == 0.)
        pots[~invalid1 & ~invalid2] = potentials.BinaryRoche_points(rs[~invalid1 & ~invalid2] / scale, q)
        pots[invalid1] = les[0]['pot_max']
        pots[invalid2] = les[1]['pot_max']

        # print pots

        # take care of pots that rounded up belong to grid but otherwise don't
        pots[(np.round(pots, 8) >= np.round(pot_range[0], 8)) & (pots < pot_range[0])] = pot_range[0]

        rs_sec = rs[sec_cond].copy()
        rs_sec[:,0] = scale - rs_sec[:,0]
        thetas = np.zeros(N)
        phis = np.zeros(N)
 
        thetas[prim_cond] = np.arccos(rs[prim_cond][:,2] / np.sqrt(np.sum(rs[prim_cond] ** 2, axis=1)))
        phis[prim_cond] = np.abs(np.arctan2(rs[prim_cond][:,1] / np.sqrt(np.sum(rs[prim_cond] ** 2, axis=1)), rs[prim_cond][:,0] / np.sqrt(np.sum(rs[prim_cond] ** 2, axis=1))))
        thetas[sec_cond] = np.arccos(rs_sec[:, 2] / np.sqrt(np.sum(rs_sec ** 2, axis=1)))
        phis[sec_cond] = np.abs(np.arctan2(rs_sec[:, 1] / np.sqrt(np.sum(rs_sec ** 2, axis=1)), rs_sec[:, 0] / np.sqrt(np.sum(rs_sec ** 2, axis=1))))

        thetas[thetas > np.pi / 2] = rot_theta_points(thetas[thetas > np.pi / 2])

        # find breaks in theta and phi for each potential
        pot_cond = (pots >= pot_range[0]) & (pots <= pot_range[1])
        neck_cond = mask_neck_points(prim_cond, sec_cond, pot_cond, pots, thetas, phis, nekmin, angle_breaks, q)

        chis = np.zeros(len(pots))
        Ss = np.zeros(len(pots))
        Is = np.zeros(len(pots))

        # find indices of point belonging to different computation regimes

        prim_grid = np.argwhere(prim_cond & pot_cond & (neck_cond == 0)).flatten()
        prim_le = np.argwhere(prim_cond & ((pots > pot_range[1]) | (pot_cond & (neck_cond == 1)))).flatten()
        sec_grid = np.argwhere(sec_cond & pot_cond & (neck_cond == 0)).flatten()
        sec_le = np.argwhere(sec_cond & ((pots > pot_range[1]) | (pot_cond & (neck_cond == 1)))).flatten()

        chis[prim_grid], Ss[prim_grid], Is[prim_grid] = return_structure_grid(pots[prim_grid], thetas[prim_grid],
                                                                              phis[prim_grid], chis_interp[0],
                                                                              Ss_interp[0], Is_interp[0][arg])
        chis[prim_le], Ss[prim_le], Is[prim_le] = return_structure_le(pots[prim_le], q, les[0], r0_scales[0])

        chis[sec_grid], Ss[sec_grid], Is[sec_grid] = return_structure_grid(pots[sec_grid], thetas[sec_grid],
                                                                           phis[sec_grid],
                                                                           chis_interp[1], Ss_interp[1],
                                                                           Is_interp[1][arg])
        chis[sec_le], Ss[sec_le], Is[sec_le] = return_structure_le(pots[sec_le] / q + 0.5 * (q - 1) / q, 1. / q,
                                                                   les[1], r0_scales[1])

        chis_sp = UnivariateSpline(paths, chis, k=1, s=0)
        taus = np.array([chis_sp.integral(paths[0], paths[i]) for i in range(N)])
        Ss_exp = Ss * np.exp(-taus)

        Iszero = np.argwhere(Is == 0.0).flatten()
        Ssezero = np.argwhere(Ss_exp == 0.0).flatten()

        if Iszero.size == 0 and Ssezero.size == 0:
            nbreak = N-1
        else:
            nbreak = np.min(np.hstack((Iszero, Ssezero)))

        if nbreak > 1:
            taus = taus[:nbreak + 1]
            Ss_exp = Ss_exp[:nbreak + 1]
            Ss_exp[(Ss_exp == np.nan) | (Ss_exp == np.inf) | (Ss_exp == -np.nan) | (Ss_exp == -np.inf)] = 0.0

            taus_u, indices = np.unique(taus, return_index=True)

            # if len(taus_u) > 2 and len(taus_u) < 10:
            #     Sexp_sp = UnivariateSpline(taus[indices], Ss_exp[indices], k=2, s=0)
            #     I = Is[nbreak] * np.exp(-taus[-1]) + Sexp_sp.integral(taus[0], taus[-1])
            if len(taus_u) > 1:
                Sexp_sp = UnivariateSpline(taus[indices], Ss_exp[indices], k=1, s=0)
                I = Is[nbreak] * np.exp(-taus[-1]) + Sexp_sp.integral(taus[0], taus[-1])
            else:
                I = 0.0

            if np.isnan(I) or I < 0.:
                I = 0.0

        else:
            taus = taus[:nbreak + 1]
            I = 0.0

        # Iinit = Is[nbreak]

    # print 'Out of loop, subdivisions %s, stepsize = %.12f, i = %i, tau = %.2f,I0 = %.5f, I_init = %.5f, I=%.5f' % (subd,
    #     stepsize, nbreak, taus[-1], Is[0], Is[nbreak], I)

    return I, theta_new, phi_new, taus[-1]

def run(Mc, normal, pot, thetas, phis, scale, r0_scales, nekmin, angle_breaks, q, pot_range, les, chis_interp, Ss_interp, Is_interp):

    tan_st_1 = np.array([normal[1], -normal[0], 0.])
    tan_st_2 = np.cross(normal, tan_st_1)

    # Cartesian orthonormal unit vectors
    c1 = np.array([1., 0., 0.])
    c2 = np.array([0., 1., 0.])
    c3 = np.array([0., 0., 1.])

    # Roche normal orthonormal unit vectors
    n1 = tan_st_1 / np.sqrt(np.sum(tan_st_1 ** 2))
    n2 = tan_st_2 / np.sqrt(np.sum(tan_st_2 ** 2))
    n3 = normal / np.sqrt(np.sum(normal ** 2))

    # Transformation matrix direction cosines

    Q11 = np.dot(c1, n1)
    Q12 = np.dot(c1, n2)
    Q13 = np.dot(c1, n3)
    Q21 = np.dot(c2, n1)
    Q22 = np.dot(c2, n2)
    Q23 = np.dot(c2, n3)
    Q31 = np.dot(c3, n1)
    Q32 = np.dot(c3, n2)
    Q33 = np.dot(c3, n3)

    R = np.array([[Q11, Q12, Q13], [Q21, Q22, Q23], [Q31, Q32, Q33]])
    # xic = grid['rhos'][j] * grid['opacs'][j]
    # mfpc = grid['mfps'][j]
    Ic_j = np.zeros(thetas.shape)
    thetas_j = np.zeros(thetas.shape)
    phis_j = np.zeros(thetas.shape)
    taus_j = np.zeros(thetas.shape)

    args_sweep = np.arange(0, len(thetas), 1).astype(int)

    for arg in args_sweep:
    # for arg in [0]:
        # print 'Mc_ind, angles_ind: ', j, arg
        # start0 = timer()
        Ic_j[arg], thetas_j[arg], phis_j[arg], taus_j[arg] = compute_intensity_points(Mc, arg, q, nekmin, angle_breaks, thetas, phis,
                                                                        chis_interp, Ss_interp, Is_interp,
                                                                        scale, r0_scales, pot_range, les, R)
        # end0 = timer()
        # print end0 - start0

    return Ic_j, thetas_j, phis_j, taus_j
