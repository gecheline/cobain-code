import numpy as np
from cobain.structure import potentials
from cobain.meshes.cylindrical import get_rbacks, xphys_to_xnorm_cb
from cobain.structure import polytropes
from scipy.interpolate import UnivariateSpline
from cobain.structure.constants import *
import astropy.constants as const

def check_bounds(xnorms, thetas):
    thetas_cond = (thetas >= 0.) & (thetas <= np.pi/2)
    xs_cond = (xnorms >= 0.) & (xnorms <= 1.)
    return (thetas_cond) & (xs_cond)

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

def normalize_xs(xs, pots, q, scale=1.):

    pots2 = pots / q + 0.5 * (q - 1) / q

    xrb1s = - get_rbacks(pots, q, 1)*scale
    xrb2s = (np.ones(len(pots)) + get_rbacks(pots, q, 2))*scale
    # print 'xphys, xrb1s, xrb2s: ', xs, xrb1s, xrb2s

    return xphys_to_xnorm_cb(xs, xrb1s, xrb2s)

def return_structure_grid():
    return None

def return_structure_le(pots,q,le,r0_scale,stepsize=False):

    Ts = le['Tc'] * le['theta_interp_func'](r0_scale / (pots - q))
    rhos = le['rhoc'] * le['theta_interp_func'](r0_scale / (pots - q)) ** le['n']
    rhos_cgs = (rhos * const.M_sun / (const.R_sun) ** 3).value * 1e-3
    opac_si = polytropes.opal_sun_points(Ts, rhos_cgs)  # this opac is in m^2/kg

    opac = (opac_si * const.M_sun / const.R_sun ** 2).value
    Ss = (stefb / const.L_sun.value * const.R_sun.value ** 2) * Ts ** 4

    if stepsize:
        return opac*rhos
    else:
        return opac * rhos, Ss, Ss

def compute_structure_points(newpoints=[],scale=1.,q=1.,interp_funcs={},pot_range=[],nekmin=0.5,r0_scales=[],stepsize=False,arg=0):

    pots = np.zeros(len(newpoints))
    invalid1 = np.all(newpoints == 0., axis=1)
    invalid2 = (newpoints[:, 0] == scale) & (newpoints[:, 1] == 0.) & (newpoints[:, 2] == 0.)
    pots[~invalid1 & ~invalid2] = potentials.BinaryRoche_points(newpoints[~invalid1 & ~invalid2] / scale, q)
    pots[invalid1] = interp_funcs['le1']['pot_max']
    pots[invalid2] = interp_funcs['le2']['pot_max']

    # take care of pots that rounded up belong to grid but otherwise don't
    pots[(np.round(pots, 8) >= np.round(pot_range[0], 8)) & (pots < pot_range[0])] = pot_range[0]

    # newpoints are separated based on whether they belong to the grid, primary LE or secondary LE
    # this could be a separate function?

    prim_cond = (newpoints[:, 0] <= nekmin)
    sec_cond = (newpoints[:, 0] > nekmin)

    grid = np.argwhere((pots >= pot_range[0]) & (pots <= pot_range[1])).flatten()
    prim_le = np.argwhere(prim_cond & (pots > pot_range[1])).flatten()
    sec_le = np.argwhere(sec_cond & (pots > pot_range[1])).flatten()

    # the x coordinates of the grid points are normalized to allow for inteprolation in the grid
    xnorms_grid = normalize_xs(newpoints[:, 0][grid], pots[grid], q, scale=scale)
    # print 'newpoints: ',
    thetas_grid = np.abs(np.arcsin(newpoints[:,1][grid]/np.sqrt(np.sum((newpoints[:,1][grid]**2,newpoints[:,2][grid]**2),axis=0))))
    # thetas_grid_old = np.abs(np.arctan(newpoints[:, 1][grid]/newpoints[:, 2][grid]))
    # thetas_grid[thetas_grid > np.pi / 2] = rot_theta_points(thetas_grid[thetas_grid > np.pi / 2])
    thetas_grid[np.isnan(thetas_grid)] = 0.0

    # print 'thetas new old: ', thetas_grid, thetas_grid_old

    # interpolate with respect to where points lay on/off the grid
    # return_structure funcs take care of stuff here
    grid_cond = check_bounds(xnorms_grid, thetas_grid)
    if stepsize:
        chis = np.zeros(len(pots))
        chis[grid][grid_cond] = interp_funcs['chi']((pots[grid][grid_cond], xnorms_grid[grid_cond], thetas_grid[grid_cond]))
	chis[grid][~grid_cond] = np.zeros(len(pots[grid][~grid_cond]))
        chis[prim_le] = return_structure_le(pots[prim_le], q, interp_funcs['le1'], r0_scales[0], stepsize=stepsize)
        chis[sec_le] = return_structure_le(pots[sec_le] / q + 0.5 * (q - 1) / q, 1. / q, interp_funcs['le2'], r0_scales[1],
                                           stepsize=stepsize)

        return chis

    else:
        chis = np.zeros(len(pots))
        Ss = np.zeros(len(pots))
        Is = np.zeros(len(pots))

        chis[grid][grid_cond] = interp_funcs['chi']((pots[grid][grid_cond], xnorms_grid[grid_cond], thetas_grid[grid_cond]))
        Ss[grid][grid_cond] = interp_funcs['J']((pots[grid][grid_cond], xnorms_grid[grid_cond], thetas_grid[grid_cond]))
        Is[grid][grid_cond] = interp_funcs['I'][arg]((pots[grid][grid_cond], xnorms_grid[grid_cond], thetas_grid[grid_cond]))
        
	chis[grid][~grid_cond] = np.zeros(len(pots[grid][~grid_cond]))
	Ss[grid][~grid_cond] = np.zeros(len(pots[grid][~grid_cond]))
	Is[grid][~grid_cond] = np.zeros(len(pots[grid][~grid_cond]))

	chis[prim_le], Ss[prim_le], Is[prim_le] = return_structure_le(pots[prim_le], q, interp_funcs['le1'], r0_scales[0], stepsize=stepsize)
        chis[sec_le], Ss[sec_le], Is[sec_le] = return_structure_le(pots[sec_le] / q + 0.5 * (q - 1) / q, 1. / q, interp_funcs['le2'],
                                           r0_scales[1], stepsize=stepsize)

        return chis, Ss, Is


def adjust_stepsize_points(r=0., ndir=5., scale=1., pot_range=[], interp_funcs={}, nekmin=0.5, q=1., r0_scales=[]):

    exps = np.linspace(-10, -4, 1000)
    stepsizes = 10 ** exps

    b = np.ones((len(stepsizes), 3)).T * stepsizes
    newpoints = r - ndir * b.T

    # in case there are points in the centers of the stars (probably never for cylindrical)
    # make sure they get the maximum pot from LE value there

    chis = compute_structure_points(newpoints=newpoints,scale=scale,q=q, interp_funcs=interp_funcs,
                                    pot_range=pot_range,nekmin=nekmin,r0_scales=r0_scales,stepsize=True)
    # compute stepsize
    taus = chis * stepsizes / 2.

    diffs = np.abs(taus - 1.)
    stepsize_final = stepsizes[np.argmin(diffs)]

    # print 'stepsize: ', stepsize_final
    return stepsize_final / 2.

def compute_integral_break_I(N,paths,chis,Ss,Is):

    chis_sp = UnivariateSpline(paths, chis, k=1, s=0)
    taus = np.array([chis_sp.integral(paths[0], paths[i]) for i in range(N)])
    Ss_exp = Ss * np.exp(-taus)

    Iszero = np.argwhere(Is == 0.0).flatten()
    Ssezero = np.argwhere(Ss_exp == 0.0).flatten()

    if Iszero.size == 0 and Ssezero.size == 0:
        nbreak = N - 1
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

    return nbreak, taus, I

def compute_intensity_points(Mc=[0.,0.,0.], arg=0, q=1., nekmin=0.5, thetas=[], phis=[],
                                                          interp_funcs={}, scale=1., r0_scales=[],
                                                          pot_range=[], R=[]):
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
    phi_0 = np.abs(np.arctan2(Mc[1] / np.sqrt(np.sum(Mc ** 2)), Mc[0] / np.sqrt(np.sum(Mc ** 2))))

    # print 'theta_i, phi_i: ', theta_i, phi_i
    if theta_0 > np.pi / 2:
        theta_0 = rot_theta(theta_0)

    # rewrite logic here to go with cylindrical mesh

    stepsize = adjust_stepsize_points(r=Mc, ndir=ndir, scale=scale, pot_range=pot_range, interp_funcs=interp_funcs,
                                      nekmin=nekmin, q=q, r0_scales=r0_scales)

    # compute structure
    rs = np.array([Mc - i * stepsize * ndir for i in range(N)])
    paths = np.array([i * stepsize for i in range(N)])

    chis, Ss, Is = compute_structure_points(newpoints=rs, scale=scale, q=q, interp_funcs=interp_funcs,
                                    pot_range=pot_range, nekmin=nekmin, r0_scales=r0_scales, stepsize=False, arg=arg)

    nbreak, taus, I = compute_integral_break_I(N,paths,chis,Ss,Is)

    Iinit = Is[nbreak]
    subd = 'no'
    if (nbreak < 1000 and Iinit != 0.0) or nbreak == N-1:
        if (nbreak < 1000 and Iinit != 0.0):
            subd = 'yes'
            div = 1000. / nbreak
            stepsize = stepsize / div
        elif nbreak == N - 1:
            subd = 'yes'
            N = N * 2
        else:
            subd = 'yes'
            div = 1000. / nbreak
            stepsize = stepsize / div
            N = N * 2

        rs = np.array([Mc - i * stepsize * ndir for i in range(N)])
        paths = np.array([i * stepsize for i in range(N)])

        chis, Ss, Is = compute_structure_points(newpoints=rs, scale=scale, q=q, interp_funcs=interp_funcs,
                                                pot_range=pot_range, nekmin=nekmin, r0_scales=r0_scales, stepsize=False,
                                                arg=arg)

        nbreak, taus, I = compute_integral_break_I(N, paths, chis, Ss, Is)

    # print 'Out of loop, subdivisions %s, stepsize = %.12f, i = %i, tau = %.2f,I0 = %.5f, I_init = %.5f, I=%.5f' % (subd, stepsize, nbreak, taus[-1], Is[0], Is[nbreak], I)


    return I, taus[-1]


def compute_transformation_matrix(normal):

    # Cartesian orthonormal unit vectors
    c1 = np.array([1., 0., 0.])
    c2 = np.array([0., 1., 0.])
    c3 = np.array([0., 0., 1.])

    # Roche normal orthonormal unit vectors

    tan_st_1 = np.array([normal[1], -normal[0], 0.])
    tan_st_2 = np.cross(normal, tan_st_1)

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

    return R

def run(Mc=[0.,0.,0.], normal=[0.,0.,0.], sc_params={}, scale=1., q=1., interp_funcs={}, **kwargs):

    r0_scales=kwargs['r0_scales']
    nekmin=kwargs['nekmin']
    pot_range=kwargs['pot_range']
    thetas = sc_params['thetas']
    phis = sc_params['phis']

    # compute the transformation matrix
    R = compute_transformation_matrix(normal)

    # prepare arrays for computations
    Ic_j = np.zeros(thetas.shape)
    taus_j = np.zeros(thetas.shape)

    args_sweep = np.arange(0, len(thetas), 1).astype(int)

    for arg in args_sweep:
    # for arg in [0]:
        # print 'Mc_ind, angles_ind: ', j, arg
        # start0 = timer()
        Ic_j[arg], taus_j[arg] = compute_intensity_points(Mc=Mc, arg=arg, q=q, nekmin=nekmin, thetas=thetas, phis=phis,
                                                          interp_funcs=interp_funcs, scale=scale, r0_scales=r0_scales,
                                                          pot_range=pot_range, R=R)
        # end0 = timer()
        # print end0 - start0

    return Ic_j, taus_j
