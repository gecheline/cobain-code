import numpy as np
import os
import scipy.interpolate as spint
from cobain.structure import potentials
import logging
logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

def xnorm_to_xphys_cb(xnorm,xRB1,xRB2):

    # converts normalized to Roche x-values

    return xnorm*(xRB2-xRB1)+xRB1

def xphys_to_xnorm_cb(xphys,xRB1,xRB2):

    # converts Roche to normalized x-values

    return (xphys-xRB1)/(xRB2-xRB1)

def pots_to_ffs(pots,q):

    # converts potentials to fillout factors (used for interpolation to find rbacks)

    crit_pots = potentials.critical_pots(q)
    # print 'crit_pots', crit_pots
    return (pots-crit_pots['pot_L1'])/(crit_pots['pot_L2']-crit_pots['pot_L1'])

def get_rbacks(pots,q,comp):

    # find the value of the radius at the back of a star by interpolating in ff and q

    dir_local = os.path.dirname(os.path.abspath(__file__)) + '/'
    if comp == 1:
        rbacks = np.load(dir_local+'tables/rbacks_forinterp_negff_prim.npy')
    elif comp==2:
        rbacks = np.load(dir_local+'tables/rbacks_forinterp_negff_sec.npy')
    else:
        raise ValueError
    qs = rbacks[0].copy()[1:]
    # print qs.min(), qs.max()
    ffs = rbacks[:, 0].copy()[1:]
    rs_rowr = np.delete(rbacks, [0], axis=0)
    rs = np.delete(rs_rowr, [0], axis=1)

    RGI = spint.RegularGridInterpolator
    f = RGI(points=[qs, ffs], values=rs)
    qs_int = q*np.ones(len(pots))
    # print qs_int
    # print qs_int.min(), qs_int.max()
    ffs_int = pots_to_ffs(pots,q)
    # print ffs_int
    # print pots.min(), pots.max()
    # print ffs_int.min(), ffs_int.max()
    rs_int = f((qs_int, ffs_int))
    return rs_int

def find_root(coeffs):

    # finds the one root of a polynomial that is physical

    roots = np.roots(coeffs)
    reals_i = roots[np.isreal(roots)]
    # print reals_i

    if reals_i.size != 0:
        reals = np.real(reals_i)
        # print reals

        if np.all(reals >= 0.):
            # print 'all positive', reals
            return np.min(reals)
        elif np.all(reals < 0.):
            return np.nan
        else:
            reals_pos = reals[reals >= 0.]
            if reals_pos.size > 1:
                return np.max(reals_pos)
            else:
                return reals_pos[0]
    else:
        return np.nan

def find_r0(x,q,pot):

    # find the local polar radius, to be used as intial value for a given x

    P = ( pot - 0.5 * ( 1. + q ) * x ** 2. + q * x ) ** 2.
    A = P * ( 2. * x ** 2. - 2. * x + 1. ) - ( 1. + q ** 2.)
    B = ( x - 1. ) ** 2. * ( P * x ** 2. - 1. ) - x ** 2. * q ** 2.
    C = 4. * q ** 2.
    D = C * ( 2. * x ** 2. - 2. * x + 1. )
    E = C * x ** 2. * ( x - 1. ) ** 2.

    u = 2. * A / P
    v = ( 2. * B * P + A ** 2 - C ) / P ** 2.
    w = ( 2. * A * B - D ) / P ** 2.
    t = ( B ** 2. - E ) / P ** 2.

    # find the roots of the polynomial r^8 + u r^6 + v r^4 + w r^2 + t
    root = find_root((1,0,u,0,v,0,w,0,t))

    return root

def build_mesh(pot_s=3.76,pot_range=0.005,npot=50,ntheta=50,scale=1.,**kwargs):

    q = kwargs['q']
    nx = kwargs['nx']
    crit_pot = potentials.critical_pots(q)['pot_L1']

    if pot_s + pot_range > crit_pot:
        raise ValueError('Boundary potential cannot be higher than critical value %s.' % crit_pot)
    else:
        pots = np.linspace(pot_s,pot_s+pot_range,npot)
        pots2 = pots / q + 0.5 * (q - 1) / q
        xs = np.linspace(0.,1.,nx)
        thetas = np.linspace(0., np.pi/2,ntheta)

        rs = np.zeros((npot*nx*ntheta,3))
        normals = np.zeros((npot * nx * ntheta, 3))
        n = 0

        # print 'doing xrb1s'
        xrb1s = - get_rbacks(pots, q, 1)
        # print 'doing xrb2s'
        xrb2s = np.ones(len(pots)) + get_rbacks(pots, q, 2)

        for i,pot in enumerate(pots):

            for j,xnorm in enumerate(xs):
                # convert the normalized x to physical
                xphys = xnorm_to_xphys_cb(xnorm, xrb1s[i], xrb2s[i])
                if xnorm == 0. or xnorm==1.:
                    r0 = 0.
                else:
                    r0 = find_r0(xphys, q, pot)

                for k,theta in enumerate(thetas):

                    print 'pot %s, theta %s, x %s' % (i, j, k)
                    # find the true radius for the given theta
                    if xnorm == 0. or xnorm == 1.:
                        rho = 0.
                    else:
                        rho = potentials.radius_newton_cylindrical(r0, xphys, theta, pot, q)

                    # print 'rho at pot=%s, x=%s, theta=%s: %s' % (pot,xnorm,theta,rho)
                    if np.isnan(rho):
                        # now only happens at points where r0 = nan because it's probably = 0.
                        rs[n] = np.array([xphys, 0., 0.])
                        # print 'nan, r0', r0
                    else:
                        rs[n] = np.array([xphys, rho * np.sin(theta), rho * np.cos(theta)])

                    nx = -potentials.dBinaryRochedx(rs[n], 1., kwargs['q'], 1.)
                    ny = -potentials.dBinaryRochedy(rs[n], 1., kwargs['q'], 1.)
                    nz = -potentials.dBinaryRochedz(rs[n], 1., kwargs['q'], 1.)
                    nn = np.sqrt(nx * nx + ny * ny + nz * nz)

                    normals[n] = np.array([nx / nn, ny / nn, nz / nn])

                    n+=1

        mesh = {}
        mesh['pots'] = pots
        mesh['xnorms'] = xs
        mesh['thetas'] = thetas
        mesh['rs'] = rs*scale
        mesh['normals'] = normals

        return mesh
