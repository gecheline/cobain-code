import numpy as np
from cobain.structure import potentials
import logging
logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

def build_component(pot_s=3.76,pot_range=0.005,npot=50,ntheta=50,nphi=50,**kwargs):

    pots = np.linspace(pot_s, pot_s + pot_range, npot)
    thetas = np.linspace(0., np.pi / 2, ntheta)
    phis = np.linspace(0., np.pi, nphi)

    rs = np.zeros((len(pots) * len(thetas) * len(phis), 3))
    normals = np.zeros((len(pots) * len(thetas) * len(phis), 3))

    n = 0
    for i, pot in enumerate(pots):

        if kwargs['type'] == 'star':
            logging.info('Building equipotential surface at pot=%s' % pot)
            for j, theta in enumerate(thetas):
                for k, phi in enumerate(phis):
                    r = potentials.radius_pot_diffrot(pot, kwargs['bs'], theta)
                    r_cs = np.array(
                        [r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)])

                    # compute the normal to the surface

                    nx = potentials.dDiffRotRochedx(r_cs, kwargs['bs'])
                    ny = potentials.dDiffRotRochedy(r_cs, kwargs['bs'])
                    nz = potentials.dDiffRotRochedz(r_cs, kwargs['bs'])
                    nn = np.sqrt(nx * nx + ny * ny + nz * nz)

                    rs[n] = r_cs
                    normals[n] = np.array([nx / nn, ny / nn, nz / nn])

        elif kwargs['type'] == 'contact':
            logging.info('Building equipotential surface at pot=%s' % pot)
            for j, theta in enumerate(thetas):
                for k, phi in enumerate(phis):

                    # if theta > np.pi / 2. - self.angle_break1 and phi < self.angle_break1:
                    #     pass
                    # if theta > np.pi/2 or phi > np.pi:
                    #     pass
                    # else:
                    # print 'Creating primary point %i %i %i of %i, %i, %i' % (
                    # i, j, k, len(pots), len(phis), len(thetas))
                    rc = np.array(
                        (kwargs['rpole'] * np.sin(theta) * np.cos(phi), kwargs['rpole'] * np.sin(theta) * np.sin(phi),
                         kwargs['rpole'] * np.cos(theta)))

                    try:
                        # print 'ok'
                        r = potentials.radius_newton_tidal(rc, pot, kwargs['q'], kwargs['nekmin'])
                        # print r
                        if np.isnan(r):
                            r = 0.
                    except:
                        r = 0.

                    r_cs = np.array(
                        [r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)])
                    # compute the normal to the surface

                    nx = -potentials.dBinaryRochedx(r_cs, 1., kwargs['q'], 1.)
                    ny = -potentials.dBinaryRochedy(r_cs, 1., kwargs['q'], 1.)
                    nz = -potentials.dBinaryRochedz(r_cs, 1., kwargs['q'], 1.)
                    nn = np.sqrt(nx * nx + ny * ny + nz * nz)

                    rs[n] = r_cs
                    normals[n] = np.array([nx / nn, ny / nn, nz / nn])
                    n+=1

    return rs, normals

def build_mesh(pot_s=3.76,pot_range=0.005,npot=50,ntheta=50,scale=1.,**kwargs):

    nphi = kwargs['nphi']
    pots = np.linspace(pot_s, pot_s + pot_range, npot)
    thetas = np.linspace(0., np.pi / 2, ntheta)
    phis = np.linspace(0., np.pi, nphi)

    if kwargs['type'] == 'star':
        rs, normals = build_component(pot_s=pot_s,pot_range=pot_range,npot=npot,ntheta=ntheta,nphi=nphi,type=type,bs=kwargs['bs'])

    elif kwargs['type'] == 'contact':
        # print 'ok'
        q = kwargs['q']

        pot2 = pot_s / q + 0.5 * (q - 1) / q
        rs1, normals1 = build_component(pot_s=pot_s,pot_range=pot_range,npot=npot,ntheta=ntheta,nphi=nphi,type=kwargs['type'],q=q,rpole=kwargs['rpole1']/scale,
                                        nekmin=kwargs['nekmin1']/scale)
        rs2, normals2 = build_component(pot_s=pot2,pot_range=pot_range,npot=npot,ntheta=ntheta,nphi=nphi, type=kwargs['type'], q=1./q, rpole=kwargs['rpole2']/scale,
                                        nekmin=kwargs['nekmin2']/scale)

        rs2[:,0] = 1.-rs2[:,0]
        normals2[:,0] = -normals2[:,0]

        rs = np.vstack((rs1, rs2))
        normals = np.vstack((normals1, normals2))

    else:
        raise ValueError('Type can only be star or contact')

    mesh = {}
    mesh['pots'] = pots
    mesh['thetas'] = thetas
    mesh['phis'] = phis
    mesh['rs'] = rs*scale
    mesh['normals'] = normals

    return mesh





