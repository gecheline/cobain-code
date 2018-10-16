import logging
import os
import pickle
import random

import astropy.constants as const
import numpy as np
import quadpy
import scipy.interpolate as spint
from scipy.interpolate import interp1d

from cobain.structure import polytropes
from cobain.structure import potentials
from cobain.structure.constants import *
from cobain.structure.potentials import nekmin

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

class Contact_Binary(object):

    def __init__(self, mass1=1.0, q=1.0, ff=0.1, pot_range=0.01, geometry = 'cylindrical', dims=np.array([50,150,50]),
                n1=3.0, n2=3.0, lebedev_ndir=5, dir=os.getcwd()):

        self.directory = dir

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.q = q
        self.q2 = 1./q
        self.mass1 = mass1
        self.mass2 = q*mass1
        zeta1, nu1 = polytropes.return_MS_factors(self.mass1)
        zeta2, nu2 = polytropes.return_MS_factors(self.mass2)
        crit_pots = potentials.critical_pots(self.q)
        self.crit_pot = crit_pots['pot_L1']

        self.teff1 = self.mass1 ** (0.25 * (nu1 + 1. / (2 * zeta1))) * 5777
        self.teff2 = self.mass2 ** (0.25 * (nu2 + 1. / (2 * zeta2))) * 5777
        self.pot = crit_pots['pot_L1'] - ff * (crit_pots['pot_L1'] - crit_pots['pot_L2'])
        self.pot2 = self.pot / self.q + 0.5 * (self.q - 1) / self.q

        # with sma=1 (still not computed) the equivalent radii will be the k-factors needed for the polytropic solution
        self.k1 = potentials.compute_equivalent_radius_tidal(pot = self.pot, q=self.q)
        self.k2 = potentials.compute_equivalent_radius_tidal(pot = self.pot2, q=self.q2)

        # # these radii are only for computation of the central values, assuming they stay the same as a normal MS single star of mass M
        self.R1 = self.mass1 ** zeta1
        self.R2 = self.mass2 ** zeta2

        self.polytropic_index1 = n1
        self.polytropic_index2 = n2
        self.dims = dims # dims are now pot, theta, phi
        self.pot_range = np.array([self.pot,self.pot+pot_range])

        # setup short characteristics parameters
        points = quadpy.sphere.Lebedev(lebedev_ndir).points
        weights = quadpy.sphere.Lebedev(lebedev_ndir).weights

        thetas = np.arccos(points[:, 2])
        phis = np.arctan2(points[:, 1], points[:, 0])

        self.ndir = lebedev_ndir
        self.nI = len(points)
        self.sc_params = {'thetas': thetas, 'phis': phis, 'ws': weights}

        le1 = polytropes.LE_tidal(self.R1, self.mass1, self.teff1, self.k1, self.q, self.polytropic_index1)
        le2 = polytropes.LE_tidal(self.R2, self.mass2, self.teff2, self.k2, self.q2, self.polytropic_index2)
        le1['pot_max'] = 1./(le1['theta_interp'][0][0]) + self.q
        le2['pot_max'] = 1./(le2['theta_interp'][0][0]) + self.q2

        self.le1 = le1
        self.le2 = le2
        self.pot1_tau = polytropes.find_opdepth_eq1_contact(le1, self.teff1, self.q)
        self.pot2_tau = polytropes.find_opdepth_eq1_contact(le2, self.teff2, self.q2)
        # set the scale so that potx_tau = user provided pot
        r01_tau = 1./(self.pot1_tau - self.q)
        r02_tau = 1./(self.pot2_tau - self.q2)
        r01_pot = 1./(self.pot - self.q)
        r02_pot = 1./(self.pot2 - self.q2)

        self.r0_scale1 = r01_tau/r01_pot # this scales the structure from polytropes to the outer potential
        self.r0_scale2 = r02_tau/r02_pot # this scales the structure from polytropes to the outer potential

        self.pot_r1 = 1. / (le1['r0_s']/self.r0_scale1) + self.q
        self.pot_r2 = 1./ (le2['r0_s']/self.r0_scale2) + self.q2
        sma1 = self.R1/potentials.compute_equivalent_radius_tidal(pot = self.pot_r1, q = self.q)
        sma2 = self.R2/potentials.compute_equivalent_radius_tidal(pot = self.pot_r2, q = self.q2)
        self.scale = 0.5*(sma1 + sma2) # this will scale the physical quantities to the dimensionless ones

        # find nekmin and the sizes of the box (probably from rpole and req from Kopal's approximations)
        self.req1 = potentials.radius_pot_contact_approx(self.pot, self.q, -1., 0.)*self.scale
        self.rpole1 = potentials.radius_pot_contact_approx(self.pot, self.q, 0., 1.)*self.scale
        self.req2 = potentials.radius_pot_contact_approx(self.pot2, self.q2, -1., 0.)*self.scale
        self.rpole2 = potentials.radius_pot_contact_approx(self.pot2, self.q2, 0., 1.)*self.scale

        xz, z = nekmin(self.pot, self.q, crit_pots['xL1'], 0.05)
        self.nekmin1 = xz*self.scale
        self.nekmin_z = z*self.scale
        self.nekmin2 = self.scale - self.nekmin1

        self.type = 'contact'
        self.geometry = geometry

    def spherical_mesh(self):

        rs, normals, pots, thetas, phis, rhos = self.create_mesh()

        self.mesh = {}
        self.mesh['rs'] = rs
        self.mesh['normals'] = normals
        self.mesh['pots'] = pots
        self.mesh['thetas'] = thetas
        self.mesh['phis'] = phis
        self.mesh['rhos'] = rhos

        # compute theta and phi coordinates of neck per potential
        dimlen = self.dims[0] * self.dims[1] * self.dims[2]
        potlen = self.dims[1] * self.dims[2]
        breaks1 = np.zeros((self.dims[0], 3))
        breaks2 = np.zeros((self.dims[0], 3))

        for i in range(self.dims[0]):
            pot_slice1 = self.mesh['rs'][i * potlen: (i + 1) * potlen]
            pot_slice2 = self.mesh['rs'][dimlen + i * potlen: dimlen + (i + 1) * potlen]

            argzeros1 = np.argwhere(np.all(pot_slice1 == 0., axis=1))
            argzeros2 = np.argwhere(
                (pot_slice2[:, 0] == self.scale) & (pot_slice2[:, 1] == 0.0) & (pot_slice2[:, 2] == 0.0))

            thetas_break1 = self.mesh['thetas'][argzeros1 / dims[1]]
            phis_break1 = self.mesh['phis'][argzeros1 % dims[1]]

            thetas_break2 = self.mesh['thetas'][argzeros2 / dims[1]]
            phis_break2 = self.mesh['phis'][argzeros2 % dims[1]]

            if thetas_break1.size == 0:
                th1min = np.pi / 2
            else:
                th1min = thetas_break1.min()

            if phis_break1.size == 0:
                ph1max = 0.
            else:
                ph1max = phis_break1.max()

            if thetas_break2.size == 0:
                th2min = np.pi / 2
            else:
                th2min = thetas_break2.min()

            if phis_break2.size == 0:
                ph2max = 0.
            else:
                ph2max = phis_break2.max()

            breaks1[i] = np.array([self.mesh['pots'][i], th1min, ph1max])
            breaks2[i] = np.array([self.mesh['pots'][i], th2min, ph2max])

        self.breaks1 = breaks1
        self.breaks2 = breaks2

        self.rescale_factor = np.average(self.rescale_factors())
        self.pickle(self.directory + 'body')


    def area_volume(self, q, D, pot):
        r0 = 1. / (pot - q)
        n = 0.5 * (q + 1.)

        V = 4. * np.pi / 3. * (D) ** 3 * r0 ** 3 * (
            1. + 2. * n * r0 ** 3 + (
            12. / 5. * q ** 2 + 8. / 5. * n * q + 32. / 5 * n ** 2) * r0 ** 6 + 15. / 7 * q ** 2 * r0 ** 8 + 18. / 9. * q ** 2 * r0 ** 10)

        S = 4. * np.pi * (D) ** 2 * r0 ** 2 * (
            1. + 4. * n / 3 * r0 ** 3 + (
            7. / 5. * q ** 2 + 14. / 15. * n * q + 56. / 15. * n ** 2) * r0 ** 6 + 9. / 7. * q ** 2 * r0 ** 8 + 11. / 9. * q ** 2 * r0 ** 10)

        return {'volume': V, 'area': S}

    def create_mesh(self):

        theta_interp1 = interp1d(self.le1['theta_interp'][0], self.le1['theta_interp'][1], fill_value='extrapolate')
        theta_interp2 = interp1d(self.le2['theta_interp'][0], self.le2['theta_interp'][1], fill_value='extrapolate')

        pots = np.linspace(self.pot_range[0], self.pot_range[1], self.dims[0])
        thetas = np.linspace(0., np.pi / 2, self.dims[1])
        phis = np.linspace(0., np.pi, self.dims[2])

        rs1 = np.zeros((len(pots) * len(thetas) * len(phis), 3))
        rhos1 = np.zeros(len(pots) * len(thetas) * len(phis))
        normals1 = np.zeros((len(pots) * len(thetas) * len(phis), 3))
        T1 = np.zeros((len(pots), len(thetas), len(phis)))
        chi1 = np.zeros((len(pots), len(thetas), len(phis)))
        S1 = np.zeros((len(pots), len(thetas), len(phis)))
        I1 = np.zeros((len(pots), len(thetas), len(phis)))

        rs2 = np.zeros((len(pots) * len(thetas) * len(phis), 3))
        rhos2 = np.zeros(len(pots) * len(thetas) * len(phis))
        normals2 = np.zeros((len(pots) * len(thetas) * len(phis), 3))
        T2 = np.zeros((len(pots), len(thetas), len(phis)))
        chi2 = np.zeros((len(pots), len(thetas), len(phis)))
        S2 = np.zeros((len(pots), len(thetas), len(phis)))
        I2 = np.zeros((len(pots), len(thetas), len(phis)))

        n = 0
        for i, pot in enumerate(pots):
            logging.info('Building primary equipotential surface at pot=%s' % pot)
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
                        (self.rpole1 * np.sin(theta) * np.cos(phi), self.rpole1 * np.sin(theta) * np.sin(phi),
                         self.rpole1 * np.cos(theta))) / self.scale

                    try:
                        r = potentials.radius_newton_tidal(rc, pot, self.q, self.nekmin1) * self.scale

                        if np.isnan(r):
                            r = 0.
                    except:
                        r=0.

                    r_cs = np.array(
                        [r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)])
                    # compute the normal to the surface

                    nx = -potentials.dBinaryRochedx(r_cs / self.scale, 1., self.q, 1.)
                    ny = -potentials.dBinaryRochedy(r_cs / self.scale, 1., self.q, 1.)
                    nz = -potentials.dBinaryRochedz(r_cs / self.scale, 1., self.q, 1.)
                    nn = np.sqrt(nx * nx + ny * ny + nz * nz)

                    rs1[n] = r_cs
                    normals1[n] = np.array([nx / nn, ny / nn, nz / nn])

                    theta_pot = theta_interp1(self.r0_scale1 / (pot - self.q))
                    T1[i, j, k], chi1[i, j, k], rhos1[n], S1[i, j, k], I1[i, j, k] = self.structure_per_point(
                        theta_pot, self.polytropic_index1, self.le1['Tc'], self.le1['rhoc'])

                    n += 1

        n = 0
        for i, pot in enumerate(pots):
            logging.info('Building secondary equipotential surface at pot=%s' % pot)
            for j, theta in enumerate(thetas):
                for k, phi in enumerate(phis):
                    # if theta > np.pi / 2. - self.angle_break2 and phi < self.angle_break2:
                    #     pass
                    # else:
                    # print 'Creating secondary point %i %i %i of %i,%i,%i' % (
                    #     i, j, k, len(pots), len(thetas), len(phis))
                    rc = np.array(
                        (self.rpole2 * np.sin(theta) * np.cos(phi),
                         self.rpole2 * np.sin(theta) * np.sin(phi),
                         self.rpole2 * np.cos(theta))) / self.scale

                    pot2 = pot / self.q + 0.5 * (self.q - 1) / self.q
                    try:
                        r = potentials.radius_newton_tidal(rc, pot2, self.q2, self.nekmin2) * self.scale

                        if np.isnan(r):
                            r = 0.
                    except:
                        r=0.

                    r_cs = np.array(
                        [r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)])
                    # compute the normal to the surface

                    nx = -potentials.dBinaryRochedx(r_cs / self.scale, 1., self.q2, 1.)
                    ny = -potentials.dBinaryRochedy(r_cs / self.scale, 1., self.q2, 1.)
                    nz = -potentials.dBinaryRochedz(r_cs / self.scale, 1., self.q2, 1.)
                    nn = np.sqrt(nx * nx + ny * ny + nz * nz)

                    rs2[n] = np.array([self.scale - r_cs[0], r_cs[1], r_cs[2]])
                    normals2[n] = np.array([-nx / nn, ny / nn, nz / nn])

                    theta_pot = theta_interp2(self.r0_scale2 / (pot2 - self.q2))
                    T2[i, j, k], chi2[i, j, k], rhos2[n], S2[i, j, k], I2[i, j, k] = self.structure_per_point(
                        theta_pot,
                        self.polytropic_index2,
                        self.le2['Tc'],
                        self.le2['rhoc'])
                    n += 1

        rs = np.vstack((rs1, rs2))
        normals = np.vstack((normals1, normals2))
        rhos = np.hstack((rhos1, rhos2))
        np.save(self.directory + 'T1_0', T1)
        np.save(self.directory + 'T2_0', T2)
        # np.save(self.directory + 'Tn_0', T_neck)
        np.save(self.directory + 'chi1_0', chi1)
        np.save(self.directory + 'chi2_0', chi2)
        # np.save(self.directory + 'chin_0', chi_neck)
        np.save(self.directory + 'J1_0', S1)
        np.save(self.directory + 'J2_0', S2)
        # np.save(self.directory + 'Jn_0', S_neck)

        for i in range(len(self.sc_params['thetas'])):
            np.save(self.directory + 'I1_0_' + str(int(i)), I1)
            np.save(self.directory + 'I2_0_' + str(int(i)), I2)
            # np.save(self.directory + 'In_0_' + str(int(i)), I_neck)

        return rs, normals, pots, thetas, phis, rhos
        # create two lobes separately, if diverging near the neck, use approx radius


    def structure_per_point(self,theta_pot,n,Tc,rhoc):

        T = Tc * theta_pot
        rho = rhoc * theta_pot ** n

        rho_cgs = (rho * const.M_sun / (const.R_sun) ** 3).value * 1e-3
        opac_si = polytropes.opal_sun(T, rho_cgs)[0]  # this opac is in m^2/kg
        opac = (opac_si * const.M_sun / const.R_sun ** 2).value

        chi = opac * rho
        J = (1./np.pi) * (stefb / const.L_sun.value * const.R_sun.value ** 2) * T ** 4

        I = J

        return T,chi,rho,J,I

    def sweep_mesh(self,points,iter_n=1):

        RGI = spint.RegularGridInterpolator

        chi1 = np.load(self.directory + 'chi1_'+str(iter_n-1)+'.npy')
        chi2 = np.load(self.directory + 'chi2_' + str(iter_n - 1)+'.npy')
        # chi_neck = np.load(self.directory + 'chin_' + str(iter_n - 1)+'.npy')
        J1 = np.load(self.directory + 'J1_'+str(iter_n-1)+'.npy')
        J2 = np.load(self.directory + 'J2_' + str(iter_n - 1)+'.npy')
        # J_neck = np.load(self.directory + 'Jn_' + str(iter_n - 1) + '.npy')


        chi1_interp = RGI(points=[self.mesh['pots'], self.mesh['thetas'], self.mesh['phis']], values=chi1)
        chi2_interp = RGI(points=[self.mesh['pots'], self.mesh['thetas'], self.mesh['phis']], values=chi2)
        # chin_interp = RGI(points=[self.mesh['xs'], self.mesh['ys'], self.mesh['zs']], values=chi_neck)
        J1_interp = RGI(points=[self.mesh['pots'], self.mesh['thetas'], self.mesh['phis']], values=J1)
        J2_interp = RGI(points=[self.mesh['pots'], self.mesh['thetas'], self.mesh['phis']], values=J2)
        # Jn_interp = RGI(points=[self.mesh['xs'], self.mesh['ys'], self.mesh['zs']], values=J_neck)

        I1_interp = []
        I2_interp = []
        # In_interp = []


        for i in range(len(self.sc_params['thetas'])):
            I1 = np.load(self.directory + 'I1_' + str(iter_n - 1) + '_' + str(int(i)) + '.npy')
            I1_interp.append(RGI(points=[self.mesh['pots'], self.mesh['thetas'], self.mesh['phis']], values=I1))

            I2 = np.load(self.directory + 'I2_' + str(iter_n - 1) + '_' + str(int(i)) + '.npy')
            I2_interp.append(RGI(points=[self.mesh['pots'], self.mesh['thetas'], self.mesh['phis']], values=I2))

            # I_neck = np.load(self.directory + 'In_' + str(iter_n - 1) + '_' + str(int(i)) + '.npy')
            # In_interp.append(RGI(points=[self.mesh['xs'], self.mesh['ys'], self.mesh['zs']], values=I_neck))

        # interpolate the Lane-Emden solution as well to compute points inside the star but outside of the mesh

        le1 = self.le1.copy()
        le2 = self.le2.copy()

        le1['theta_interp_func'] = interp1d(self.le1['theta_interp'][0],self.le1['theta_interp'][1], fill_value='extrapolate')
        le2['theta_interp_func'] = interp1d(self.le2['theta_interp'][0], self.le2['theta_interp'][1], fill_value='extrapolate')

        # setup the arrays that the computation will output
        I_new = np.zeros((len(points), self.nI))
        thetas_new = np.zeros((len(points), self.nI))
        phis_new = np.zeros((len(points), self.nI))
        taus_new = np.zeros((len(points), self.nI))

        for j, indx in enumerate(points):
            # print 'Entering rt computation'
            logging.info('Computing intensities for point %i of %i' % (j+1, len(points)))
            r = self.mesh['rs'][indx]
            if np.all(r==0.) or (r[0]==self.scale and r[1]==0. and r[2]==0.):
                pass
            else:
                #r_abs = np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
                potc = potentials.BinaryRoche(r / self.scale, self.q)
                I_new[j], thetas_new[j], phis_new[j], taus_new[j] = cobain.radiative_transfer.contact_binary.spherical.run(r, self.mesh['normals'][indx],
                                                                                                                           potc, self.sc_params['thetas'],
                                                                                                                           self.sc_params['phis'], self.scale, [self.r0_scale1, self.r0_scale2], self.nekmin1,
                                                                                                                           [self.breaks1, self.breaks2],
                                                                                                                           self.q, self.pot_range, [le1, le2],
                                                                                                                           [chi1_interp, chi2_interp], [J1_interp, J2_interp],
                                                                                                                           [I1_interp, I2_interp],
                                                                                                                           )

        return I_new, thetas_new, phis_new

    def conserve_energy(self, I_new, points):

        J = np.zeros(len(points))
        I_em = np.zeros(len(points))
        F = np.zeros(len(points))
        T = np.zeros(len(points))
        chi = np.zeros(len(points))

        ws = self.sc_params['ws']

        for i, indx in enumerate(points):

            J[i] = np.sum(ws * I_new[i])

            cond_out = self.sc_params['thetas'] <= np.pi/2
            cond_in = self.sc_params['thetas'] > np.pi/2

            #I_em[i] = 0.5*np.pi*np.sum(ws[cond_out] * I_new[i][cond_out])
            F[i] = 2 * np.pi *np.sum(ws[cond_out]*I_new[i][cond_out]*np.cos(self.sc_params['thetas'][cond_out])) - \
                   2 * np.pi *np.sum(ws[cond_in]*I_new[i][cond_in]*np.cos(self.sc_params['thetas'][cond_in]))
            # F[i] = np.sum(ws[cond_out] * I_new[i][cond_out]) - np.sum(ws[cond_in] * I_new[i][cond_in])

            rho = self.mesh['rhos'][indx]

            # T[i] = (F[i]/stefb_sol)**0.25
            # T[i] = self.teff_cb * (F[i] / self.flux) ** 0.25
            T[i] = (J[i] * np.pi / (stefb / const.L_sun.value * const.R_sun.value ** 2))**0.25

            if T[i] == 0.0:
                chi[i] = 0.0

            else:
                rho_cgs = (rho * const.M_sun / (const.R_sun) ** 3).value * 1e-3
                opac_si = polytropes.opal_sun(T[i], rho_cgs)[0]  # this opac is in m^2/kg
                opac_new = (opac_si * const.M_sun / const.R_sun ** 2).value
                chi[i] = opac_new * rho

        return J, T, chi

    def conserve_luminosity(self,F1,F2,luminosity):

        dtheta = (np.pi/2)/self.dims[1]
        dphi = np.pi/self.dims[2]
        comp_len = self.dims[0]*self.dims[1]*self.dims[2]

        scales = np.zeros(self.dims[0])
        Ss_i = np.zeros((self.dims[0],2*self.dims[1]*self.dims[2]))
        Fs_i = np.zeros((self.dims[0],2*self.dims[1]*self.dims[2]))
        Ls = np.zeros(self.dims[0])

        for i in range(self.dims[0]):
            rs1 = self.mesh['rs'][i*self.dims[1]*self.dims[2]:(i+1)*self.dims[1]*self.dims[2]]
            rs2 = self.mesh['rs'][comp_len+i*self.dims[1]*self.dims[2]:comp_len+(i+1)*self.dims[1]*self.dims[2]]
            rs2[:,0] = self.scale - rs2[:,0]
            rs = np.vstack((rs1, rs2))

            normals1 = self.mesh['normals'][i*self.dims[1]*self.dims[2]:(i+1)*self.dims[1]*self.dims[2]]
            normals2 = self.mesh['normals'][comp_len+i * self.dims[1] * self.dims[2]:comp_len+(i + 1) * self.dims[1] * self.dims[2]]
            normals2[:,0] = -normals2[:,0]
            normals = np.vstack((normals1,normals2))

            pots = np.ones(len(rs))*self.mesh['pots'][i]
            thetas1 = np.zeros(len(rs1))
            phis1 = np.zeros(len(rs1))
            thetas2 = np.zeros(len(rs2))
            phis2 = np.zeros(len(rs2))

            for j in range(self.dims[1]):
                thetas1[j*self.dims[2]:(j+1)*self.dims[2]] = self.mesh['thetas'][j]
                thetas2[j * self.dims[2]:(j + 1) * self.dims[2]] = self.mesh['thetas'][j]
                for k in range(self.dims[2]):
                    # print 'i,j,k:', i,j,k
                    # print 'phis n: ', k+j*self.dims[2]
                    phis1[k+j*self.dims[2]] = self.mesh['phis'][k]
                    phis2[k + j * self.dims[2]] = self.mesh['phis'][k]

            thetas = np.hstack((thetas1,thetas2))
            phis = np.hstack((phis1,phis2))

            rabs = np.sum(rs**2,axis=1)**0.5
            cos_gammas = np.zeros(len(rs))

            for l,r in enumerate(rs):
                cos_gammas[l] = np.dot(r/rabs[l],normals[l])

            dSs = dtheta * dphi * rabs**2 * np.sin(thetas) / cos_gammas
            dSs[(np.isnan(dSs)) | (np.isinf(dSs))] = 0.0
            Ss_i[i] = dSs
            Fs1 = F1[i*self.dims[1]*self.dims[2]:(i+1)*self.dims[1]*self.dims[2]]
            Fs2 = F2[i * self.dims[1] * self.dims[2]:(i + 1) * self.dims[1] * self.dims[2]]
            Fs = np.hstack((Fs1,Fs2))
            Fs_i[i] = Fs
            Ls[i] = 4*np.sum(dSs*Fs) # the 4 comes from the fact that we technically only compute 1/4 of the star
            scales[i] = luminosity/(np.sum(4*dSs*Fs))

        return rs, normals, cos_gammas, Ss_i, Fs_i, Ls, scales

    def rescale_factors(self):

        points = random.sample(range(int(0.5*self.dims[0]*self.dims[1]*self.dims[2]), int(0.9*self.dims[0]*self.dims[1]*self.dims[2])), 5)

        rescale_factors = np.zeros(len(self.sc_params['thetas']))
        Is, thetas, phis = self.sweep_mesh(points, iter_n=1)

        for l in range(len(self.sc_params['thetas'])):
            Il = np.load(self.directory+'I1_0_'+str(int(l))+'.npy').flatten()
            mask = (Is[:,l]!=0.0)
            rescale_factors[l] = np.average(Il[points][mask]/Is[:,l][mask])

        return rescale_factors

    def pickle(self, filename):
        f = file(filename, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    @staticmethod
    def unpickle(filename):
        with file(filename, 'rb') as f:
            return pickle.load(f)
