import logging
import os
import pickle
import random

import astropy.constants as const
import numpy as np
import quadpy
import scipy.interpolate as spint
from scipy.interpolate import interp1d

from cobain.meshes import cylindrical, spherical
from cobain.structure import polytropes
from cobain.structure import potentials
from cobain.structure.constants import *
from cobain.structure.potentials import nekmin
from cobain.radiative_transfer.contact_binary import spherical as rt_spherical
from cobain.radiative_transfer.contact_binary import cylindrical as rt_cylindrical

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

class Contact_Binary(object):


    def __init__(self, mass1=1.0, q=1.0, ff=0.1, pot_range=0.01, geometry = 'cylindrical', dims=np.array([5,5,15]),
                n1=3.0, n2=3.0, lebedev_ndir = 5, dir=os.getcwd()+'/'):

        self.directory = dir

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.q = q
        self.ff = ff
        self.q2 = 1./q
        self.mass1 = mass1
        self.mass2 = q*mass1
        zeta1, nu1 = polytropes.return_MS_factors(self.mass1)
        zeta2, nu2 = polytropes.return_MS_factors(self.mass2)
        crit_pots = potentials.critical_pots(self.q)
        self.crit_pots = crit_pots

        self.teff1 = self.mass1 ** (0.25 * (nu1 + 1. / (2 * zeta1))) * 5777
        self.teff2 = self.mass2 ** (0.25 * (nu2 + 1. / (2 * zeta2))) * 5777
        self.pot = self.crit_pots['pot_L1'] - self.ff * (self.crit_pots['pot_L1'] - self.crit_pots['pot_L2'])
        self.pot2 = self.pot / self.q + 0.5 * (self.q - 1) / self.q


        # # these radii are only for computation of the central values, assuming they stay the same as a normal MS single star of mass M
        self.R1 = self.mass1 ** zeta1
        self.R2 = self.mass2 ** zeta2

        self.polytropic_index1 = n1
        self.polytropic_index2 = n2
        self.dims = dims # dims are now pot, theta, phi or pot, theta, nx
        self.pot_range = pot_range

        self.type = 'contact'
        self.geometry = geometry

        self.initialize_le_params()
        self.initialize_rt_setup(lebedev_ndir=lebedev_ndir)


    def build_mesh(self):

        self.create_mesh()
        self.populate_mesh()
        self.finalize_mesh()



    def initialize_le_params(self):

        # with sma=1 (still not computed) the equivalent radii will be the k-factors needed for the polytropic solution
        self.k1 = potentials.compute_equivalent_radius_tidal(pot = self.pot, q=self.q)
        self.k2 = potentials.compute_equivalent_radius_tidal(pot = self.pot2, q=self.q2)

        le1 = polytropes.LE_tidal(self.R1, self.mass1, self.teff1, self.k1, self.q, self.polytropic_index1)
        le2 = polytropes.LE_tidal(self.R2, self.mass2, self.teff2, self.k2, self.q2, self.polytropic_index2)
        le1['pot_max'] = 1. / (le1['theta_interp'][0][0]) + self.q
        le2['pot_max'] = 1. / (le2['theta_interp'][0][0]) + self.q2

        self.le1 = le1
        self.le2 = le2
        self.pot1_tau = polytropes.find_opdepth_eq1_contact(le1, self.teff1, self.q)
        self.pot2_tau = polytropes.find_opdepth_eq1_contact(le2, self.teff2, self.q2)
        # set the scale so that potx_tau = user provided pot
        r01_tau = 1. / (self.pot1_tau - self.q)
        r02_tau = 1. / (self.pot2_tau - self.q2)
        r01_pot = 1. / (self.pot - self.q)
        r02_pot = 1. / (self.pot2 - self.q2)

        self.r0_scale1 = r01_tau / r01_pot  # this scales the structure from polytropes to the outer potential
        self.r0_scale2 = r02_tau / r02_pot  # this scales the structure from polytropes to the outer potential

        self.pot_r1 = 1. / (le1['r0_s'] / self.r0_scale1) + self.q
        self.pot_r2 = 1. / (le2['r0_s'] / self.r0_scale2) + self.q2
        sma1 = self.R1 / potentials.compute_equivalent_radius_tidal(pot=self.pot_r1, q=self.q)
        sma2 = self.R2 / potentials.compute_equivalent_radius_tidal(pot=self.pot_r2, q=self.q2)
        self.scale = 0.5 * (sma1 + sma2)  # this will scale the physical quantities to the dimensionless ones

        self.req1 = potentials.radius_pot_contact_approx(self.pot, self.q, -1., 0.) * self.scale
        self.rpole1 = potentials.radius_pot_contact_approx(self.pot, self.q, 0., 1.) * self.scale
        self.req2 = potentials.radius_pot_contact_approx(self.pot2, self.q2, -1., 0.) * self.scale
        self.rpole2 = potentials.radius_pot_contact_approx(self.pot2, self.q2, 0., 1.) * self.scale

        xz, z = nekmin(self.pot, self.q, self.crit_pots['xL1'], 0.05)
        self.nekmin1 = xz * self.scale
        self.nekmin_z = z * self.scale
        self.nekmin2 = self.scale - self.nekmin1


    def initialize_rt_setup(self,lebedev_ndir):

        # setup short characteristics parameters
        points = quadpy.sphere.Lebedev(lebedev_ndir).points
        weights = quadpy.sphere.Lebedev(lebedev_ndir).weights

        thetas = np.arccos(points[:, 2])
        phis = np.arctan2(points[:, 1], points[:, 0])

        self.ndir = lebedev_ndir
        self.nI = len(points)
        self.sc_params = {'thetas': thetas, 'phis': phis, 'ws': weights}

    def create_mesh(self):

        if self.geometry == 'cylindrical':
            self.mesh = cylindrical.build_mesh(pot_s=self.pot, pot_range=self.pot_range, npot=self.dims[0], ntheta=self.dims[2], nx=self.dims[1], q=self.q, scale=self.scale)

        elif self.geometry == 'spherical':
            self.mesh = spherical.build_mesh(pot_s=self.pot, pot_range=self.pot_range, npot=self.dims[0], ntheta=self.dims[1], nphi=self.dims[2],
                                        q=self.q, type='contact', rpole1=self.rpole1, rpole2=self.rpole2, nekmin1=self.nekmin1,
                                        nekmin2=self.nekmin2, scale=self.scale)

        else:
            raise ValueError('Geometry %s not supported, can only be one of cylindrical or spherical' % self.geometry)


    def populate_mesh(self):

        potlen = self.dims[1]*self.dims[2]
        dimlen = self.dims[0]*self.dims[1]*self.dims[2]
        arrlen = len(self.mesh['rs'])

        pots2 = self.mesh['pots'] / self.q + 0.5 * (self.q - 1) / self.q
        theta_interp1 = interp1d(self.le1['theta_interp'][0], self.le1['theta_interp'][1], fill_value='extrapolate')
        theta_interp2 = interp1d(self.le2['theta_interp'][0], self.le2['theta_interp'][1], fill_value='extrapolate')

        # separate stars based on coordinates for initial population
        if self.geometry == 'spherical':
            pargs = np.arange(0,dimlen,1).astype(int)
            sargs = np.arange(dimlen,arrlen,1).astype(int)
        
        elif self.geometry == 'cylindrical':
            pargs = np.argwhere(self.mesh['rs'][:,0] <= self.nekmin1).flatten()
            sargs = np.argwhere(self.mesh['rs'][:,0] > self.nekmin1).flatten()

        else:
            raise ValueError('Geometry %s not supported' % self.geometry)
        
        pots_rs = np.zeros(arrlen)

        for j, pot in enumerate(self.mesh['pots']):
            if self.geometry == 'cylindrical':
                pots_rs[j*potlen:(j+1)*potlen] = pot

            elif self.geometry == 'spherical':
                pots_rs[j * potlen:(j + 1) * potlen] = pot
                pots_rs[dimlen+j * potlen:dimlen+(j + 1) * potlen] = pot
            else:
                raise ValueError('Geometry %s not supported' % self.geometry)

        # these are only per potential
        theta_pots1 = theta_interp1(self.r0_scale1 / (self.mesh['pots'] - self.q))
        theta_pots2 = theta_interp2(self.r0_scale2 / (pots2 - 1./self.q))

        Ts1, chis1, rhos1, Js1 = self.populate_mesh_component(theta_pots1,self.le1['Tc'],self.le1['rhoc'],self.polytropic_index1)
        Ts2, chis2, rhos2, Js2 = self.populate_mesh_component(theta_pots2,self.le2['Tc'],self.le2['rhoc'],self.polytropic_index2)

        # convert all structural quantities to rs-sized arrays (equal values per potential)

        rhos = np.zeros(arrlen)
        Ts = np.zeros(arrlen)
        chis = np.zeros(arrlen)
        Js = np.zeros(arrlen)


        # SOMETHING IS WRONG HERE!!!!!!!!!!!!!!
        for j, pot in enumerate(self.mesh['pots']):
            for arr in ['rhos','Ts','chis','Js']:
                print 'Populating potential {}, array {}'.format(pot,arr)
                rsarr = locals()['%s' % arr]
                potsarr1 = locals()['%s1' % arr]
                potsarr2 = locals()['%s2' % arr]

                pargs_pot = pargs[pots_rs[pargs]==pot]
                sargs_pot = sargs[pots_rs[sargs]==pot]
                # pargs_pot = np.argwhere((self.mesh['rs'][:,0] <= self.nekmin1) & (pots_rs==pot)).flatten()
                # sargs_pot = np.argwhere((self.mesh['rs'][:, 0] > self.nekmin1) & (pots_rs == pot)).flatten()

                rsarr[pargs_pot] = potsarr1[j]
                rsarr[sargs_pot] = potsarr2[j]

                # print potsarr1[j], potsarr2[j]
                # print rsarr[pargs_pot], rsarr[sargs_pot]

        self.mesh['rhos'] = rhos

        if self.geometry == 'spherical':
            np.save(self.directory + 'pots1', pots_rs[pargs].reshape(self.dims))
            np.save(self.directory + 'pots2', pots_rs[sargs].reshape(self.dims))
            np.save(self.directory + 'pots', pots_rs)
            np.save(self.directory + 'pargs', pargs)
            np.save(self.directory + 'sargs', sargs)
            np.save(self.directory + 'T1_0', Ts[pargs].reshape(self.dims))
            np.save(self.directory + 'T2_0', Ts[sargs].reshape(self.dims))
            np.save(self.directory + 'chi1_0', chis[pargs].reshape(self.dims))
            np.save(self.directory + 'chi2_0', chis[sargs].reshape(self.dims))
            np.save(self.directory + 'J1_0', Js[pargs].reshape(self.dims))
            np.save(self.directory + 'J2_0', Js[sargs].reshape(self.dims))

            for i in range(len(self.sc_params['thetas'])):
                np.save(self.directory + 'I1_0_' + str(int(i)), Js[pargs].reshape(self.dims))
                np.save(self.directory + 'I2_0_' + str(int(i)), Js[sargs].reshape(self.dims))

        elif self.geometry == 'cylindrical':
            np.save(self.directory + 'pots', pots_rs.reshape(self.dims))
            np.save(self.directory+'T_0', Ts.reshape(self.dims))
            np.save(self.directory+'chi_0', chis.reshape((self.dims)))
            np.save(self.directory+'J_0', Js.reshape(self.dims))

            for i in range(len(self.sc_params['thetas'])):
                np.save(self.directory + 'I_0_' + str(int(i)), Js.reshape(self.dims))


    def populate_mesh_component(self,theta_pots, Tc, rhoc, n):

        # populates the per-component mesh with a polytropic solution
        # Js, Ts and chis are the initial blackbody values which change with each iteration
        # rhos remain unchanged

        Ts = Tc * theta_pots
        rhos = rhoc * theta_pots ** n

        rhos_cgs = (rhos * const.M_sun / (const.R_sun) ** 3).value * 1e-3
        opacs_si = polytropes.opal_sun_points(Ts, rhos_cgs)  # this opac is in m^2/kg
        opacs = (opacs_si * const.M_sun / const.R_sun ** 2).value

        chis = opacs * rhos
        Js = (1. / np.pi) * (stefb / const.L_sun.value * const.R_sun.value ** 2) * Ts ** 4

        return Ts, chis, rhos, Js


    def finalize_mesh(self):

        if self.geometry == 'spherical':

            # In spherical geometry, parts of the mesh are "empty" near the neck
            # this part of the code computes the spherical angles at which each equipotential becomes "empty"

            dims = self.dims

            # compute theta and phi coordinates of neck per potential
            dimlen = self.dims[0] * self.dims[1] * self.dims[2]
            potlen = self.dims[1] * self.dims[2]
            breaks1 = np.zeros((self.dims[0], 3))
            breaks2 = np.zeros((self.dims[0], 3))

            for i in range(dims[0]):
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

        # compute the rescaling factor and pickle the mesh body
        # self.rescale_factor = np.average(self.rescale_factors())
        self.rescale_factor = np.average(self.rescale_factors())
        self.pickle(self.directory + 'body')


    def prep_interp_funcs(self, iter_n=1):

        # open the relevant files and prepare the interpolation functions to be used in sweep_mesh
        RGI = spint.RegularGridInterpolator

        # interpolate the Lane-Emden solution as well to compute points inside the star but outside of the mesh

        le1 = self.le1.copy()
        le2 = self.le2.copy()

        le1['theta_interp_func'] = interp1d(self.le1['theta_interp'][0], self.le1['theta_interp'][1],
                                            fill_value='extrapolate')
        le2['theta_interp_func'] = interp1d(self.le2['theta_interp'][0], self.le2['theta_interp'][1],
                                            fill_value='extrapolate')

        if self.geometry == 'spherical':

            chi1 = np.load(self.directory + 'chi1_' + str(iter_n - 1) + '.npy')
            chi2 = np.load(self.directory + 'chi2_' + str(iter_n - 1) + '.npy')
            J1 = np.load(self.directory + 'J1_' + str(iter_n - 1) + '.npy')
            J2 = np.load(self.directory + 'J2_' + str(iter_n - 1) + '.npy')

            I1_interp = []
            I2_interp = []

            chi1_interp = RGI(points=[self.mesh['pots'], self.mesh['thetas'], self.mesh['phis']], values=chi1)
            chi2_interp = RGI(points=[self.mesh['pots'], self.mesh['thetas'], self.mesh['phis']], values=chi2)
            J1_interp = RGI(points=[self.mesh['pots'], self.mesh['thetas'], self.mesh['phis']], values=J1)
            J2_interp = RGI(points=[self.mesh['pots'], self.mesh['thetas'], self.mesh['phis']], values=J2)

            for i in range(len(self.sc_params['thetas'])):
                I1 = np.load(self.directory + 'I1_' + str(iter_n - 1) + '_' + str(int(i)) + '.npy')
                I1_interp.append(RGI(points=[self.mesh['pots'], self.mesh['thetas'], self.mesh['phis']], values=I1))

                I2 = np.load(self.directory + 'I2_' + str(iter_n - 1) + '_' + str(int(i)) + '.npy')
                I2_interp.append(RGI(points=[self.mesh['pots'], self.mesh['thetas'], self.mesh['phis']], values=I2))

            return {'le1': le1, 'le2': le2, 'chi1': chi1_interp, 'chi2': chi2_interp, 'J1': J1_interp, 'J2': J2_interp,
                    'I1': I1_interp, 'I2': I2_interp}

        elif self.geometry == 'cylindrical':

            chi = np.load(self.directory + 'chi_' + str(iter_n - 1) + '.npy')
            J = np.load(self.directory + 'J_' + str(iter_n - 1) + '.npy')

            chi_interp = RGI(points=[self.mesh['pots'], self.mesh['xnorms'], self.mesh['thetas']], values=chi)
            J_interp = RGI(points=[self.mesh['pots'], self.mesh['xnorms'], self.mesh['thetas']], values=J)

            I_interp = []

            for i in range(len(self.sc_params['thetas'])):
                I = np.load(self.directory + 'I_' + str(iter_n - 1) + '_' + str(int(i)) + '.npy')
                I_interp.append(RGI(points=[self.mesh['pots'], self.mesh['xnorms'], self.mesh['thetas']], values=I))

            return {'le1': le1, 'le2': le2, 'chi': chi_interp, 'J':J_interp, 'I': I_interp}

        else:
            raise ValueError('Geometry %s not supported for RT' % self.geometry)


    def sweep_mesh(self,points,iter_n=1):

        interp_funcs = self.prep_interp_funcs(iter_n=iter_n)
        # setup the arrays that the computation will output
        I_new = np.zeros((len(points), self.nI))
        # thetas_new = np.zeros((len(points), self.nI))
        # phis_new = np.zeros((len(points), self.nI))
        taus_new = np.zeros((len(points), self.nI))

        for j, indx in enumerate(points):
            # print 'Entering rt computation'
            logging.info('Computing intensities for point %i of %i' % (j+1, len(points)))
            r = self.mesh['rs'][indx]
            if np.all(r==0.) or (r[0]==self.scale and r[1]==0. and r[2]==0.):
                pass
            else:
                if self.geometry == 'spherical':
                    I_new[j], taus_new[j] = rt_spherical.run(Mc=r, normal=self.mesh['normals'][indx],
                            sc_params=self.sc_params, scale = self.scale, q = self.q, interp_funcs = interp_funcs,
                            r0_scales = [self.r0_scale1, self.r0_scale2], nekmin = self.nekmin1,
                            angle_breaks = [self.breaks1, self.breaks2], pot_range = [self.pot,self.pot+self.pot_range])

                elif self.geometry == 'cylindrical':
                    I_new[j], taus_new[j] = rt_cylindrical.run(Mc=r, normal=self.mesh['normals'][indx],
                            sc_params=self.sc_params, scale=self.scale, q=self.q, interp_funcs=interp_funcs,
                            r0_scales=[self.r0_scale1,self.r0_scale2], nekmin=self.nekmin1,
                            pot_range=[self.pot,self.pot + self.pot_range])

                else:
                    raise ValueError('Geometry %s not supported for RT' % self.geometry)

        return I_new, taus_new


    def rescale_factors(self):

        points = random.sample(range(int(0.5*self.dims[0]*self.dims[1]*self.dims[2]), int(0.9*self.dims[0]*self.dims[1]*self.dims[2])), 5)

        rescale_factors = np.zeros(len(self.sc_params['thetas']))
        Is, taus = self.sweep_mesh(points, iter_n=1)

        for l in range(len(self.sc_params['thetas'])):
            Il = np.load(self.directory+'I_0_'+str(int(l))+'.npy').flatten()
            mask = (Is[:,l]!=0.0)
            rescale_factors[l] = np.average(Il[points][mask]/Is[:,l][mask])

        return rescale_factors

    def conserve_energy(self, I_new, points):

        J = np.zeros(len(points))
        F = np.zeros(len(points))
        T = np.zeros(len(points))
        chi = np.zeros(len(points))

        ws = self.sc_params['ws']

        for i, indx in enumerate(points):

            J[i] = np.sum(ws * I_new[i])

            # cond_out = self.sc_params['thetas'] <= np.pi/2
            # cond_in = self.sc_params['thetas'] > np.pi/2
            #
            # #I_em[i] = 0.5*np.pi*np.sum(ws[cond_out] * I_new[i][cond_out])
            # F[i] = 2 * np.pi *np.sum(ws[cond_out]*I_new[i][cond_out]*np.cos(self.sc_params['thetas'][cond_out])) - \
            #        2 * np.pi *np.sum(ws[cond_in]*I_new[i][cond_in]*np.cos(self.sc_params['thetas'][cond_in]))
            # # F[i] = np.sum(ws[cond_out] * I_new[i][cond_out]) - np.sum(ws[cond_in] * I_new[i][cond_in])

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

    def pickle(self, filename):
        f = file(filename, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    @staticmethod
    def unpickle(filename):
        with file(filename, 'rb') as f:
            return pickle.load(f)
