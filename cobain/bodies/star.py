import numpy as np
import os
from scipy.interpolate import interp1d
import scipy.interpolate as spint
from cobain.structure import polytropes
from cobain.structure import potentials
import quadpy
from cobain.structure.constants import *
import astropy.constants as const
import pickle
import cobain.radiative_transfer as rt
import random
import logging
logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)


class DiffRot(object):

    def __init__(self, mass=1.0, radius=1.0, teff=5777.0, n=3.0, dims=[50,50,50], bs=[0.1,0.,0], pot_range=0.01,
                 lebedev_ndir=5,  dir=os.getcwd()):

        self.directory = dir

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        # setup star parameters
        self.mass = mass
        self.radius = radius
        self.teff = teff
        self.polytropic_index = n
        self.dims = dims
        self.type = 'diffrot'
        self.bs = bs

        # setup short characteristics parameters
        points = quadpy.sphere.Lebedev(lebedev_ndir).points
        weights = quadpy.sphere.Lebedev(lebedev_ndir).weights

        thetas = np.arccos(points[:, 2])
        phis = np.arctan2(points[:, 1], points[:, 0])

        self.ndir = lebedev_ndir
        self.nI = len(points)
        self.sc_params = {'thetas': thetas, 'phis': phis, 'ws': weights}

        # setup the scale for polytropes / sma for vertices
        # compute Lane-Emden structure
        le = polytropes.LE_diffrot(self.bs, self.radius, self.mass, self.teff)
        self.le = le
        # radius_s = polytropes.radius_pot(1. / le['r0_s'], self.bs, self.radius, 0.) #TECHNICALLY WHAT YOU'RE COMPUTING HERE IS THE POLAR RADIUS NOT UNDISTORTED
        self.radius = radius
        self.pot_r = 1. / le['r0_s']
        # find the potential where the optical depth equals 1 or 2/3 if grey
        self.pot = polytropes.find_opdepth_eq1_diffrot(le, self.teff)
        self.pot_range = [self.pot, self.pot+pot_range]

        # fix the scale so that the polar radius corresponding to the Roche potential
        # is the tau=2/3 radius
        self.scale = radius

        # compute the luminosity and flux of the star from area and teff
        area = self.area_volume(1. / self.pot, self.bs, self.radius)['area'] * self.scale ** 2
        self.luminosity = (area / (4 * np.pi)) * (teff / 5777) ** 4  # in solar luminosity
        self.flux = self.luminosity / area

        rs, normals, pots, thetas, phis, rhos = self.create_mesh()
        # rs, normals, pots, thetas, phis = self.create_mesh_old()

        self.mesh = {}
        self.mesh['rs'] = rs
        self.mesh['normals'] = normals
        self.mesh['pots'] = pots
        self.mesh['thetas'] = thetas
        self.mesh['phis'] = phis
        self.mesh['rhos'] = rhos

        self.rescale_factor = np.average(self.rescale_factors())
        self.pickle(self.directory+'body')

    def area_volume(self, r0, bs, R):
        V = 4. * np.pi / 3. * (R) ** 3 * r0 ** 3 * (
            1 + bs[0] * r0 ** 3 + 0.4 * bs[1] * r0 ** 5 + 1.6 * bs[0] ** 2 * r0 ** 6 + 8. / 35. * bs[
                2] * r0 ** 7 + 12. / 7. * bs[0] * bs[1] * r0 ** 8 + (
                128. / 105. * bs[0] * bs[2] + 16. / 35. * bs[1] ** 2) * r0 ** 10 + 208. / 35. * bs[0] ** 2 * bs[
                1] * r0 ** 11 + 64. / 99. * bs[1] * bs[2] * r0 ** 12)

        S = 4. * np.pi * (R) ** 2 * r0 ** 2 * (
            1 + 2. / 3. * bs[0] * r0 ** 3 + 4. / 15. * bs[1] * r0 ** 5 + 14. / 15. * bs[0] ** 2 * r0 ** 6 + 16. / 105. *
            bs[
                2] * r0 ** 7 + 36. / 35. * bs[0] * bs[1] * r0 ** 8 + (
                704. / 945. * bs[0] * bs[2] + 88. / 315. * bs[1] ** 2) * r0 ** 10 + 1024. / 315. * bs[0] ** 2 * bs[
                1] * r0 ** 11 + 832. / 2079. * bs[1] * bs[2] * r0 ** 12)

        return {'volume': V, 'area': S}

    def create_mesh(self):

        theta_interp = interp1d(self.le['theta_interp'][0],self.le['theta_interp'][1], fill_value='extrapolate')

        pots = np.linspace(self.pot_range[0], self.pot_range[1], self.dims[0])
        thetas = np.linspace(0.,np.pi/2,self.dims[1])
        phis = np.linspace(0.,np.pi,self.dims[2])

        rs = np.zeros((len(pots)*len(thetas)*len(phis),3))
        rhos = np.zeros(len(pots) * len(thetas) * len(phis))
        normals = np.zeros((len(pots) * len(thetas) * len(phis), 3))
        T = np.zeros((len(pots),len(thetas),len(phis)))
        chi = np.zeros((len(pots), len(thetas), len(phis)))
        S = np.zeros((len(pots), len(thetas), len(phis)))
        I = np.zeros((len(pots),len(thetas),len(phis)))

        n = 0
        for i,pot in enumerate(pots):
            logging.info('Building equipotential surface at pot=%s' % pot)
            for j,theta in enumerate(thetas):
                for k,phi in enumerate(phis):

                    r = potentials.radius_pot_diffrot(pot,self.bs,self.radius,theta)
                    r_cs = np.array([r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(theta)])

                    # compute the normal to the surface

                    nx = potentials.dDiffRotRochedx(r_cs/self.scale, self.bs)
                    ny = potentials.dDiffRotRochedy(r_cs/self.scale, self.bs)
                    nz = potentials.dDiffRotRochedz(r_cs/self.scale, self.bs)
                    nn = np.sqrt(nx * nx + ny * ny + nz * nz)

                    rs[n] = r_cs
                    normals[n] = np.array([nx/nn,ny/nn,nz/nn])

                    theta_pot = theta_interp(1./pot)
                    T[i,j,k], chi[i,j,k], rhos[n], S[i,j,k], I[i,j,k] = self.structure_per_point(pot,theta_pot)

                    n += 1

        # T_chi_S_I = {}
        # T_chi_S_I['T'] = T
        # T_chi_S_I['chi'] = chi
        # T_chi_S_I['S'] = S
        # T_chi_S_I['I'] = I

        # np.save(self.directory+'T_chi_S_I_0', T_chi_S_I)
        np.save(self.directory+'T_0', T)
        np.save(self.directory+'chi_0', chi)
        np.save(self.directory+'J_0', S)
        for l in range(len(self.sc_params['thetas'])):
            np.save(self.directory+'I_0_'+str(int(l)), I)
        return rs, normals, pots, thetas, phis, rhos


    def structure_per_point(self,pot,theta_pot):
        T = self.le['Tc'] * theta_pot
        rho = self.le['rhoc'] * theta_pot ** self.polytropic_index

        rho_cgs = (rho * const.M_sun / (const.R_sun) ** 3).value * 1e-3
        opac_si = polytropes.opal_sun(T, rho_cgs)[0]  # this opac is in m^2/kg
        opac = (opac_si * const.M_sun / const.R_sun ** 2).value

        chi = opac * rho

        J = (1./np.pi) * (stefb / const.L_sun.value * const.R_sun.value ** 2) * T ** 4
        I = J

        return T,chi,rho,J,I

    def sweep_mesh(self,points,iter_n=1):

        RGI = spint.RegularGridInterpolator

        chi = np.load(self.directory + 'chi_'+str(iter_n-1)+'.npy')
        J = np.load(self.directory + 'J_'+str(iter_n-1)+'.npy')

        chi_interp = RGI(points=[self.mesh['pots'], self.mesh['thetas'], self.mesh['phis']], values=chi)
        J_interp = RGI(points=[self.mesh['pots'], self.mesh['thetas'], self.mesh['phis']], values=J)

        I_interp = []

        for i in range(len(self.sc_params['thetas'])):
            I = np.load(self.directory + 'I_' + str(iter_n - 1) + '_' + str(int(i)) + '.npy')
            I_interp.append(RGI(points=[self.mesh['pots'], self.mesh['thetas'], self.mesh['phis']], values=I))


        le = self.le.copy()
        le['theta_interp_func'] = spint.interp1d(self.le['theta_interp'][0],self.le['theta_interp'][1], fill_value='extrapolate')

        # setup the arrays that the computation will output
        I_new = np.zeros((len(points), self.nI))
        taus_new = np.zeros((len(points), self.nI))
        thetas_new = np.zeros((len(points), self.nI))
        phis_new = np.zeros((len(points), self.nI))
        nsteps_new = np.zeros((len(points), self.nI))

        # rt_contact.rt_spline will also need to take the le params
        for j, indx in enumerate(points):
            # print 'Entering rt computation'
            logging.info('Computing intensities for point %i of %i' % (j + 1, len(points)))
            potc = potentials.compute_diffrot_potential(self.mesh['rs'][indx], self.pot_range[0],
                                                        self.bs, self.scale)
            # potc = self.mesh['pots'][indx]
            I_new[j], thetas_new[j], phis_new[j] = rt.star_diffrot.run(self.mesh['rs'][indx], self.mesh['normals'][indx],
                                                              potc, self.sc_params['thetas'],
                                                              self.sc_params['phis'], self.scale, self.bs,
                                                              self.pot_range, le, chi_interp, J_interp, I_interp)

        return I_new, thetas_new, phis_new

    def conserve_energy(self, I_new, points):

        J = np.zeros(len(points))
        I_em = np.zeros(len(points))
        F = np.zeros(len(points))
        T = np.zeros(len(points))
        chi = np.zeros(len(points))
        # rhos = np.load(self.directory + 'rho.npy')
        rhos = self.mesh['rhos']
        ws = self.sc_params['ws']

        for i, indx in enumerate(points):

            J[i] = np.sum(ws * I_new[i])

            cond_out = self.sc_params['thetas'] <= np.pi/2
            cond_in = self.sc_params['thetas'] >= np.pi/2

            #I_em[i] = 0.5*np.pi*np.sum(ws[cond_out] * I_new[i][cond_out])
            F[i] = 4*np.pi*np.sum(ws[cond_out]*I_new[i][cond_out]*np.cos(self.sc_params['thetas'][cond_out])) - \
                   4*np.pi*np.sum(ws[cond_in]*I_new[i][cond_in]*np.cos(self.sc_params['thetas'][cond_in]))

            rho = rhos[indx]

            T[i] = self.teff*(F[i]/self.flux)**0.25

            rho_cgs = (rho * const.M_sun / (const.R_sun) ** 3).value * 1e-3
            opac_si = polytropes.opal_sun(T[i], rho_cgs)[0]  # this opac is in m^2/kg
            opac_new = (opac_si * const.M_sun / const.R_sun ** 2).value
            chi[i] = opac_new*rho

        return J, T, chi

    def conserve_luminosity(self,F):

        dtheta = (np.pi/2)/self.dims[1]
        dphi = np.pi/self.dims[2]

        scales = np.zeros(self.dims[0])
        Ss_i = np.zeros((self.dims[0],self.dims[1]*self.dims[2]))
        Fs_i = np.zeros((self.dims[0],self.dims[1]*self.dims[2]))
        Ls = np.zeros(self.dims[0])

        for i in range(self.dims[0]):
            rs = self.mesh['rs'][i*self.dims[1]*self.dims[2]:(i+1)*self.dims[1]*self.dims[2]]
            normals = self.mesh['normals'][i*self.dims[1]*self.dims[2]:(i+1)*self.dims[1]*self.dims[2]]
            pots = np.ones(len(rs))*self.mesh['pots'][i]
            thetas = np.zeros(len(rs))
            phis = np.zeros(len(rs))
            for j in range(self.dims[1]):
                thetas[j*self.dims[2]:(j+1)*self.dims[2]] = self.mesh['thetas'][j]
                for k in range(self.dims[2]):
                    # print 'i,j,k:', i,j,k
                    # print 'phis n: ', k+j*self.dims[2]
                    phis[k+j*self.dims[2]] = self.mesh['phis'][k]

            rabs = np.sum(rs**2,axis=1)**0.5
            cos_gammas = np.zeros(len(rs))

            for l,r in enumerate(rs):
                cos_gammas[l] = np.dot(r/rabs[l],normals[l])

            dSs = self.scale**2 * dtheta * dphi * rabs**2 * np.sin(thetas) / cos_gammas
            Ss_i[i] = dSs
            Fs = F[i*self.dims[1]*self.dims[2]:(i+1)*self.dims[1]*self.dims[2]]
            Fs_i[i] = Fs
            Ls[i] = 4*np.sum(dSs*Fs)
            scales[i] = self.luminosity/(np.sum(4*dSs*Fs))

        return Ss_i,Fs_i,Ls, scales

    def rescale_factors(self):

        points = random.sample(range(int(0.7*self.dims[0]*self.dims[1]*self.dims[2]), int(0.8*self.dims[0]*self.dims[1]*self.dims[2])), 5)

        rescale_factors = np.zeros(len(self.sc_params['thetas']))
        Is, thetas, phis = self.sweep_mesh(points, iter_n=1)

        for l in range(len(self.sc_params['thetas'])):
            Il = np.load(self.directory+'I_0_'+str(int(l))+'.npy').flatten()
            rescale_factors[l] = np.average(Il[points]/Is[:,l])

        return rescale_factors

    def pickle(self, filename):
        f = file(filename, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    @staticmethod
    def unpickle(filename):
        with file(filename, 'rb') as f:
            return pickle.load(f)
