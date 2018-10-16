import numpy as np
import matplotlib.pyplot as plt
import cobain

mass1 = 1.0 # mass of the primary star
q = 1.0 # mass ratio of the system
FF = 0.9 # fillout factor
pot_range = 0.01 # range of potentials to build the mesh in
dims_cyl = [20,50,25] # dimensions of each component in the mesh [potentials, thetas, phis]
dims_sph = [20, 25, 25]
n1 = 3.0 # polytropic index of the primary
n2 = 3.0 # polytropic index of the secondary
lebedev_ndir = 5 # order of Lebedev quadratures

cb_sph = cobain.bodies.binary.Contact_Binary(mass1 = mass1, q = q, ff = FF, pot_range = pot_range,
                                             geometry='spherical', dims = dims_sph, n1 = n1, n2 = n2,
                                             lebedev_ndir = lebedev_ndir, dir = 'sph/')
print 'Created spherical contact'

cb_cyl = cobain.bodies.binary.Contact_Binary(mass1 = mass1, q = q, ff = FF, pot_range = pot_range,
                                             geometry='cylindrical', dims = dims_cyl, n1 = n1, n2 = n2,
                                             lebedev_ndir = lebedev_ndir, dir = 'cyl/')

print 'Created cylindrical contact'


cb_sph.build_mesh()
print 'Populated spherical'

cb_cyl.build_mesh()
print 'Populated cylindrical'

# cb_sph = cobain.bodies.Contact_Binary.unpickle('sph/body')
# cb_cyl = cobain.bodies.Contact_Binary.unpickle('cyl/body')
# args_to_sweep = np.random.uniform(0,25000,500).astype(int)
args_to_sweep = np.arange(0,25000,10).astype(int)
Is_sph, taus_sph = cb_sph.sweep_mesh(args_to_sweep)
Is_cyl, taus_cyl = cb_cyl.sweep_mesh(args_to_sweep)

pots_sph = cobain.bodies.plotting.compute_pots(cb_sph)
pots_sph = np.hstack((pots_sph,pots_sph))
pots_cyl = cobain.bodies.plotting.compute_pots(cb_cyl)

for i in range(len(cb_cyl.sc_params['thetas'])):
    plt.plot(pots_sph[args_to_sweep], np.log10(Is_sph[:,i]), 'b=ko', label='spherical')
    plt.plot(pots_cyl[args_to_sweep], np.log10(Is_cyl[:,i]), 'r.', label='cylindrical')
    plt.xlabel('pot')
    plt.ylabel('I (dir %s)' % i)
    plt.legend(loc='best')
    plt.savefig('test_cyl_sph/I_%s_whole.png' % i)
    plt.close()

np.save('test_cyl_sph/I_sph_whole', Is_sph)
np.save('test_cyl_sph/I_cyl_whole', Is_cyl)
np.save('test_cyl_sph/args_whole', args_to_sweep)

# plt.scatter(cb_sph.mesh['rs'][:,0], cb_sph.mesh['rs'][:,2], s=1, c='r')
# plt.scatter(cb_cyl.mesh['rs'][:,0], cb_cyl.mesh['rs'][:,2], s=1, c='b')
#
# T1_sph = np.load('sph/T1_0.npy')
# T_cyl = np.load('cyl/T_0.npy')
# pots_sph = np.load('sph/pots1.npy')
# pots_cyl = np.load('cyl/pots.npy')
#
# plt.plot(pots_sph.flatten(), T1_sph.flatten(), 'bx')
# plt.plot(pots_cyl.flatten(), T_cyl.flatten(), 'rx')
# plt.show()



# cb_cyl = cobain.bodies.binary_new.Contact_Binary.unpickle('cyl/body')
# points = range(0,1000,20)
# I_new, taus_new = cb_cyl.sweep_mesh(points,iter_n=1)
# np.save('I_new', I_new)
#
# pargs = np.argwhere(cb_sph.mesh['rs'][:,0] < cb_sph.nekmin1).flatten()
# sargs = np.argwhere(cb_sph.mesh['rs'][:,0] >= cb_sph.nekmin1).flatten()
#
# plt.scatter(cb_sph.mesh['rs'][:,0][pargs], cb_sph.mesh['rs'][:,2][pargs], color='b')
# plt.scatter(cb_sph.mesh['rs'][:,0][sargs], cb_sph.mesh['rs'][:,2][sargs], color='r')
# plt.show()





