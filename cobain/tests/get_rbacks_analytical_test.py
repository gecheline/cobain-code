import numpy as np
from cobain.structure import potentials
import scipy.interpolate as spint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def return_coeffs(q,pot):
    a = 1. + q
    b = 1. + 3 * q
    c = 2 * (q - pot)
    d = 2 * (q - pot + 1.)
    e = 2.

qs = np.linspace(0.01,1.,1000)
ffs = np.linspace(-0.01,1.0,1000)
roots_n_prim = np.zeros((len(qs)+1,len(ffs)+1))
roots_n_prim[0][1:] = qs
roots_n_prim[:,0][1:] = ffs
roots_n_sec = np.zeros((len(qs)+1,len(ffs)+1))
roots_n_sec[0][1:] = qs
roots_n_sec[:,0][1:] = ffs

def rback(q,pot):
    a = 1. + q
    b = 1. + 3 * q
    c = 2 * (q - pot)
    d = 2 * (q - pot + 1.)
    e = 2.

    coeffs = (a, b, c, d, e)
    roots = np.roots(coeffs)
    return roots[-1]

for i,q in enumerate(qs):
    for j,ff in enumerate(ffs):
        print 'i %s j %s' % (i,j)
        crit_pots = potentials.critical_pots(q)
        L1, L2 = crit_pots['pot_L1'], crit_pots['pot_L2']
        pot =  L1 + ff * (L2 - L1)
        pot2 = pot / q + 0.5 * (q - 1) / q

        # do primary
        roots_n_prim[i+1,j+1] = rback(q,pot)
        # do secondary
        roots_n_sec[i + 1, j + 1] = rback(1./q, pot2)

        # r0 = potentials.radius_pot_contact_approx(pot, q, -1., 0.)
        # r_ns.append(potentials.radius_newton_spherical(r0, -1., 0., 0., pot, q, 1.))
        #
        # roots_n.append(roots[-1])
        #
        # print 'r from roots: %s; r from newton: %s' % (roots[-1],r_ns[-1])


np.savetxt('rbacks_forinterp_negff_prim.csv', roots_n_prim, delimiter=',')
np.save('rbacks_forinterp_negff_prim.npy', roots_n_prim)
np.savetxt('rbacks_forinterp_negff_sec.csv', roots_n_sec, delimiter=',')
np.save('rbacks_forinterp_negff_sec.npy', roots_n_sec)
'''
rbacks = np.load('rbacks_forinterp.npy')
qs = rbacks[0].copy()[1:]
ffs = rbacks[:,0].copy()[1:]
rs_rowr = np.delete(rbacks, [0], axis=0)
rs = np.delete(rs_rowr, [0], axis=1)

xx, yy = np.meshgrid(qs,ffs)
RGI = spint.RegularGridInterpolator
f = RGI(points=[qs, ffs], values=rs)
qs_int = np.random.uniform(0.01,1.,1000)
ffs_int = np.random.uniform(0.001,0.999,1000)

rs_int = f((qs_int, ffs_int))

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(xx.T, yy.T, rs, cmap=cm.coolwarm,alpha=0.5)
ax.scatter(qs_int, ffs_int, rs_int, s = 1, c = rs_int, cmap = cm.coolwarm)
ax.set_xlabel('q')
ax.set_ylabel('ff')
ax.set_zlabel('req')
plt.show()
'''