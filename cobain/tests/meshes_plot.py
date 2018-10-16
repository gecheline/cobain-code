import numpy as np
import cobain.meshes.spherical
import cobain.meshes.cylindrical
import matplotlib.pyplot as plt
from cobain.structure import potentials
from matplotlib import rc
rc('text', usetex=True)


q = 1.
pot = 3.7
pot_range = 0.001
npot = 1

pot2 = pot / q + 0.5 * (q - 1) / q

rpole1 = potentials.radius_pot_contact_approx(pot, q, 0., 1.)
rpole2 = potentials.radius_pot_contact_approx(pot2, 1./q, 0., 1.)

xz, z = potentials.nekmin(pot, q, 0.5, 0.05)
nekmin1 = xz
nekmin2 = 1. - nekmin1


mesh_cyl = cobain.meshes.cylindrical.build_mesh(pot_s=pot, pot_range=pot_range, npot=npot, ntheta=15, nx=50, q=q)
mesh_sph = cobain.meshes.spherical.build_mesh(pot_s=pot, pot_range=pot_range, npot=npot, ntheta=15, nphi=15,
                                              q=q, type='contact', rpole1=rpole1, rpole2=rpole2, nekmin1=nekmin1, nekmin2=nekmin2)

rs_sph = mesh_sph['rs']
rs_cyl = mesh_cyl['rs']
ns_sph = mesh_sph['normals']
ns_cyl = mesh_cyl['normals']
np.savetxt('rs_sph_samesize.csv', rs_sph, delimiter=',')
np.savetxt('rs_cyl_samesize.csv', rs_cyl, delimiter=',')
# rs_sph = np.loadtxt('rs_sph_samesize.csv', delimiter=',')
# rs_cyl = np.loadtxt('rs_cyl_samesize.csv', delimiter=',')

plt.figure(figsize=(6,4))
plt.scatter(rs_cyl[:,0], rs_cyl[:,2], s=10, marker='x', linewidth=1, c='r', label='cylindrical')
plt.scatter(rs_sph[:,0], rs_sph[:,2], s=10, marker='.', c='k', label='spherical')
plt.quiver(rs_cyl[:,0], rs_cyl[:,2], ns_cyl[:,0],ns_cyl[:,2],color='r',pivot='tail')
plt.quiver(rs_sph[:,0], rs_sph[:,2], ns_sph[:,0],ns_sph[:,2],color='k',pivot='tail')
plt.xlabel('$x$')
plt.ylabel('$z$')
plt.legend()
plt.tight_layout()
plt.savefig('cyl_sph_normals.pdf')
plt.show()