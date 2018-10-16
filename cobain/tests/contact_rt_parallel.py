import numpy as np
import cobain
import sys
import os
from mpi4py import MPI

start = int(sys.argv[1])
end = int(sys.argv[2])
#iter_n = int(sys.argv[3])
iter_n = 1

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
nproc = size

WORKTAG = 0
DIETAG = 1

dirc = 'cyl_m1.0_q1.0_ff0.45/'
dirc_nodes = 'cyl_m1.0_q1.0_ff0.45_nodes_iter'+str(iter_n)+'/'

if rank == 0:
    if not os.path.exists(dirc_nodes):
        os.makedirs(dirc_nodes)

    st = cobain.bodies.binary.Contact_Binary.unpickle(dirc+'body')
    dimbreak = int(st.dims[0]*st.dims[1]*st.dims[2])
    points_0 = np.arange(0,len(st.mesh['rs']),1).astype(int)[start:end]

    print 'points len: ', len(points_0)
    mod = len(points_0) / (nproc)
    if mod == 0:
        points_sep = np.split(points_0, nproc)
        for i in range(1, size):
            comm.send([st, points_sep[i]], dest=i, tag=WORKTAG)
        points = points_sep[0]
    else:
        whole_split = len(points_0) / nproc
        points_sep = np.split(points_0[:whole_split * nproc], nproc)
        points = np.hstack((points_sep[-1], points_0[whole_split * nproc:])).flatten()

        for i in range(1, size):
            comm.send([st, points_sep[i - 1]], dest=i, tag=WORKTAG)

    print 'rank 0 sent everything to child nodes'

    print 'Sweeping...'
    I_new, taus_new = st.sweep_mesh(points,iter_n=iter_n)

    print 'Saving intensity arrays...'
    for filee, arr in zip(['points_', 'I_', 'taus_'],
                          [points, I_new,  taus_new]):
        np.save(dirc_nodes + filee + str(rank) + '_' + str(start), arr)

    print 'Intensity computation complete, computing J and F'
    J, T, chi = st.conserve_energy(I_new, points)

    print 'Saving J and F arrays...'
    for filee, arr in zip(['J_', 'T_', 'chi_'],
                          [J, T, chi]):
        np.save(dirc_nodes + filee + str(rank) + '_' + str(start), arr)

    print 'Saved arrays.'

else:
    print 'rank ', rank, 'received points from parent'
    status = MPI.Status()
    data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
    st = data[0]
    points = data[1]
    print '# of points to sweep:', len(points)

    print 'Sweeping...'
    I_new, taus_new = st.sweep_mesh(points,iter_n=iter_n)

    print 'Saving intensity arrays...'
    for filee, arr in zip(['points_', 'I_', 'taus_'],
                          [points, I_new, taus_new]):
        np.save(dirc_nodes + filee + str(rank) + '_' + str(start), arr)

    print 'Intensity computation complete, computing J and F'
    J, T, chi = st.conserve_energy(I_new, points)

    print 'Saving J and F arrays...'
    for filee, arr in zip(['J_', 'T_', 'chi_'],
                          [J, T, chi]):
        np.save(dirc_nodes + filee + str(rank) + '_' + str(start), arr)

    print 'Saved arrays.'
