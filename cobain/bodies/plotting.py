import numpy as np
import os
import matplotlib.pyplot as plt
from cobain.structure import potentials
from matplotlib import rc
rc('text', usetex=True)

def compute_pots(body,comp='1'):

    len_body = body.dims[0] * body.dims[1] * body.dims[2]
    len_pot = body.dims[1] * body.dims[2]

    if comp == '1' or comp == '2':
        pots = np.zeros(len_body)

        for i in range(body.dims[0]):
            pots[i * len_pot:(i + 1) * len_pot] = body.mesh['pots'][i]

    elif comp == 'n':
        pots = np.zeros(len_body)

        for i in range(len(pots)):
            pots[i] = potentials.BinaryRoche(body.mesh['rs'][2 * len_body + i] / body.scale, body.q)

    else:
        pots = np.zeros(len_body)

        for i in range(body.dims[0]):
            pots[i * len_pot:(i + 1) * len_pot] = body.mesh['pots'][i]

    return pots


def compute_fill_area_diffs(pots,fs,i,skip=250):

    xpots = pots
    pots_plot = []
    pots_plot_err = []
    ymins = []
    ymaxs = []
    ydiff_mins = []
    ydiff_maxs = []

    for j, pot in enumerate(xpots):
        ys0 = np.log10(fs[0][pots == pot])
        ys = np.log10(fs[i][pots == pot])
        ys_prev = np.log10(fs[i - 1][pots == pot])

        if ys[~np.isinf(ys)].size > 0:
            pots_plot.append(pot)
            ymins.append(np.nanmin(ys))
            if j >= 15:
                ymaxs.append(np.nanmax(ys0))
            else:
                ymaxs.append(np.nanmax(ys))
        if ys_prev[~np.isinf(ys_prev)].size > 0:
            pots_plot_err.append(pot)
            dmin = np.nanmin(ys - ys_prev)
            dmax = np.nanmax(ys - ys_prev)

            if i != 0:
                if dmin < 0:
                    ydiff_mins.append(dmin)
                    if dmax < 0:
                        ydiff_maxs.append(0.)
                    else:
                        ydiff_maxs.append(dmax)
                else:
                    ydiff_mins.append(0.)
                    ydiff_maxs.append(dmax)
            else:
                ydiff_mins.append(0.)
                ydiff_maxs.append(0.)

    # print ymins, ymaxs
    pots_plot, ymins, ymaxs = np.array(pots_plot), np.array(ymins), np.array(ymaxs)
    pots_plot_err, ydiff_mins, ydiff_maxs = np.array(pots_plot_err), np.array(ydiff_mins), np.array(
        ydiff_maxs)

    return pots_plot, ymins, ymaxs, pots_plot_err, ydiff_mins, ydiff_maxs

def plot_function_pot(pots,fs,cond,xlabel,ylabel,filename,skip=250,savefig=False):
    
    plt.figure(figsize=(8, 6))
    ax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax_err = plt.subplot2grid((3, 1), (2, 0), rowspan=1)

    for i in range(len(fs)):
        pots_plot, ymins, ymaxs, pots_plot_err, ydiff_mins, ydiff_maxs = compute_fill_area_diffs(pots, fs, i, skip=250)

        ax.fill_between(pots_plot, ymins, ymaxs, alpha=0.5, linewidth=3, label=r'%s' % i)
        ax.plot(pots[cond][::skip], np.log10(fs[i][cond])[::skip], '.', markersize=3)

        ax_err.fill_between(pots_plot_err, ydiff_mins, ydiff_maxs, alpha=0.2,
                            label=r'iter$_{%s}$ - iter_${%s}$' % (i, i - 1))
        if i == 0:
            ax_err.plot(pots[cond][::skip], np.zeros(len(pots[cond][::skip])), '.',
                        label=r'iter$_{%s}$ - iter_${%s}$' % (i, i - 1), markersize=3)
        else:
            ax_err.plot(pots[cond][::skip], np.log10(fs[i][cond])[::skip] - np.log10(fs[i - 1][cond])[::skip], '.',
                        label=r'iter$_{%s}$ - iter_${%s}$' % (i, i - 1), markersize=3)

    ax_err.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax_err.set_ylabel(r'$i_n - i_{n-1}$')
    ax.legend(loc='lower right', title=r'Iter.')
    # ax_err.set_title(r'Log differences between consecutive iterations', y=0.95)
    # plt.tight_layout()
    if savefig:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_intensity_dir(body,simdir,iter_n,direction_no,pot_range,comp=1,skip=1,savefig=False):

    # body is the whole class (for ease of access to everything)
    # iter_n - number of iterations to plot

    pots = compute_pots(body, comp=comp)
    cond = (pots >= pot_range[0]) & (pots <= pot_range[1])

    Is = []

    if body.type == 'contact':
        Is.append(np.load(body.directory + 'I%s_%s_%s.npy' % (comp, 0, direction_no)).flatten())

        for i in range(1,iter_n+1):
            Is.append(np.load(simdir + 'I%s_%s_%s.npy' % (comp, i, direction_no)).flatten())
    elif body.type == 'diffrot':
        Is.append(np.load(body.directory + 'I_%s_%s.npy' % (0, direction_no)).flatten())

        for i in range(1, iter_n + 1):
            Is.append(np.load(simdir + 'I_%s_%s.npy' % (i, direction_no)).flatten())
    else:
        raise TypeError('Object type not recognized')

    xlabel = r'$\Omega$'
    ylabel = r'$I_{%s}$' % direction_no
    if not os.path.exists(simdir+'pics/'):
        os.makedirs(simdir+'pics/')
    filename = simdir+'pics/I%s_%s_%s.png' % (comp, iter_n, direction_no)

    plot_function_pot(pots, Is, cond, xlabel, ylabel, filename, skip=skip, savefig=savefig)


def plot_mean_intensity(body,simdir,iter_n,pot_range,comp=1,skip=1,savefig=False):

    pots = compute_pots(body, comp=comp)
    cond = (pots >= pot_range[0]) & (pots <= pot_range[1])
    xlabel = r'$\Omega$'
    if not os.path.exists(simdir+'pics/'):
        os.makedirs(simdir+'pics/')

    Js = []
    if body.type == 'contact':
        Js.append(np.load(body.directory + 'J%s_%s.npy' % (comp, 0)).flatten())
        for i in range(1,iter_n+1):
            Js.append(np.load(simdir + 'J%s_%s.npy' % (comp, i)).flatten())
    elif body.type == 'diffrot':
        Js.append(np.load(body.directory + 'J_%s.npy' % 0).flatten())
        for i in range(1,iter_n+1):
            Js.append(np.load(simdir + 'J_%s.npy' % i).flatten())
    else:
        raise TypeError('Object type not recognized')

    ylabel = r'$J$'
    filename = simdir+'pics/J%s_%s.png' % (comp, iter_n)
    plot_function_pot(pots[cond], Js, cond, xlabel, ylabel, filename, skip=skip, savefig=savefig)


def plot_temperature(body,simdir,iter_n,pot_range,comp=1,skip=1,savefig=False):

    pots = compute_pots(body, comp=comp)
    cond = (pots >= pot_range[0]) & (pots <= pot_range[1])
    xlabel = r'$\Omega$'
    if not os.path.exists(simdir + 'pics/'):
        os.makedirs(simdir + 'pics/')

    Ts = []
    if body.type == 'contact':
        Ts.append(np.load(body.directory + 'T%s_%s.npy' % (comp, 0)).flatten())
        for i in range(1, iter_n + 1):
            Ts.append(np.load(simdir + 'T%s_%s.npy' % (comp, i)).flatten())
    elif body.type == 'diffrot':
        Ts.append(np.load(body.directory + 'T_%s.npy' % 0).flatten())
        for i in range(1, iter_n + 1):
            Ts.append(np.load(simdir + 'T_%s.npy' % i).flatten())
    else:
        raise TypeError('Object type not recognized')

    ylabel = r'$T$'
    filename = simdir + 'pics/T%s_%s.png' % (comp, iter_n)
    plot_function_pot(pots, Ts, cond, xlabel, ylabel, filename, skip=skip, savefig=savefig)



