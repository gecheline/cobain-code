import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.special import legendre
import scipy.interpolate as spint
from cobain.structure import potentials
from mpl_toolkits.mplot3d import axes3d
import sys

compute = True

def Roche_cylindrical(r, x = 0., theta = 0., q = 1.):
    return 1./(x**2+r**2)**0.5 + q*(1./((x-1)**2+r**2)**0.5 - x) + 0.5*(1+q)*(x**2+(r*np.sin(theta))**2)

def radius_newton_tidal(r0, x, theta, pot, q):

    def Roche(r,x=x, theta=theta, q=q):
        return pot - Roche_cylindrical(r,x=x,theta=theta,q=q)

    try:
        return newton(Roche, r0, args=(x,theta,q), maxiter=100000, tol=1e-8)

    except:
        return np.nan

def radius_pot_contact_approx(pot,q,lam,nu):

    def P(order):
        return legendre(order)(lam)

    a0 = q*P(2)+0.5*(q+1.)*(1-nu**2)
    r0 = 1./(pot-q)

    func = 1. + r0**3 * a0 + r0**4 * q * P(3) + r0**5 * q * P(4) + r0**6 * (q * P(5) + 3.*a0**2) + r0**7 * (q * P(6) + 7. * q * a0 * P(3)) + \
           r0**8 * (q * P(7) + 8. * q * a0 * P(4) + 4 * q**2 * P(3)**2) + r0**9 * (q * P(8) + 9. * q * a0 * P(5) + 9. * q**2 * P(3) * P(4)) + \
           r0**10 * (q * P(9) + 10. * q * a0 * P(6) + 5. * q**2 * (P(4)**2 + 2. * P(3) * P(5)))

    return r0 * func

def find_r0(x,q,pot):

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
    # roots = np.roots((1,0,u,0,v,0,w,0,t))
    # reals_i = roots[np.isreal(roots)]
    # if reals_i.size != 0:
    #     reals = np.real(reals_i)
    #     # print reals
    #
    #     if np.all(reals >= 0.):
    #         # print 'all positive', reals
    #         return np.min(reals)
    #     elif np.all(reals < 0.):
    #         return np.nan
    #     else:
    #         reals_pos = reals[reals >= 0.]
    #         if reals_pos.size > 1:
    #             return np.max(reals_pos)
    #         else:
    #             return reals_pos[0]
    # else:
    #     return np.nan

def find_root(coeffs):
    roots = np.roots(coeffs)
    reals_i = roots[np.isreal(roots)]
    print reals_i
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

def find_back_front(pot, q):

    crit_pot = potentials.critical_pots(q)['pot_L1']

    den = (1. + q)
    A_b = (1. + 3*q)/den
    B_b = 2.*(q-pot)/den
    C_b = 2.*(q-pot+1.)/den
    D_b = 2./den
    A_f = -A_b
    B_f = B_b
    C_f = 2.*(q+pot+1.)/den
    D_f = -D_b

    if pot > crit_pot:
        # if star is detached, find both the back and front radius
        r_b = find_root((1,A_b,B_b,C_b,D_b))
        r_f = find_root((1,A_f,B_f,C_f,D_f))

    else:
        # if star is contact, find only the back radius
        r_b = find_root((1, A_b, B_b, C_b, D_b))
        r_f = np.nan

    return r_b, r_f


q = 1.0
# pot = 2.5
pots = np.linspace(3.76,3.74,20)

if compute == True:
    xs = np.linspace(0.,1.,100)
    # these are now normalized xs on range (-rback1, rback2) with a gap in between the components if detached

    # thetas = np.linspace(0.,np.pi,20)
    thetas = np.array([0.])
    rs = np.zeros((len(pots)*len(xs)*len(thetas),3))
    Is = np.zeros((len(pots),len(xs),len(thetas)))
    r0s = np.zeros(len(xs))
    rs_bf = np.zeros((len(pots),4))
    i = 0
    j = 0

    for m, pot in enumerate(pots):

        # for each potential, the four critical points need to be computed
        # rb1, rf1 = find_back_front(pot, q)
        # rb2, rf2 = find_back_front(pot/q+0.5*(q-1)/q, 1./q)
        rb1 = potentials.radius_newton_tidal([-0.5,0.,0.], pot, q, 10.)
        rf1 = potentials.radius_newton_tidal([0.5, 0., 0.], pot, q, 10.)
        rb2 = potentials.radius_newton_tidal([-0.5,0.,0.], pot/q+0.5*(q-1)/q, 1./q, 10.)
        rf2 = potentials.radius_newton_tidal([0.5, 0., 0.], pot/q+0.5*(q-1)/q, 1./q, 10.)

        rs_bf[m] = np.array([rb1,rf1,1.-rf2,1+rb2])

        d = rb1+rf1+rf2+rb2
        xd = (rb1+rf1)/d
        yd = (rf2+rb2)/d
        deltax = (1 - rf2) - rf1

        for n,x_norm in enumerate(xs):

            # EACH NORMALIZED X NEEDS TO BE CONVERTED
            if np.isnan(rf1):
                x = x_norm*d-rb1
            else:
                if x_norm <= xd:
                    x = x_norm*d - rb1
                else:
                    x = x_norm*d + deltax - rb1

            # find the polar radius for the given x
            r0 = find_r0(x,q,pot)

            if ~np.isnan(r0):
                r0s[j] = r0
                for k,theta in enumerate(thetas):

                    # find the true radius for the given theta
                    rho = radius_newton_tidal(r0,x,theta,pot,q)
                    print rho
                    if np.isnan(rho):
                        # pass
                        rs[i] = np.array([x, 0., 0.])
                    else:
                        rs[i] = np.array([x,rho*np.sin(theta),rho*np.cos(theta)])
                        Is[m,n,k] = pot**2
                    i+=1
            else:
                for theta in thetas:
                    rs[i] = np.array([x, 0., 0.])
                    i += 1
                # i += len(thetas)
        j+=1

    np.save('xs', xs)
    np.save('r0s', r0s)
    np.save('pots', pots)
    np.save('thetas', thetas)
    np.save('Is', Is)

    lenpot = len(xs) * len(thetas)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for j, pot in enumerate(pots):
        ax.scatter(rs[:, 0][j * lenpot:(j + 1) * lenpot], rs[:, 2][j * lenpot:(j + 1) * lenpot], s=1)

    # ax.set_ylim((-0.01,0.5))
    plt.show()

else:
    xs = np.load('xs.npy')
    r0s = np.load('r0s.npy')
    pots = np.load('pots.npy')
    thetas = np.load('thetas.npy')
    Is = np.load('Is.npy')

'''
RGI = spint.RegularGridInterpolator
I_interp = RGI(points=[pots, xs, thetas], values=Is)

Np = 1000000
xs = np.random.uniform(-1.,1.5,Np)
# ys = np.random.uniform(0.,0.45,Np)
# zs = np.random.uniform(0.,0.45,Np)

rs_int = np.random.uniform(0.,0.4,Np)
thetas_int = np.random.uniform(0.,np.pi,Np)
zs = rs_int/np.sqrt((np.tan(thetas_int))**2+1)
ys = np.tan(thetas_int)*zs
# rs_int = np.sqrt(ys**2+zs**2)
# thetas_int = np.arctan2(ys,zs)
pots_int = Roche_cylindrical(rs_int, xs, thetas_int, q)
Is_int = np.zeros(Np)
cond = (pots_int >= np.min(pots)) & (pots_int <= np.max(pots))
Is_int[cond] = I_interp((xs[cond], pots_int[cond], thetas_int[cond]))
Is_int[pots_int > np.max(pots)] = -200

plt.rc('text', usetex=True)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
p=ax.scatter(xs[cond], ys[cond], zs[cond], c=Is_int[cond]-pots_int[cond]**2, s=1)
fig.colorbar(p,label='$f_{interp} - f_{true}$')
plt.show()

# plt.plot(pots, pots**2, 'k-')
# plt.plot(pots_int,Is_int,'r.')
# plt.xlim((np.min(pots)-0.5, np.max(pots)+0.5))
# plt.show()

plt.scatter(rs_int[cond], pots_int[cond]**2, c='k', s=2)
plt.scatter(rs_int[cond], Is_int[cond], c=xs[cond], s=1, cmap='rainbow')
plt.colorbar(label=r'$x$ coordinate')
plt.xlabel(r'$r_{pole}$')
plt.ylabel(r'$f(\Omega)$')
plt.show()
'''