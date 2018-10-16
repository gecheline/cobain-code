import numpy as np
from scipy.optimize import newton
from scipy.special import legendre

def critical_pots(q, sma=1., d=1., F=1.):
    # L1
    dxL = 1.0
    L1 = 1e-3
    while abs(dxL) > 1e-6:
        dxL = - dBinaryRochedx([L1, 0.0, 0.0], 1., q, 1.) / d2BinaryRochedx2([L1, 0.0, 0.0], 1.,
                                                                                                   q, 1.)
        L1 = L1 + dxL
    Phi_L1 = BinaryRoche(np.array([L1, 0.0, 0.0]), q, 1., 1.)

    # L2
    if (q > 1.0):
        q2 = 1.0 / q
    else:
        q2 = q

    dxL = 1.1e-6

    D = d * sma
    factor = (q2 / 3 / (q2 + 1)) ** 1. / 3.
    xL = 1 + factor + 1. / 3. * factor ** 2 + 1. / 9. * factor ** 3
    while (abs(dxL) > 1e-6):
        xL = xL + dxL
        Force = F * F * (q2 + 1) * xL - 1.0 / xL / xL - q2 * (xL - D) / abs((D - xL) ** 3) - q2 / D / D
        dxLdF = 1.0 / (F * F * (q2 + 1) + 2.0 / xL / xL / xL + 2 * q2 / abs((D - xL) ** 3))
        dxL = -Force * dxLdF

    if (q > 1.0):
        xL = D - xL
    xL2 = xL
    Phi_L2 = 1.0 / abs(xL2) + q * (1.0 / abs(xL2 - 1) - xL2) + 1. / 2. * (q + 1) * xL2 * xL2

    return {'xL1': L1, 'xL2': xL2, 'pot_L1': Phi_L1, 'pot_L2': Phi_L2}

def BinaryRoche_spherical(r, q, lam, nu):
    return 1. / r + q * ((1. - 2 * lam * r + r ** 2) ** (-0.5) - lam * r) + 0.5 * (q + 1.) * r ** 2 * (1 - nu ** 2)

def BinaryRoche_cylindrical(r,q,x,theta):
    return 1. / np.sqrt(x ** 2 + r ** 2) + q * (1. / np.sqrt((x - 1) ** 2 + r ** 2) - x) + 0.5 * (q + 1.) * (x ** 2 + r ** 2 * (np.sin(theta))**2)

def BinaryRoche(r,q,D=1.,F=1.):
    return 1.0 / np.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) + q * (
    1.0 / np.sqrt((r[0] - D) * (r[0] - D) + r[1] * r[1] + r[2] * r[2]) - r[0] / D / D) + 0.5 * F * F * (1 + q) * (
    r[0] * r[0] + r[1] * r[1])

def BinaryRoche_points(r,q,D=1.,F=1.):
    return 1.0 / np.sqrt(r[:,0] * r[:,0] + r[:,1] * r[:,1] + r[:,2] * r[:,2]) + q * (
    1.0 / np.sqrt((r[:,0] - D) * (r[:,0] - D) + r[:,1] * r[:,1] + r[:,2] * r[:,2]) - r[:,0] / D / D) + 0.5 * F * F * (1 + q) * (
    r[:,0] * r[:,0] + r[:,1] * r[:,1])

def radius_newton_tidal(rc, pot, q, xmin):
    r_start = np.sqrt(rc[0] ** 2 + rc[1] ** 2 + rc[2] ** 2)
    lam = rc[0] / r_start
    mu = rc[1] / r_start
    nu = rc[2] / r_start

    def Roche(r):
        return pot - BinaryRoche(r*np.array([lam,mu,nu]), q)

    try:
        r = newton(Roche, r_start, maxiter=100000, tol=1e-8)
        vc = r * np.array([lam, mu, nu])

        if ((vc[0] < 0. and vc[0] > -1. and vc[1] >= 0. and vc[2] >= 0.) or (abs(vc[0]) <= xmin) and (vc[1] >= 0.) and (vc[2] >= 0.)):
            # do this for separate components cause they may have different diverging strips
            return r
        else:
            return np.nan
    except:
        return np.nan

def radius_newton_spherical(r0, lam, mu, nu, pot, q, xmin):
    r_start = r0

    def Roche(r):
        return pot - BinaryRoche(r*np.array([lam,mu,nu]), q)

    try:
        r = newton(Roche, r_start, maxiter=100000, tol=1e-8)
        vc = r * np.array([lam, mu, nu])

        if ((vc[0] < 0. and vc[0] > -1. and vc[1] >= 0. and vc[2] >= 0.) or (abs(vc[0]) <= 1.2*xmin) and (vc[1] >= 0.) and (vc[2] >= 0.)):
            # do this for separate components cause they may have different diverging strips
            return r
        else:
            return np.nan
    except:
        return np.nan

def radius_newton_cylindrical(r0, x, theta, pot, q):

    def Roche(r,x=x, theta=theta, q=q):
        return pot - BinaryRoche_cylindrical(r,x=x,theta=theta,q=q)

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


def dBinaryRochedx(r, D, q, F):
    with np.errstate(divide='ignore', invalid='ignore'):
        return -r[0] * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) ** -1.5 - q * (r[0] - D) * ((r[0] - D) * (r[0] - D) + r[1] * r[1] + r[2] * r[2]) ** -1.5 - q / D / D + F * F * (1 + q) * r[0]


def d2BinaryRochedx2(r, D, q, F):
    with np.errstate(divide='ignore', invalid='ignore'):
        return (2 * r[0] * r[0] - r[1] * r[1] - r[2] * r[2]) / (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) ** 2.5 + q * (2 * (r[0] - D) * (r[0] - D) - r[1] * r[1] - r[2] * r[2]) / ((r[0] - D) * (r[0] - D) + r[1] * r[1] + r[2] * r[2]) ** 2.5 + F * F * (1 + q)


def dBinaryRochedy(r, D, q, F):
    with np.errstate(divide='ignore', invalid='ignore'):
        return -r[1] * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) ** -1.5 - q * r[1] * ((r[0] - D) * (r[0] - D) + r[1] * r[1] + r[2] * r[2]) ** -1.5 + F * F * (1 + q) * r[1]


def dBinaryRochedz(r, D, q, F):
    with np.errstate(divide='ignore', invalid='ignore'):
        return -r[2] * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) ** -1.5 - q * r[2] * ((r[0] - D) * (r[0] - D) + r[1] * r[1] + r[2] * r[2]) ** -1.5


def dBinaryRochedr(r, q, D, F):
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = (r * r).sum()
        r1 = np.sqrt(r2)

        return -1. / r2 - q * (r1 - r[0] / r1 * D) / ((r[0] - D) * (r[0] - D) + r[1] * r[1] + r[2] * r[2]) ** 1.5 - q * r[0] / r1 / D / D + F * F * (1 + q) * (1 - r[2] * r[2] / r2) * r1


def project_onto_potential(r, D, q, F):
    """
    TODO: add documentation
    """
    n_iter = 0

    rmag, rmag0 = np.sqrt((r * r).sum()), 0
    lam, nu = r[0] / rmag, r[2] / rmag
    dc = np.array(
        (lam, np.sqrt(1 - lam * lam - nu * nu), nu))  # direction cosines -- must not change during reprojection

    while np.abs(rmag - rmag0) > 1e-6 and n_iter < 100:
        rmag0 = rmag
        rmag = rmag0 - BinaryRoche(rmag0 * dc, q, D, F) / dBinaryRochedr(rmag0 * dc, q, D, F)
        n_iter += 1
    if n_iter == 100:
        print 'projection did not converge'

    return rmag


def compute_equivalent_radius_tidal(pot, q):
    # dimensionless

    n = 0.5 * (q + 1.)

    r0 = 1. / (pot - q)
    return r0 * (1. + 2. * n / 3 * r0 ** 3 + (
    4. / 5. * q ** 2 + 8. / 15 * n * q + 76. / 45 * n ** 2) * r0 ** 6 + 5. / 7 * q ** 2 * r0 ** 8 + 2. / 3 * q ** 2 * r0 ** 10)


def compute_diffrot_potential(vertex, pot_r, bs, scale):
    rabs = np.sqrt(vertex[0] ** 2 + vertex[1] ** 2 + vertex[2] ** 2)
    theta = np.arccos(vertex[2] / rabs)
    r = radius_pot_diffrot(pot_r, bs, scale, theta)

    if rabs <= r+1e-12:
        return DiffRotRoche(vertex / scale, bs)
    else:
        return 0.


def DiffRotRoche(r, bs):
    if r.shape == (3,):
        return (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** (-0.5) + 0.5 * (r[0] ** 2 + r[1] ** 2) * (
            bs[0] + 0.5 * bs[1] * (r[0] ** 2 + r[1] ** 2) + 1. / 3. * bs[2] * (r[0] ** 2 + r[1] ** 2) ** 2)
    else:
        return (r[:,0] ** 2 + r[:,1] ** 2 + r[:,2] ** 2) ** (-0.5) + 0.5 * (r[:,0] ** 2 + r[:,1] ** 2) * (
            bs[0] + 0.5 * bs[1] * (r[:,0] ** 2 + r[:,1] ** 2) + 1. / 3. * bs[2] * (r[:,0] ** 2 + r[:,1] ** 2) ** 2)


def dDiffRotRochedx(r, bs):
    return r[0] / (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** 0.5 + r[0] * (
        bs[0] + 0.5 * bs[1] * (r[0] ** 2 + r[1] ** 2) + 1. / 3. * bs[2] * (r[0] ** 2 + r[1] ** 2) ** 2) + 0.5 * r[
        0] * (r[0] ** 2 +r[1] ** 2) * (bs[1] + 4. / 3. * bs[2] * (r[0] ** 2 + r[1] ** 2))


def dDiffRotRochedy(r, bs):
    return r[1] / (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** 0.5 + r[1] * (
        bs[0] + 0.5 * bs[1] * (r[0] ** 2 + r[1] ** 2) + 1. / 3. * bs[2] * (r[0] ** 2 + r[1] ** 2) ** 2) + 0.5 * r[
        1] * (r[0] ** 2 + r[1] ** 2) * ( bs[1] + 4. / 3. * bs[2] * (r[0] ** 2 + r[1] ** 2))


def dDiffRotRochedz(r, bs):
    return r[2] / (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** 0.5


def radius_pot_diffrot(pot, bs, theta):
    r0 = 1. / pot
    x = 1. - (np.cos(theta)) ** 2

    r_start = r0*(1. + (bs[0]*r0**3*x)/2. + (bs[1]*r0**5*x**2)/4. + (3*bs[0]**2*r0**6*x**2)/4. +
    (bs[2]*r0**7*x**3)/6. + bs[0]*bs[1]*r0**8*x**3 + (3*bs[0]**3*r0**9*x**3)/2. +
    (5*(3*bs[1]**2 + 8*bs[0]*bs[2])*r0**10*x**4)/48. + (55*bs[0]**2*bs[1]*r0**11*x**4)/16. +
    (13*bs[0]*(3*bs[1]**2 + 4*bs[0]*bs[2])*r0**13*x**5)/16. +
    (35*bs[0]**2*(9*bs[1]**2 + 8*bs[0]*bs[2])*r0**16*x**6)/24. +
    (r0**12*x**4*(55*bs[0]**4 + 8*bs[1]*bs[2]*x))/16. +
    (51*bs[0]*r0**18*x**6*(7*bs[0]**5 + 2*bs[1]**3*x + 8*bs[0]*bs[1]*bs[2]*x))/16. +
    (7*r0**15*x**5*(78*bs[0]**5 + 5*bs[1]**3*x + 40*bs[0]*bs[1]*bs[2]*x))/64. +
    (7*r0**14*x**5*(117*bs[0]**3*bs[1] + 2*bs[2]**2*x))/72. +
    (17*r0**17*x**6*(315*bs[0]**4*bs[1] + 12*bs[1]**2*bs[2]*x + 16*bs[0]*bs[2]**2*x))/144. +
    (19*r0**19*x**7*(51*bs[0]**3*bs[1]**2 + 34*bs[0]**4*bs[2] + bs[1]*bs[2]**2*x))/16.)

    def Roche(r):
        return pot - (1./r + 0.5 * r**2 * x * (bs[0] + 0.5*bs[1]*x*r**2 + 1./3. * bs[2] * x**2 * r**4))

    r = newton(Roche, r_start, maxiter = 1000)

    return r

def nekmin(omega_in,q,x0=0.5,z0=0.5):

    '''Computes the position of the neck (minimal radius) in an contact_binary star1'''

    def Omega_xz(q,x,z):
        return 1./np.sqrt(x**2+z**2)+q/np.sqrt((1-x)**2+z**2)+(q+1)*x**2/2.-q*x

    def Omega_xy(q,x,y):
        return 1./np.sqrt(x**2+y**2)+q/np.sqrt((1-x)**2+y**2)+(q+1)*(x**2+y**2)/2.-q*x

    def dOmegadx_z(q,x,z):
        return -x/(x**2+z**2)**(3./2)+q*(1-x)/((1-x)**2+z**2)**(3./2.)+(q+1)*x-q

    def dOmegadx_y(q,x,y):
        return -x/(x**2+y**2)**(3./2)+q*(1-x)/((1-x)**2+y**2)**(3./2.)+(q+1)*x-q

    def dOmegadz(q,x,z):
        return -z/(x**2+z**2)**(3./2)-q*z/((1-x)**2+z**2)**(3./2.)

    def dOmegady(q,x,y):
        return -y/(x**2+y**2)**(3./2)-q*y/((1-x)**2+y**2)**(3./2.)+(q+1)*y

    def d2Omegadx2_z(q,x,z):
        return (2*x**2-z**2)/(x**2+z**2)**(5./2)+q*(2*(1-x)**2-z**2)/((1-x)**2+z**2)**(5./2)+(q+1)

    def d2Omegadx2_y(q,x,y):
        return (2*x**2-y**2)/(x**2+y**2)**(5./2)+q*(2*(1-x)**2-y**2)/((1-x)**2+y**2)**(5./2)+(q+1)

    def d2Omegadxdz(q,x,z):
        return 3*x*z/(x**2+z**2)**(5./2)-3*q*x*(1-x)/((1-x)**2+z**2)**(5./2)

    def d2Omegadxdy(q,x,y):
        return 3*x*y/(x**2+y**2)**(5./2)-3*q*x*(1-x)/((1-x)**2+y**2)**(5./2)

    xz,z = x0,z0
    dxz, dz = 1.,1.

    # find solution in xz plane
    while abs(dxz)>1e-8 and abs(dz)>1e-8:

        delz = 1.
        z=0.05
        while abs(delz) > 0.000001:
            delom = omega_in - Omega_xz(q,xz,z)
            delz = delom/dOmegadz(q,xz,z)
            z = abs(z+delz)

        DN = np.array([[dOmegadx_z(q,xz,z),dOmegadz(q,xz,z)],[d2Omegadx2_z(q,xz,z),d2Omegadxdz(q,xz,z)]])
        EN = np.array([omega_in-Omega_xz(q,xz,z),(-1)*dOmegadx_z(q,xz,z)])

        a,b,c,d = DN[0][0],DN[0][1],DN[1][0],DN[1][1]

        if (a*d-b*c)!=0.:
            DNINV = 1./(a*d-b*c)*np.array([[d,(-1)*b],[(-1)*c,d]])
            #DNINV = inv(DN)

            dd = np.dot(DNINV,EN)
            dxz,dz = dd[0],dd[1]
            xz=xz+dxz
            z=z+dz
        else:
            xz = xz+0.5
            z = z+0.5
            dxz = 1.
            dz = 1.

    return xz,z