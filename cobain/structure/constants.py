import astropy.constants as const

G = const.G.value  # G in si units, needs to be converted in solar
GG_sol = const.G.value * const.M_sun.value / (const.R_sun.value ** 3)
mp_kB_sol = const.m_p.value / const.k_B.value * (const.R_sun.value) ** 2
stefb = 5.67036713 * 1e-8
stefb_sol = 0.5 * stefb * 1e-30