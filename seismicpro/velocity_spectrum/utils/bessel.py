import numpy as np
from numba import njit, prange


PP = np.array([
    7.96936729297347051624E-4,
    8.28352392107440799803E-2,
    1.23953371646414299388E0,
    5.44725003058768775090E0,
    8.74716500199817011941E0,
    5.30324038235394892183E0,
    9.99999999999999997821E-1,
    ], dtype=np.float64)

PQ = np.array([
    9.24408810558863637013E-4,
    8.56288474354474431428E-2,
    1.25352743901058953537E0,
    5.47097740330417105182E0,
    8.76190883237069594232E0,
    5.30605288235394617618E0,
    1.00000000000000000218E0,
    ], dtype=np.float64)

QP = np.array([
    -1.13663838898469149931E-2,
    -1.28252718670509318512E0,
    -1.95539544257735972385E1,
    -9.32060152123768231369E1,
    -1.77681167980488050595E2,
    -1.47077505154951170175E2,
    -5.14105326766599330220E1,
    -6.05014350600728481186E0,
    ], dtype=np.float64)

QQ = np.array([
    6.43178256118178023184E1,
    8.56430025976980587198E2,
    3.88240183605401609683E3,
    7.24046774195652478189E3,
    5.93072701187316984827E3,
    2.06209331660327847417E3,
    2.42005740240291393179E2,
    ], dtype=np.float64)

YP= np.array([
    1.55924367855235737965E4,
    -1.46639295903971606143E7,
    5.43526477051876500413E9,
    -9.82136065717911466409E11,
    8.75906394395366999549E13,
    -3.46628303384729719441E15,
    4.42733268572569800351E16,
    -1.84950800436986690637E16,
    ], dtype=np.float64)

YQ= np.array([
   1.04128353664259848412E3,
    6.26107330137134956842E5,
    2.68919633393814121987E8,
    8.64002487103935000337E10,
    2.02979612750105546709E13,
    3.17157752842975028269E15,
    2.50596256172653059228E17,
    ], dtype=np.float64)

RP = np.array([
    -4.79443220978201773821E9,
    1.95617491946556577543E12,
    -2.49248344360967716204E14,
    9.70862251047306323952E15,
    ], dtype=np.float64)

RQ = np.array([
    4.99563147152651017219E2,
    1.73785401676374683123E5,
    4.84409658339962045305E7,
    1.11855537045356834862E10,
    2.11277520115489217587E12,
    3.10518229857422583814E14,
    3.18121955943204943306E16,
    1.71086294081043136091E18,
    ], dtype=np.float64)

DR1 = np.float64(5.78318596294678452118E0)
DR2 = np.float64(3.04712623436620863991E1)

SQ2OPI = (2 / np.pi) ** 0.5
M_PI_4 = np.pi / 4
M_2_PI = 2 / np.pi

@njit
def polevl(x, coef):
    res = 0
    for c in coef:
        res = res * x + c
    return res

@numba.njit(fastmath=True, parallel=True)
def j0(array):
    """ First kind zero order Bessel function."""
    res = np.empty_like(array)
    for i in numba.prange(len(array)):
        x = array[i]
        if x < 0:
            x = -x
        if x <= 5.0:
            z = x * x
            if x < 1e-5:
                res[i] =  1.0 - z / 4.0
                continue
            p = (z - DR1) * (z - DR2)
            p = p * polevl(z, RP) / polevl(z, RQ)
            res[i] = p
        else:
            w = 5.0 / x
            q = 25.0 / (x * x)
            p = polevl(q, PP[-6:]) / polevl(q, PQ[-6:])
            q = polevl(q, QP[-7:]) / polevl(q, QQ[-7:]);
            xn = x - np.pi / 4
            p = p * np.cos(xn) - w * q * np.sin(xn)
            res[i] =  (p * SQ2OPI / np.sqrt(x))
    return res

@numba.njit(fastmath=True, parallel=True)
def y0(array):
    """ Second kind zero order Bessel function."""
    res = np.empty_like(array)
    j0_x = j0(array)
    for i in numba.prange(len(array)):
        x = array[i]
        if x <= 5.0:
            if x == 0:
                res[i] = -np.inf
                continue
            elif x < 0:
                res[i] = np.nan
                continue
            z = x * x
            w = polevl(z, YP[-7:]) / polevl(z, YQ[-7:])
            w += M_2_PI * np.log(x) *j0_x[i]
            res[i] = w
        else:
            w = 5.0 / x
            z = 25.0 / (x * x)
            p = polevl(z, PP[-6:]) / polevl(z, PQ[-6:])
            q = polevl(z, QP[-7:]) / polevl(z, QQ[-7:])
            xn = x - M_PI_4
            p = p * np.sin(xn) + w * q * np.cos(xn)
            res[i] =  (p * SQ2OPI / np.sqrt(x))
    return res