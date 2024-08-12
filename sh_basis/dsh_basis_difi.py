import numpy as np

from sh_basis.utils import lmax2N
from sh_basis.legendre import legendre


def dsh_basis_difi(lmax, grid):
    rad = np.pi / 180
    N_data = grid.shape[1]

    cos_theta = np.cos(grid[0] * rad)
    sin_theta = np.sin(grid[0] * rad)
    # convert to row vector if input parameter is scalar
    # number of parameters

    N_nm = lmax2N(lmax)
    A_r_0 = np.zeros((N_nm, N_data))
    A_theta_0 = np.zeros((N_nm, N_data))
    A_phi_0 = np.zeros((N_nm, N_data))
    # Cycles through all the coefficients for degree n, order m model.
    k = 0
    Pnm = np.zeros((lmax+2, lmax+1, np.size(cos_theta, 0)))
    dPnm = np.zeros((lmax+2, lmax+1, np.size(cos_theta, 0)))

    [Pnm, dPnm] = legendre(90-grid[0], lmax)

    # Cycle through all of n and m
    for n in range(1, lmax+1):
        rn1 = (6371.2 / grid[2])**(n+2)
        for m in range(n+1):
            index = int(n * (n + 1) / 2 + m)
            if m == 0:
                # no h terms for g10, g20, g30 etc...
                A_r_0[k] = (n+1)*rn1*Pnm[index]
                A_theta_0[k] = rn1*dPnm[index]
                A_phi_0[k] = -rn1*0
                k += 1
            else:
                alpha = m*grid[1]*rad
                cos_phi = np.cos(alpha)
                sin_phi = np.sin(alpha)
                # g terms, as in g11, g21, g22 etc...
                A_r_0[k] = (n+1)*rn1*Pnm[index]*cos_phi
                A_theta_0[k] = rn1*dPnm[index]*cos_phi
                A_phi_0[k] = -rn1*Pnm[index]/sin_theta*(-m*sin_phi)
                k += 1
                # h terms as in h11, h21, h22 etc...
                A_r_0[k] = (n+1)*rn1*Pnm[index]*sin_phi
                A_theta_0[k] = rn1*dPnm[index]*sin_phi
                A_phi_0[k] = -rn1*Pnm[index]/sin_theta*m*cos_phi
                k += 1

    return A_r_0, A_theta_0, A_phi_0
