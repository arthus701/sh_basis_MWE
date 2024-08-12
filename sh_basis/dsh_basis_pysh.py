import numpy as np
import pyshtools as pysh


def dsh_basis_pysh(
    maxdegree: int,
    loc: np.ndarray,
) -> np.ndarray:
    """
    Calculates the frechet matrix for the given stations and maximum degree
    Parameters
    ----------
    maxdegree
        Maximum spherical degree
    loc
        coordinates of stations. Each row contains:
            colatitude in radians
            longitude in radians
            radius in km

    Returns
    -------
    frechxyz
        size = stations X 3 X nr_coeffs matrix:
        contains the frechet coefficients first dx, then dy, then dz
    """
    schmidt_total = int((maxdegree+1) * (maxdegree + 2) / 2)
    frechxyz = np.zeros(((maxdegree+1)**2 - 1, loc.shape[1], 3))
    schmidt_p = np.zeros((loc.shape[1], schmidt_total))
    schmidt_dp = np.zeros((loc.shape[1], schmidt_total))
    schmidt_p = np.zeros((loc.shape[1], schmidt_total))
    schmidt_dp = np.zeros((loc.shape[1], schmidt_total))
    for it in range(loc.shape[1]):
        schmidt_p[it], schmidt_dp[it] = \
            pysh.legendre.PlmSchmidt_d1(
                maxdegree,
                np.cos(np.deg2rad(loc[0, it]))
            )
    schmidt_dp *= -np.sin(np.deg2rad(loc[0]))[:, None]
    counter = 0
    # dx, dy, dz in separate rows to increase speed
    for n in range(1, maxdegree+1):
        index = int(n * (n+1) / 2)
        mult_factor = (6371.2 / loc[2]) ** (n+1)
        # first g_n^0
        frechxyz[counter, :, 0] = mult_factor * schmidt_dp[:, index]
        # frechxyz[counter, 1] = 0
        frechxyz[counter, :, 2] = -mult_factor * (n+1) * schmidt_p[:, index]
        counter += 1
        for m in range(1, n+1):
            # Then the g-elements
            frechxyz[counter, :, 0] = mult_factor\
                * np.cos(m * np.deg2rad(loc[1])) * schmidt_dp[:, index+m]
            frechxyz[counter, :, 1] = m / np.sin(np.deg2rad(loc[0])) \
                * mult_factor \
                * np.sin(m * np.deg2rad(loc[1])) * schmidt_p[:, index+m]
            frechxyz[counter, :, 2] = -mult_factor * (n+1)\
                * np.cos(m * np.deg2rad(loc[1])) * schmidt_p[:, index+m]
            counter += 1
            # Now the h-elements
            frechxyz[counter, :, 0] = mult_factor\
                * np.sin(m * np.deg2rad(loc[1])) * schmidt_dp[:, index+m]
            frechxyz[counter, :, 1] = -m / np.sin(np.deg2rad(loc[0])) \
                * mult_factor \
                * np.cos(m * np.deg2rad(loc[1])) * schmidt_p[:, index+m]
            frechxyz[counter, :, 2] = -mult_factor * (n+1)\
                * np.sin(m * np.deg2rad(loc[1])) * schmidt_p[:, index+m]
            counter += 1

    return frechxyz
