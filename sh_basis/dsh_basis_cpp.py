"""
Utility functions for `paleokalmag`.
"""

from warnings import warn

import numpy as np

from sh_basis.utils import lm2i, REARTH


def _fix_z(z):
    """ Hotfix for single inputs. """
    try:
        z.shape[1]
        return z
    except IndexError:
        return np.atleast_2d(z).T


try:
    from _csh_basis import _dspharm

    def dsh_basis_cpp(lmax, z, R=REARTH, internal=True, mmax=None):
        # If you ever encounter a problem with this version, check that your
        # z is fortran order!
        _z = _fix_z(z)[0:3]
        if not np.isfortran(_z):
            _z = np.asfortranarray(_z)
        if mmax is None:
            return _dspharm(int(lmax), _z, R, internal, int(lmax))
        if lmax < mmax:
            raise ValueError(
                "mmax has to be smaller than or equal to lmax."
            )
        inds = []
        for ell in range(1, lmax+1):
            for emm in range(min(ell, mmax)+1):
                if emm == 0:
                    inds.append(lm2i(ell, emm))
                    continue

                inds.append(lm2i(ell, emm))
                inds.append(lm2i(ell, -emm))

        ret = _dspharm(int(lmax), _z, R, internal, int(mmax))
        return ret[inds]

except ImportError as e:
    print(e)
    warn("c++-accelerated version could not be loaded. Falling back to "
         "python.",
         UserWarning)
    from utils import dsh_basis as _dsh_basis

    def dsh_basis_cpp(lmax, z, R=REARTH, internal=True):
        if not internal:
            raise ValueError(
                "pymagglobal implementation doesn't support external sources."
            )
        z = _fix_z(z)
        return _dsh_basis(int(lmax), z, R=R)

finally:
    dsh_basis_cpp.__doc__ = \
        """
        Evaluate the magnetic field basis functions (derivatives of spherical
        harmonics).

        Parameters
        ----------
        lmax : int
            The maximum spherical harmonics degree
        z : array
            The points at which to evaluate the basis functions, given as

            * z[0, :] contains the colatitude in degrees.
            * z[1, :] contains the longitude in degrees.
            * z[2, :] contains the radius in kilometers.

            Any additional columns are ignored.
        R : float, optional
            The reference radius in kilometers. Default is REARTH=6371.2 km.
        internal : bool, optional, c++ version only
            Whether the sources are of internal (default) or external origin.
        mmax : int, optional, c++ version only
            Maximum spherical harmonics order. Sometimes it is desirable to
            let the degree run higher than the order, e.g. in ionospheric
            modeling. If None, go until lmax (default).

        Returns
        -------
        array
            An array of shape (lmax*(lmax+2), 3*z.shape[1]), containing the
            basis functions.
        """
