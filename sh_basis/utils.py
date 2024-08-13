import numpy as np
from scipy.special import lpmn, gammaln

# Earth's radius in km
REARTH = 6371.2


def lmax2N(lmax):
    '''Returns the number of Gauss coefficients up to and including the degree
    lmax.

    Parameters
    ----------
    lmax : int
        The maximal spherical harmonic degree.

    Returns
    -------
    int
        The number of Gauss coefficients.
    '''
    _intcheck(lmax)
    if np.any(lmax < 1):
        raise ValueError("Degree < 1")
    return lmax * (lmax+2)


def _intcheck(x):
    '''Cast an array or scalar to integer'''
    if not isinstance(x, int):
        raise ValueError(f"Input has to be integer and not {type(x)}.")


def lm2i(ell, m):
    '''Returns the "standard order" index corresponding to a Gauss coefficient
    of degree l and order m. The "standard order" is as follows

    +-----+-----+-----+
    |  i  |  l  |  m  |
    +=====+=====+=====+
    |  0  |  1  |  0  |
    +-----+-----+-----+
    |  1  |  1  |  1  |
    +-----+-----+-----+
    |  2  |  1  | -1  |
    +-----+-----+-----+
    |  3  |  2  |  0  |
    +-----+-----+-----+
    |  4  |  2  |  1  |
    +-----+-----+-----+
    |  5  |  2  | -1  |
    +-----+-----+-----+
    |  6  |  2  |  2  |
    +-----+-----+-----+
    |  7  |  2  | -2  |
    +-----+-----+-----+
    |       ...       |
    +-----+-----+-----+

    Parameters
    ----------
    ell : int
        The degree of a Gauss coefficient.
    m : int
        The order of a Gauss coefficient.

    Returns
    -------
    int
        The corresponding index.

    Examples
    --------
    >>> lm2i(1, 0)
    0
    >>> lm2i(4, -1)
    17
    '''
    _intcheck(ell)
    _intcheck(m)
    if np.any(ell < 1):
        raise ValueError("Degree < 1")
    if np.any(ell < np.abs(m)):
        raise ValueError("Order |m|>l")
    i = ell*ell - 1 + 2 * np.abs(m)
    return int(i - (0 < m))


def i2lm_l(i):
    '''Returns the degree l of a Gauss coefficient at index i, in the "standard
    order". The "standard order" is as follows

    +-----+-----+-----+
    |  i  |  l  |  m  |
    +=====+=====+=====+
    |  0  |  1  |  0  |
    +-----+-----+-----+
    |  1  |  1  |  1  |
    +-----+-----+-----+
    |  2  |  1  | -1  |
    +-----+-----+-----+
    |  3  |  2  |  0  |
    +-----+-----+-----+
    |  4  |  2  |  1  |
    +-----+-----+-----+
    |  5  |  2  | -1  |
    +-----+-----+-----+
    |  6  |  2  |  2  |
    +-----+-----+-----+
    |  7  |  2  | -2  |
    +-----+-----+-----+
    |       ...       |
    +-----+-----+-----+

    Parameters
    ----------
    i : int
        The index of a Gauss coefficient.

    Returns
    -------
    int
        The corresponding degree.

    Examples
    --------
    >>> i2lm_l(0)
    1
    >>> i2lm_l(21)
    4
    '''
    _intcheck(i)
    if i < 0:
        raise ValueError("Index has to be non-negative.")
    return int(np.sqrt(i+1))


def i2lm_m(i):
    '''Returns the order m of a Gauss coefficient at index i, in the "standard
    order". The "standard order" is as follows

    +-----+-----+-----+
    |  i  |  l  |  m  |
    +=====+=====+=====+
    |  0  |  1  |  0  |
    +-----+-----+-----+
    |  1  |  1  |  1  |
    +-----+-----+-----+
    |  2  |  1  | -1  |
    +-----+-----+-----+
    |  3  |  2  |  0  |
    +-----+-----+-----+
    |  4  |  2  |  1  |
    +-----+-----+-----+
    |  5  |  2  | -1  |
    +-----+-----+-----+
    |  6  |  2  |  2  |
    +-----+-----+-----+
    |  7  |  2  | -2  |
    +-----+-----+-----+
    |       ...       |
    +-----+-----+-----+

    Parameters
    ----------
    i : int
        The index of a Gauss coefficient.

    Returns
    -------
    int
        The corresponding order.

    Examples
    --------
    >>> i2lm_m(0)
    0
    >>> i2lm_m(21)
    -3
    '''
    _intcheck(i)
    if i < 0:
        raise ValueError("Index has to be non-negative.")
    ell = i2lm_l(i)
    j = i + 1 - ell*ell

    return int((-1)**(1-j % 2) * (j + j % 2)/2)


def equi_sph(n, twopi=True):
    '''Regularly place points on the surface of the unit sphere [Deserno]_. the
       number of output points may differ from n.

    Parameters
    ----------
    n : int
        The approximate number of points
    twopi : bool, optional
        Whether to include 2*pi as a duplicate of 0 for the longitutinal angle
        (useful for plotting purposes).

    Returns
    -------
    array of shape (2, m)
        polar and azimutal angles in radians of m points, that are
        equally spaced on the sphere. m is approximately n.

    References
    ----------
    .. [Deserno] M. Deserno, "How to generate equidistributed points on the
          surface of a sphere", Max-Planck-Institut für Polymerforschung, 2004.
    '''
    _intcheck(n)
    a = 4*np.pi/n
    d = np.sqrt(a)
    m_t = int(round(np.pi/d))
    d_t = np.pi/m_t
    d_p = a/d_t
    ts = []
    ps = []
    for m in range(m_t):
        t = np.pi*(m+0.5)/m_t
        m_p = int(round(2*np.pi*np.sin(t)/d_p))
        for n in range(m_p + 1 if twopi else m_p):
            ts.append(t)
            ps.append(2*np.pi*n/m_p)
    return np.array([ts, ps], dtype=float)


def get_grid(n, R=REARTH, t=1900., random=False, twopi=True):
    '''Get input points on the sphere. This is a convenience routine to allow
       easy construction of field maps and synthetic data from the models.

    Parameters
    ----------
    n : int
        The approximate number of points. Due to the points being equally
        spaced on the sphere, the actual number may be slightly higher.
    R : float, optional
        The radius of the sphere. By default this is the Earth's radius.
    t : float, optional
        The epoch at which the points are generated.
    random : bool, optional
        If true, exactly n random points are returned. This is useful for
        generating synthetic data.
    twopi : bool, optional
        Whether to include 2*pi as a duplicate of 0 for the longitutinal angle
        (useful for plotting purposes). Only used if random is False.

    Returns
    -------
    grid : array of shape (4, n')
        * grid[0] contains colatitudes in degrees.
        * grid[1] contains longitudes in degrees.
        * grid[2] contains radii in km.
        * grid[3] contains dates in years.

        n' is approximately n.
    '''
    if random:
        n_ret = n
        n = 100*n
        twopi = False

    angles = equi_sph(n, twopi=twopi)
    grid = np.zeros((4, angles.shape[1]), order='F')
    grid[0] = np.rad2deg(angles[0])
    grid[1] = np.rad2deg(angles[1])
    grid[2] = R
    grid[3] = t

    if random:
        inds = np.random.choice(grid.shape[1], size=n_ret, replace=False)
        return grid[:, inds]
    else:
        return grid


def dsh_basis(lmax, z, out=None, R=REARTH):
    '''Write the magnetic field basis functions, evaluated at points z, into
    the array out. These are basically the derivatives of the spherical
    harmonics in spherical coordinates with some scaling factors applied. Note
    that it is assumed that the coefficients that are multiplied to this basis
    are given at the radius R. This implementation is based on a specific
    recursion of the Legendre polynomials, to guarantee sane behavior at the
    poles. See [Du]_ for further details.

    Parameters
    ----------
    lmax : int
        The maximal spherical harmonics degree.
    z : array of shape (3, n)
        The points at which to evaluate the basis functions, given as
            * z[0, :] contains the colatitude in degrees.
            * z[1, :] contains the longitude in degrees.
            * z[2, :] contains the radius in kilometers.

    out : array of shape (lmax2N(lmax), 3*n), optional
        The output array in which the basis functions are to be stored. This is
        included for historic reasons. Originally the function was to be
        replacable by a C-based function from the related pyfield library at
        a later stage [FieldTools]_.
    R : float, optional
        The reference radius for the coefficients.

    Returns
    -------
    out : array of shape (lmax2N(lmax), 3*n)
        Array in which the basis functions are stored, as follows:
            * out[i, 0::3]  contains the North component of the basis
                            corresponding to degree l and order m,
                            where i = lm2i(l, m).
            * out[i, 1::3] contains the East component of the basis.
            * out[i, 2::3] contains the Down component of the basis.

    References
    ----------
    .. [Du] J. Du, "Non-singular spherical harmonic expressions of geomagnetic
        vector and gradient tensor fields in the local north-oriented reference
        frame.", Geosci. Model Dev., vol. 8, pages 1979-1990, 2014.
    .. [FieldTools] H. Matuschek and S. Mauerberger, "FieldTools - A toolbox
        for manipulating vector fields on the sphere", GFZ Data Services, 2019.
        DOI: `10.5880/fidgeo.2019.033
        <http://doi.org/10.5880/fidgeo.2019.033>`_
    '''
    _intcheck(lmax)
    # check input consitency
    N = lmax2N(lmax)

    if out is None:
        out = np.empty((N, 3*z.shape[1]))
    if out.shape[0] != N:
        raise ValueError(f"Inconsistent input. Number of coefficients is {N}, "
                         f"but the given array is of shape {out.shape}.")
    try:
        if out.shape[1] != 3*z.shape[1]:
            raise ValueError(f"Inconsistent input. {z.shape[1]} points "
                             f"given, but the given field-array is of shape "
                             f"{out.shape}.")
    except IndexError:
        raise IndexError(f"Both input arrays have to be two dimensional. "
                         f"Inputs are of shape {z.shape} and {out.shape}.")

    # XXX z is in degrees!!!
    cos_t = np.cos(np.deg2rad(z[0, :]))
    out[:, :] = 0.
    Plm_all = np.array([lpmn(lmax, lmax, x)[0] for x in cos_t])
    # Degree l=0 is needed by one of the recurrence formulas
    for ell in range(0, lmax + 1):
        for m in range(0, ell + 1):
            Plm = Plm_all[:, m, ell]

            # North Component #
            # The following recurrence formula is used:
            # d_theta P_l^m = -1/2[(l+m)(l-m+1)P_l^{m-1} - P_l^{m+1}]
            #
            # P_l^{-1} term is treated separately
            if (0 < ell and m == 1):
                i = lm2i(ell, 0)  # According order is zero
                # The pre-factors of the recurrence formula and the scaling for
                # negative orders cancel out. The plus sign accounts for the
                # Condon-Shortley phase
                out[i, 0::3] += Plm/2
            # P_l^{m-1} term: m -> m+1
            if (0 < ell and m < ell):  # Skip l=0; m=0, ..., l-1
                i = lm2i(ell, m+1)
                # Zero and positive orders
                out[i, 0::3] -= (ell+m+1)*(ell-m)*Plm/2
                # Negative orders
                if (m+1 != 0):  # Do not visit m=0 twice
                    out[i+1, 0::3] -= (ell+m+1)*(ell-m)*Plm/2
            # P_l^{m+1} term: m -> m-1
            if (0 < ell and 0 < m):  # Skip l=0; m=1, ..., l
                i = lm2i(ell, m-1)
                # Zero and positive orders
                out[i, 0::3] += Plm/2
                # Negative orders
                if (m-1 != 0):  # Do not visit m=0 twice
                    out[i+1, 0::3] += Plm/2

            # East Component
            # The following recurrence formulas is used:
            # P_l^m/sin(theta)
            #       = -1/(2m)[P_{l-1}^{m-1} + (l+m-1)(l+m)P_{l-1}^{m-1}]
            # Does not visit m=0 at all since it is zero anyway
            # P_{l-1}^{m+1} term: l -> l+1 and m -> m-1
            if (ell < lmax and 1 < m):  # Skip l=lmax; m=2, ..., l
                i = lm2i(ell+1, m-1)
                out[i, 1::3] -= Plm/2/(m-1)
                out[i+1, 1::3] -= Plm/2/(m-1)
            # P_{l-1}^{m-1} term: l -> l+1 and m -> m+1
            if (ell < lmax):  # Skip l=lmax; m=0, ..., l
                i = lm2i(ell+1, m+1)
                out[i, 1::3] -= (ell+m+1)*(ell+m+2)*Plm/2/(m+1)
                out[i+1, 1::3] -= (ell+m+1)*(ell+m+2)*Plm/2/(m+1)

            # Down Component
            if (0 < ell):  # Skip l=0; m=0, ..., l
                i = lm2i(ell, m)
                # Zero and positive orders
                out[i, 2::3] = Plm
                # Negative orders
                if m != 0:  # Do not visit m=0 twice
                    out[i+1, 2::3] = Plm

    # Arrays of degrees and orders
    m = np.fromiter(map(i2lm_m, range(N)), int, N).reshape(-1, 1)
    ell = np.fromiter(map(i2lm_l, range(N)), int, N).reshape(-1, 1)

    # sqrt 2 to account for real form
    out *= np.sqrt(2 - (m == 0))
    # Condon-Shortley Phase
    out *= np.where(m % 2, -1, 1)

    # Schmidt semi-norm
    out *= np.exp((gammaln(ell-np.abs(m)+1) - gammaln(ell+np.abs(m)+1))/2)

    # North component, 1:2 instead of 1 for easier broadcasting
    out[:, 0::3] *= np.cos((m < 0)*np.pi/2 - np.abs(m)*np.deg2rad(z[1:2, :]))
    # East Component
    out[:, 1::3] *= np.sin((m < 0)*np.pi/2 - np.abs(m)*np.deg2rad(z[1:2, :]))
    out[:, 1::3] *= -np.abs(m)
    # Down component
    out[:, 2::3] *= np.cos((m < 0)*np.pi/2 - np.abs(m)*np.deg2rad(z[1:2, :]))
    out[:, 2::3] *= -(ell+1)

    # Scaling for sources of internal origin
    out *= np.repeat((R/z[2:3, :])**(ell+2), 3, axis=1)

    return out


def scaling(r_from, r_to, lmax):
    '''Calculate the scaling matrix for Gauss-coefficients'''
    res = np.ones(lmax2N(lmax))
    for i in (np.arange(lmax)+1):
        i = int(i)  # lm2i expects integers...
        res[lm2i(i, 0):lm2i(i, -i)+1] *= (r_from/r_to)**(i+2)
    return res


def yr2lt(times):
    '''Translate times given in years CE into kilo-years `before present
    <https://en.wikipedia.org/wiki/Before_Present>`_ often used for
    longterm models.

    Parameters
    ----------
    times: float or int
        Years CE.

    Returns
    -------
    float
       Kilo years before present (backward counting from 1/1/1950).

    Examples
    --------
    >>> yr2lt(1950)
    0.0
    >>> yr2lt(-50)
    2.0
    '''
    return -(times - 1950) / 1000


def lt2yr(times):
    '''Translate kilo-years `before present
    <https://en.wikipedia.org/wiki/Before_Present>`_ into years CE.

    Parameters
    ----------
    times: float or int
       Kilo-years before present (backward counting from 1/1/1950).

    Returns
    -------
    float or int
        Years CE.

    Examples
    --------
    >>> lt2yr(0)
    1950
    >>> lt2yr(2.)
    -50.0
    '''
    return -times*1000 + 1950
