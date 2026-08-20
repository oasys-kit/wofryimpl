#
# **Focusing with Laue crystal** Theory and equations from:
# Guigay and Ferrero "Dynamical focusing by bent Laue crystals" Acta Cryst. (2016). A72, 489–499
#
"""
Semi-analytical crystal propagator for a cylindrically bent Laue crystal.

This module computes the Bragg-reflected wavefield downstream of a flat or
cylindrically bent perfect crystal in Laue (transmission) geometry, following
the dynamical-theory (Takagi-Taupin) solutions of:

  * Guigay, Ferrero et al., Acta Cryst. (2013) A69, 91-97   (symmetric case),
  * Guigay & Ferrero,       Acta Cryst. (2016) A72, 489-499 (asymmetric case).

The diffracted amplitude on the exit surface is obtained from an influence
function (the crystal propagator): a Bessel function J0 in the symmetric case
(asymmetry angle alfa = 0) and a confluent hypergeometric / Kummer function
M(i*beta, 1, i*y) in the asymmetric case. The field at any downstream distance
q is then obtained by Fresnel propagation, which for a point source at distance
p reduces to the single-integral expressions of GF2016 (eqs. 23, 24, 30, 31).

Conventions and units
---------------------
* All lengths are in millimetres (R, thickness, p, q), angles in radians
  internally (alfa is given in degrees), photon energy in keV.
* x is the transverse coordinate (mm) on the observation plane; q = 0 is the
  exit surface, perpendicular to the Bragg-diffracted direction.
* Two absorption mechanisms are present: the *anomalous* (Borrmann) absorption,
  always included through the complex parameters Z and omega, and the *normal*
  (photoelectric) absorption factor att = exp(-k*(t1+t2)/2*Im(chi0)), which can
  be switched off with apply_absorption=False (to reproduce paper2013.py).
"""

import numpy

try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid  # numpy < 2.0 has no numpy.trapezoid
import mpmath
import scipy
import time

from scipy.special import jv as BesselJ
import scipy.constants as codata
from scipy import interpolate

from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
from wofryimpl.util import materials_library as ml

def hyp1f1_series_small(a, b, z, terms=20):
    """
    Confluent hypergeometric (Kummer) function M(a, b, z) by its power series,
    accurate for small |z|.

    Parameters
    ----------
    a, b : complex
        Kummer parameters (here a = i*beta, b = 1).
    z : complex
        Argument (here z = i*yprime).
    terms : int, optional
        Maximum number of series terms; the sum stops early once a term falls
        below 1e-15 in magnitude. Default 20.

    Returns
    -------
    complex
        The value of M(a, b, z).
    """
    result = 1.0
    term = 1.0
    for k in range(1, terms):
        term *= (a + k - 1) * z / ((b + k - 1) * k)
        result += term
        if abs(term) < 1e-15:
            break
    return result

# FIX (bug #6, 2026): cache of (Gamma(a), Gamma(1-a)) used by the asymptotic branch of
# fast_hyp1f1; a = 1j*kap is fixed for a given crystal configuration, so the two mpmath
# gamma evaluations are paid only once.
_GAMMA_CACHE = {}

def _cached_gammas(a):
    key = complex(a)
    if key not in _GAMMA_CACHE:
        _GAMMA_CACHE[key] = (complex(mpmath.gamma(key)), complex(mpmath.gamma(1 - key)))
    return _GAMMA_CACHE[key]

def fast_hyp1f1(kap, yprime):
    """
    Fast evaluation of the Kummer function M(i*kap, 1, i*yprime).

    Chooses the cheapest accurate method for the magnitude of the argument:
    a first-order expansion for |yprime| < 1e-8, the power series
    (:func:`hyp1f1_series_small`) when it converges safely, the two-term
    large-argument asymptotic form when its error is below ~1e-3, and
    ``mpmath.hyp1f1`` otherwise. It is a drop-in replacement for
    ``mpmath.hyp1f1(1j*kap, 1, 1j*yprime)`` used when ``use_fast_hyp1f1`` is
    enabled.

    FIX (bug #6, 2026): the previous version was wrong in two regimes.
    (i) Large |yprime|: it kept a single (scrambled) asymptotic term, but for
    a purely imaginary argument z = i*y BOTH terms of the asymptotic expansion
    M(a,1,z) ~ (-z)^(-a)/Gamma(1-a) + e^z z^(a-1)/Gamma(a)  [DLMF 13.7.2]
    are of comparable magnitude -- they are the two Borrmann branches -- so
    dropping one changed the predicted focusing completely (the foci appeared
    at the alpha-mirrored positions). (ii) Small |yprime| but large |kap|
    (small asymmetry angles): the 20-term series does not converge for
    |kap*yprime| >~ 20 and produced garbage. Both branches are now guarded and
    fall back to mpmath when their accuracy is not guaranteed.

    Parameters
    ----------
    kap : complex
        Kummer parameter beta = Omega / A (complex in general).
    yprime : float
        Real argument y' = A*gamma*(a^2 - v^2)/sin^2(2 theta_B).

    Returns
    -------
    complex
        The value of M(i*kap, 1, i*yprime).
    """
    kap = complex(kap)
    y = float(yprime)
    yp_abs = abs(y)

    if yp_abs < 1e-8:
        return 1.0 + 1j * (kap * y)

    # Series expansion: the terms are bounded by ((1+|kap|)|y|)^n / n!, so the largest
    # term is ~e^((1+|kap|)|y|) and float cancellation stays below ~1e-16*e^20 ~ 5e-8
    # under the guard below; the term count covers the peak at n ~ (1+|kap|)|y|.
    eff = (1 + abs(kap)) * yp_abs
    if eff < 20:
        return hyp1f1_series_small(1j * kap, 1, 1j * y, terms=40 + int(2 * eff))

    if yp_abs > 100 and (abs(kap) + 1) ** 2 / yp_abs < 1e-3:
        # Two-term asymptotic expansion, error O(|a|^2/|z|) < 1e-3 [DLMF 13.7.2 with b=1]
        z = 1j * y
        a = 1j * kap
        ga, g1a = _cached_gammas(a)
        t1 = numpy.exp(-a * numpy.log(-z)) / g1a          # (-z)^(-a) / Gamma(1-a)
        t2 = numpy.exp(z + (a - 1) * numpy.log(z)) / ga   # e^z z^(a-1) / Gamma(a)
        return complex(t1 + t2)

    return complex(mpmath.hyp1f1(1j * kap, 1, 1j * y))

class LaueCrystalFocusing():
    """
    Crystal propagator for a flat or cylindrically bent Laue crystal.

    Holds the crystal/geometry parameters and provides scans of the
    Bragg-reflected wavefield: transverse profiles at a fixed distance
    (:meth:`xscan` and its variants) and the on-axis intensity versus distance
    (:meth:`qscan`). The symmetric case (``alfa_deg == 0``) is evaluated with the
    Bessel-function influence function; the asymmetric case uses the Kummer
    function. See the module docstring for theory, conventions and units.
    """
    def __init__(self,
                 crystal_descriptor="Si",
                 hkl=[1, 1, 1],
                 R=2000,
                 poisson_ratio=0.2201,
                 photon_energy_in_keV=20.0,
                 thickness=0.250,  # mm
                 p=29000.0,  # mm
                 alfa_deg=2.0,  # CAN BE POSITIVE OR NEGATIVE)
                 integration_points=500,
                 use_fast_hyp1f1=0,
                 apply_absorption=True,
                 chih2=None,
                 verbose=1,
                 materials_library=None,
                 ):
            """
            Parameters
            ----------
            crystal_descriptor : str
                Crystal material name understood by xraylib (e.g. "Si").
            hkl : list of int
                Miller indices [h, k, l] of the reflection.
            R : float
                Radius of curvature in mm (positive: source faces the convex
                side, per GF2016). Use a large value for a flat crystal.
            poisson_ratio : float
                Poisson ratio nu of the (isotropic) crystal; enters rho=nu/(1-nu).
            photon_energy_in_keV : float
                Photon energy in keV.
            thickness : float
                Crystal thickness t in mm (normal to the surface).
            p : float
                Source-to-crystal distance in mm (p = 0: point source on the
                entrance surface).
            alfa_deg : float
                Asymmetry angle in degrees (0 = symmetric; may be + or -).
            integration_points : int
                Number of points used to evaluate the propagator integrals.
            use_fast_hyp1f1 : int
                Evaluation strategy for the Kummer function (asymmetric case only):
                0 = exact ``mpmath.hyp1f1``, point by point (original, slow);
                1 = :func:`fast_hyp1f1` approximations, point by point (original fast);
                2 = exact ``mpmath.hyp1f1`` computed once on the integration grid
                and cached (FEATURE 2026, recommended: exact results, the Kummer
                cost is paid a single time per scan or map).
            apply_absorption : bool
                If True (default) include the normal (photoelectric) absorption
                factor att = exp(-k*(t1+t2)/2*Im(chi0)). If False, set att = 1
                to reproduce paper2013.py. Anomalous (Borrmann) absorption is
                always included via the complex Z and omega.
            chih2 : complex or None
                If not None, override the computed chi_h*chi_hbar with this value
                (used to reproduce the exact susceptibility product quoted in a
                reference, e.g. GF2013). The default None uses the value derived
                from the structure factors (xraylib).
            verbose : int
                If truthy, print crystal data and progress information.
            """
            self._crystal_descriptor = crystal_descriptor
            self._hkl = hkl
            self._R = R # mm
            self._poisson_ratio = poisson_ratio
            self._photon_energy_in_keV = photon_energy_in_keV
            self._thickness = thickness # mm
            self._p = p # mm
            self._alfa_deg = alfa_deg  # CAN BE POSITIVE OR NEGATIVE
            self._integration_points = integration_points
            self._use_fast_hyp1f1 = use_fast_hyp1f1
            # FEATURE (2026): apply_absorption toggles the NORMAL (photoelectric) absorption factor
            # att = exp(-k*0.5*(t1+t2)*Im(chi0)). True (default) = physical behaviour; False reproduces
            # paper2013.py (which sets attsym=1). Does NOT affect anomalous (Borrmann) absorption,
            # which is always included via the complex Z and omega.
            self._apply_absorption = apply_absorption
            # FEATURE (2026): optional override of chi_h*chi_hbar. If not None, the value computed
            # from the structure factors is replaced by this number in all the _calculate_constats_*
            # helpers, e.g. to reproduce the exact chi_h*chi_hbar quoted in a reference such as GF2013.
            self._chih2 = chih2
            self._verbose = verbose
            if materials_library is None:
                self._materials_library = ml
            else:
                self._materials_library = materials_library

    def get_crystal_data(self):
        """
        Compute the Bragg angle and susceptibility Fourier coefficients with
        xraylib for the configured crystal, reflection and photon energy.

        The susceptibilities are obtained from the structure factors as
        chi = -r_e * lambda^2 * F / (pi * V).

        Returns
        -------
        braggAngle : float
            Bragg angle in radians.
        chi0 : complex
            Conjugate of the mean susceptibility chi_0 (so that Im(chi0) > 0 and
            the normal-absorption factor att decays in the GF2016 convention).
        chiH : complex
            The h susceptibility chi_h (as computed, NOT conjugated).
        chiHbar : complex
            The -h susceptibility chi_{-h} = chi_hbar (as computed). The product
            chiH*chiHbar then has Im > 0 for Si, matching the reference values,
            so no conjugation is needed when forming chih2 = chi_h*chi_hbar.
        """
        #
        # get crystal data for silicon crystal
        #
        cryst = ml.Crystal_GetCrystal(self._crystal_descriptor)

        # print some info
        if self._verbose:
            print("  Unit cell dimensions [A] are %f %f %f" % (cryst['a'], cryst['b'], cryst['c']))
            print("  Unit cell angles are %f %f %f" % (cryst['alpha'], cryst['beta'], cryst['gamma']))
            print("  Unit cell volume [A] is %f" % (cryst['volume']))

        #
        # define miller indices and compute dSpacing
        #

        hh = self._hkl[0]
        kk = self._hkl[1]
        ll = self._hkl[2]
        debyeWaller = 1.0
        rel_angle = 1.0  # ratio of (incident angle)/(bragg angle) -> we work at Bragg angle

        dspacing = ml.Crystal_dSpacing(cryst, hh, kk, ll)
        if self._verbose: print("dspacing: %f A" % dspacing)
        #
        # define energy and get Bragg angle
        #
        ener = self._photon_energy_in_keV  # 12.398 # keV
        braggAngle = ml.Bragg_angle(cryst, ener, hh, kk, ll)
        if self._verbose: print("Bragg angle: %f degrees" % (braggAngle * 180 / numpy.pi))

        #
        # get the structure factor (at a given energy)
        #
        f0 = ml.Crystal_F_H_StructureFactor(cryst, ener, 0, 0, 0, debyeWaller, 1.0)
        fH = ml.Crystal_F_H_StructureFactor(cryst, ener, hh, kk, ll, debyeWaller, 1.0)
        fHbar = ml.Crystal_F_H_StructureFactor(cryst, ener, -hh, -kk, -ll, debyeWaller, 1.0)
        if self._verbose: print("f0: (%f , %f)" % (f0.real, f0.imag))
        if self._verbose: print("fH: (%f , %f)" % (fH.real, fH.imag))
        if self._verbose: print("fHbar: (%f , %f)" % (fHbar.real, fHbar.imag))

        #
        # convert structure factor in chi (or psi) = - classical_e_radius wavelength^2 fH /(pi volume)
        #
        codata = scipy.constants.physical_constants
        codata_c,  _, _ = codata["speed of light in vacuum"]
        codata_h,  _, _ = codata["Planck constant"]
        codata_ec, _, _ = codata["elementary charge"]
        codata_r,  _, _ = codata["classical electron radius"]

        ev2meter = codata_h * codata_c / codata_ec
        wavelength = ev2meter / (ener * 1e3)
        if self._verbose: print("Photon energy: %f keV" % ener)
        if self._verbose: print("Photon wavelength: %f A" % (1e10 * wavelength))

        volume = cryst['volume'] * 1e-10 * 1e-10 * 1e-10  # volume of silicon unit cell in m^3
        cte = - codata_r * wavelength * wavelength / (numpy.pi * volume)

        chi0 = cte * f0
        chiH = cte * fH
        chiHbar = cte * fHbar

        if self._verbose: print("chi0: (%e , %e)" % (chi0.real, chi0.imag))
        if self._verbose: print("chiH: (%e , %e)" % (chiH.real, chiH.imag))
        if self._verbose: print("chiHbar: (%e , %e)" % (chiHbar.real, chiHbar.imag))

        # chi0 is conjugated so that the NORMAL-absorption factor att = exp[-k(t1+t2)/2*Im chi0]
        # decays (raw Im(chi0) < 0 in this xraylib convention). chiH and chiHbar (= chi_{-h}) are
        # returned as computed, so that chih2 = chiH*chiHbar reproduces the reference product directly
        # (Im > 0 for Si) with no conjugate "fix" needed downstream.
        # FIX (bug #7, 2026): cast to plain python scalars. The structure factors may come out of
        # the materials library as 1-element numpy arrays; downstream these reached
        # mpmath.hyp1f1 (via kap = Omega/A), which raises "TypeError: cannot create mpf from
        # array([...])", and they also propagated array-ness to every returned amplitude.
        braggAngle = float(numpy.asarray(braggAngle).ravel()[0])
        chi0 = complex(numpy.asarray(chi0).ravel()[0])
        chiH = complex(numpy.asarray(chiH).ravel()[0])
        chiHbar = complex(numpy.asarray(chiHbar).ravel()[0])
        return braggAngle, numpy.conjugate(chi0), chiH, chiHbar

    # FEATURE (2026): vectorized influence function with per-instance caching.
    def _influence_on_grid(self, v, a=None, alfa=None, k=None, teta=None, chih2=None,
                           acrist=None, gamma=None, kap=None, **kwargs):
        """
        Influence function (crystal propagator amplitude) evaluated on an ARRAY
        of transverse coordinates ``v``: the Bessel form J0(Z*sqrt(a^2-v^2)) for
        the symmetric case (alfa == 0), or the Kummer form M(i*kap, 1, i*y')
        with y' = acrist*gamma*(a^2-v^2)/sin^2(2*teta) for the asymmetric case.

        The result is cached per instance: the influence function depends only
        on the crystal configuration and on ``v`` -- NOT on the observation
        coordinates x, q -- so a scan (x-scan, q-scan, 2D map, diffraction
        profile) computes the expensive Kummer values only once and reuses them
        for every point. This speeds up scans by orders of magnitude compared
        to the previous per-point evaluation. The cache key hashes the ``v``
        array and the physical parameters, so mutating the instance parameters
        or changing the grid invalidates it naturally.

        Extra keyword arguments (the rest of a ``_calculate_constats_*`` dict)
        are ignored, so the dicts can be passed with ``**kwds`` directly.
        """
        v = numpy.asarray(v, dtype=float)
        key = (v.size, hash(v.tobytes()), complex(kap) if kap is not None else None,
               complex(chih2), float(acrist), float(gamma), float(a), float(alfa),
               float(k), float(teta))
        cache = getattr(self, '_influence_cache', None)
        if cache is None:
            cache = self._influence_cache = {}
        if key in cache:
            return cache[key]

        arg1 = numpy.clip(a ** 2 - v ** 2, 0.0, None)
        if alfa == 0:
            Z = k * numpy.sqrt(complex(chih2)) / numpy.sin(2 * teta)
            kum = BesselJ(0, Z * numpy.sqrt(arg1))
        else:
            # always the EXACT Kummer function (mpmath); this cached-grid path is what
            # use_fast_hyp1f1=2 selects, and exactness is part of its contract.
            yprime = acrist * gamma * arg1 / numpy.sin(2 * teta) ** 2
            kum = numpy.empty(v.size, dtype=complex)
            kk = complex(kap)
            for i in range(v.size):
                kum[i] = complex(mpmath.hyp1f1(1j * kk, 1, 1j * float(yprime[i])))

        cache[key] = kum
        return kum

    def _influence_values(self, v, a=None, alfa=None, k=None, teta=None, chih2=None,
                          acrist=None, gamma=None, kap=None, **kwargs):
        """
        Influence function on the array ``v``, dispatching on ``use_fast_hyp1f1``:

        * 0 -- exact Kummer, evaluated point by point with mpmath (the original,
          slow behaviour; no caching);
        * 1 -- approximated Kummer via :func:`fast_hyp1f1`, point by point (the
          original fast behaviour; no caching);
        * 2 -- exact Kummer computed ONCE on the grid and cached
          (:meth:`_influence_on_grid`); recommended: exact results with the
          Kummer cost paid a single time per scan/map.

        The symmetric case (alfa == 0) uses the exact (and cheap) Bessel form
        for every mode. Extra keyword arguments are ignored so the
        ``_calculate_constats_*`` dicts can be passed with ``**kwds``.
        """
        if alfa == 0 or self._use_fast_hyp1f1 == 2:
            return self._influence_on_grid(v, a=a, alfa=alfa, k=k, teta=teta, chih2=chih2,
                                           acrist=acrist, gamma=gamma, kap=kap)
        v = numpy.asarray(v, dtype=float)
        arg1 = numpy.clip(a ** 2 - v ** 2, 0.0, None)
        yprime = acrist * gamma * arg1 / numpy.sin(2 * teta) ** 2
        kum = numpy.empty(v.size, dtype=complex)
        kk = complex(kap)
        if self._use_fast_hyp1f1:  # mode 1: fast approximations
            for i in range(v.size):
                kum[i] = complex(fast_hyp1f1(kk, float(yprime[i])))
        else:                      # mode 0: exact, point by point
            for i in range(v.size):
                kum[i] = complex(mpmath.hyp1f1(1j * kk, 1, 1j * float(yprime[i])))
        return kum

    #
    # interface for q=0 or finite q
    #
    def xscan(self, q=1000.0, npoints_x=10, a_factor=1, a_center=0.0, filename=""):
        """
        Transverse intensity/amplitude profile at a fixed distance q, for a
        point source.

        Dispatches to the appropriate equation depending on the source distance
        p and observation distance q: GF2016 eq. 23 (p=0, q=0), eq. 24 (p=0,
        finite q), eq. 30 (finite p, q=0) or eq. 31 (finite p, finite q).

        Parameters
        ----------
        q : float
            Crystal-to-observation distance in mm (q = 0 is the exit surface).
        npoints_x : int
            Number of transverse sampling points.
        a_factor : float
            Half-width of the x window in units of a = t1*sin(2 theta_B)/2; the
            scan spans [-a*a_factor, a*a_factor].
        a_center : float
            Offset (mm) subtracted from the x grid (to centre on the pattern).
        filename : str
            If non-empty, save the resulting wavefront to this HDF5 file.

        Returns
        -------
        xx : ndarray
            Transverse coordinates in mm.
        yy_amplitude : ndarray of complex
            Complex Bragg-reflected amplitude at each x.
        output_wavefront : GenericWavefront1D
            WOFRY 1D wavefront wrapping (xx, yy_amplitude).
        """
        if self._p == 0:
            if q == 0:
                txt = "xscan_at_q0_and_p0() (Guigay & Ferrero 2016 eq 23 http://dx.doi.org/10.1107/S2053273316006549)"
            else:
                txt = "xscan_at_finite_q_and_p0() (Guigay & Ferrero 2016 eq 24 http://dx.doi.org/10.1107/S2053273316006549)"
        else:
            if q == 0:
                txt = "xscan_at_q0() (Guigay & Ferrero 2016 eq 30 http://dx.doi.org/10.1107/S2053273316006549)"
            else:
                txt = "xscan_at_finite_q() (Guigay & Ferrero 2016 eq 31 http://dx.doi.org/10.1107/S2053273316006549)"

        print("Calculating x-scan")
        print("    at p=%.3f mm, q=%.3f..." % (self._p, q))
        print("    using %s" % (txt))
        t0 = time.time()

        if self._p == 0:
            if q == 0:
                out = self.xscan_at_q0_and_p0(npoints_x=npoints_x, a_factor=a_factor, a_center=a_center, filename=filename)
            else:
                out = self.xscan_at_finite_q_and_p0(q, npoints_x=npoints_x, a_factor=a_factor, a_center=a_center, filename=filename)
        else:
            if q == 0:
                out = self.xscan_at_q0(npoints_x=npoints_x, a_factor=a_factor, a_center=a_center, filename=filename)
            else:
                out = self.xscan_at_finite_q(q, npoints_x=npoints_x, a_factor=a_factor, a_center=a_center, filename=filename)

        print("Calculation time: ", time.time() - t0)
        return out

    # x-scan at p=q=0 using Guigay % Ferrero 2016 eq 23
    def xscan_at_q0_and_p0(self, npoints_x=10, a_factor=1, a_center=0.0, filename=""):
        """
        Transverse profile on the exit surface for a point source on the
        entrance surface (p = 0, q = 0), using GF2016 eq. 23. Arguments and
        return values are as in :meth:`xscan`.
        """
        kwds = self._calculate_constats_for_equation23_2016()
        a = kwds['a']

        # x-scan at q=0
        print("a=%.3f mm..." % (a))

        xx = numpy.linspace(-a * a_factor, a * a_factor, npoints_x) - a_center

        # FEATURE (2026): _equation23_2016 is vectorized; evaluate the whole grid at once.
        yy_amplitude = numpy.asarray(self._equation23_2016(xx, **kwds), dtype=complex)

        # create and write wofry wavefront
        output_wavefront = GenericWavefront1D.initialize_wavefront_from_arrays(
            1e-3 * xx, yy_amplitude, y_array_pi=None, wavelength=1e-10)
        output_wavefront.set_photon_energy(1e3 * self._photon_energy_in_keV)
        if filename != "":
            output_wavefront.save_h5_file(filename,
                                          subgroupname="wfr", intensity=True, phase=False, overwrite=True,
                                          verbose=False)
            print("File %s written to disk" % filename)

        return xx, yy_amplitude, output_wavefront

    # x-scan at p=0, finite q, using Guigay % Ferrero 2016 eq 24
    def xscan_at_finite_q_and_p0(self, q=1000.0, npoints_x=10, a_factor=1, a_center=0.0, filename=""):
        """
        Transverse profile at a finite distance q for a point source on the
        entrance surface (p = 0), using GF2016 eq. 24. Arguments and return
        values are as in :meth:`xscan`.
        """
        kwds = self._calculate_constats_for_equation23_2016() #?????????????
        a = kwds['a']

        print("a=%.3f mm..." % (a))

        xx = numpy.linspace(-a * a_factor, a * a_factor, npoints_x) - a_center
        yy_amplitude = numpy.zeros_like(xx, dtype=complex)

        print(f"Progress: 0%")
        for j in range(xx.size):
            progress = (j + 1) / xx.size * 100
            if progress % 10 == 0:  print(f"Progress: {progress:.0f}%")
            amplitude = self._equation24_2016(xx[j], q, **kwds)
            yy_amplitude[j] = amplitude
        print(f"Progress: 100%")

        # create and write wofry wavefront
        output_wavefront = GenericWavefront1D.initialize_wavefront_from_arrays(
            1e-3 * xx, yy_amplitude, y_array_pi=None, wavelength=1e-10)
        output_wavefront.set_photon_energy(1e3 * self._photon_energy_in_keV)
        if filename != "":
            output_wavefront.save_h5_file(filename,
                                          subgroupname="wfr", intensity=True, phase=False, overwrite=True,
                                          verbose=False)
            print("File %s written to disk" % filename)

        return xx, yy_amplitude, output_wavefront


    # x-scan at q=0 using Guigay % Ferrero 2016 eq 30
    def xscan_at_q0(self, npoints_x=10, a_factor=1, a_center=0.0, filename=""):
        """
        Transverse profile on the exit surface (q = 0) for a point source at a
        finite distance p, using GF2016 eq. 30. Raises if p == 0 (use
        :meth:`xscan_at_q0_and_p0` instead). Arguments and return values are as
        in :meth:`xscan`.
        """
        if self._p == 0:
            raise Exception("For p=0 please use xscan_at_q0_and_p0()")

        kwds = self._calculate_constats_for_equation30_2016()
        a = kwds['a']

        # x-scan at q=0
        print("a=%.3f mm..." % (a))

        xx = numpy.linspace(-a * a_factor, a * a_factor, npoints_x) - a_center
        yy_amplitude = numpy.zeros_like(xx, dtype=complex)

        print(f"Progress: 0%")
        for j in range(xx.size):
            progress = (j + 1) / xx.size * 100
            if progress % 10 == 0:  print(f"Progress: {progress:.0f}%")
            amplitude = self._equation30_2016(xx[j], **kwds)
            yy_amplitude[j] = amplitude
        print(f"Progress: 100%")

        # create and write wofry wavefront
        output_wavefront = GenericWavefront1D.initialize_wavefront_from_arrays(
            1e-3 * xx, yy_amplitude, y_array_pi=None, wavelength=1e-10)
        output_wavefront.set_photon_energy(1e3 * self._photon_energy_in_keV)
        if filename != "":
            output_wavefront.save_h5_file(filename,
                                          subgroupname="wfr", intensity=True, phase=False, overwrite=True,
                                          verbose=False)
            print("File %s written to disk" % filename)

        return xx, yy_amplitude, output_wavefront

    # x-scan at finite q using Guigay % Ferrero 2016 eq 31
    def xscan_at_finite_q(self, q=1000.0, npoints_x=10, a_factor=1, a_center=0.0, filename=""):
        """
        Transverse profile at a finite distance q for a point source at a finite
        distance p, using GF2016 eq. 31. Raises if p == 0 (use
        :meth:`xscan_at_finite_q_and_p0` instead). Arguments and return values
        are as in :meth:`xscan`.
        """
        if self._p == 0:
            raise Exception("For p=0 please use xscan_at_finite_q_and_p0()")

        kwds = self._calculate_constats_for_equation31_2016()
        a = kwds['a']

        # x-scan at q=0
        print("a=%.3f mm..." % (a))

        xx = numpy.linspace(-a * a_factor, a * a_factor, npoints_x) - a_center
        yy_amplitude = numpy.zeros_like(xx, dtype=complex)

        print(f"Progress: 0%")
        for j in range(xx.size):
            progress = (j + 1) / xx.size * 100
            if progress % 10 == 0:  print(f"Progress: {progress:.0f}%")
            amplitude = self._equation31_2016(xx[j], q, **kwds)
            yy_amplitude[j] = amplitude
        print(f"Progress: 100%")

        # create and write wofry wavefront
        output_wavefront = GenericWavefront1D.initialize_wavefront_from_arrays(
            1e-3 * xx, yy_amplitude, y_array_pi=None, wavelength=1e-10)
        output_wavefront.set_photon_energy(1e3 * self._photon_energy_in_keV)
        if filename != "":
            output_wavefront.save_h5_file(filename,
                                          subgroupname="wfr", intensity=True, phase=False, overwrite=True,
                                          verbose=False)
            print("File %s written to disk" % filename)

        return xx, yy_amplitude, output_wavefront

####################################
    # x-scan at p=q=0 using Guigay % Ferrero 2016 eq 23
    def xscan_for_external_wavefront(self,
                                     Phi=None,
                                     Phi_tau=None, # in mm !!!!!!!!!!!!
                                     npoints_x=10,
                                     a_factor=1,
                                     a_center=0.0,
                                     filename=""):
        """
        Exit-surface field for an arbitrary incident field (rather than a point
        source), using GF2016 eq. 28: D_h(x) is the integral of the incident
        amplitude over the entrance coordinate tau weighted by the crystal
        propagator P(x, tau).

        Parameters
        ----------
        Phi : ndarray of complex, optional
            Incident complex amplitude on the entrance surface; defaults to a
            uniform (plane-wave) field.
        Phi_tau : ndarray, optional
            Entrance-surface coordinates tau (mm) for ``Phi``; interpolated
            internally. Defaults to the x grid.
        npoints_x, a_factor, a_center, filename :
            As in :meth:`xscan`.

        Returns
        -------
        xx, yy_amplitude, output_wavefront
            As in :meth:`xscan`.
        """
        ##################################
        # from srxraylib.plot.gol import plot
        # plot(Phi_tau, numpy.abs(Phi)**2)
        ##################################


        kwds = self._calculate_constats_for_equation31_2016()
        a = kwds['a']

        # x-scan at q=0
        print("xscan_for_external_wavefront() (Guigay & Ferrero 2016 eq 28 http://dx.doi.org/10.1107/S2053273316006549)")
        print("a=%.3f mm..." % (a))

        xx = numpy.linspace(-a * a_factor, a * a_factor, npoints_x) - a_center
        yy_amplitude = numpy.zeros_like(xx, dtype=complex)

        ##
        if Phi is None: Phi = numpy.ones_like(xx, dtype=complex)
        if Phi_tau is None: Phi_tau = xx
        ##


        # interpolator
        f_mag = interpolate.interp1d(Phi_tau, numpy.abs(Phi), kind='linear', bounds_error=False, fill_value=0)
        f_phase = interpolate.interp1d(Phi_tau, numpy.angle(Phi), kind='linear', bounds_error=False, fill_value=0)

        print(f"Progress: 0%")
        for j in range(xx.size):
            progress = (j + 1) / xx.size * 100
            if progress % 10 == 0:  print(f"Progress: {progress:.0f}%")
            # amplitude = self._equation28_2016(xx[j], Phi, Phi_tau, **kwds)
            amplitude = self._equation28_2016(xx[j], f_mag, f_phase, **kwds)
            yy_amplitude[j] = amplitude
        print(f"Progress: 100%")

        # create and write wofry wavefront
        output_wavefront = GenericWavefront1D.initialize_wavefront_from_arrays(
            1e-3 * xx, yy_amplitude, y_array_pi=None, wavelength=1e-10)
        output_wavefront.set_photon_energy(1e3 * self._photon_energy_in_keV)
        if filename != "":
            output_wavefront.save_h5_file(filename,
                                          subgroupname="wfr", intensity=True, phase=False, overwrite=True,
                                          verbose=False)
            print("File %s written to disk" % filename)

        return xx, yy_amplitude, output_wavefront


    ###################################
    def diffraction_profile_angle_scan(self, THETA):
        """
        Angular (far-field) diffraction profile of the Laue crystal.

        The complex diffraction profile is the Fourier transform (Fraunhofer /
        far-field limit) of the exit-surface field D_h(x):

            D_h(theta) = integral_{-a}^{+a} D_h(x, q=0) exp(i k theta x) dx,

        where theta is the observation angle around the diffracted-beam
        direction and a = t sin(2 theta_B) / 2 is the half-width of the
        Borrmann fan on the exit surface. The exit field D_h(x, q=0) is the
        point-source (p = q = 0) field of Guigay & Ferrero 2016 (GF2016)
        eq. 23 (Kummer form, asymmetric case) or, for alfa == 0, the Bessel
        form (GF2013 eq. 10 / Kato 1961). Equivalently this is the
        q -> infinity limit of the propagated field GF2016 eq. 24, in which
        (x - x_c) / q -> theta.

        Note that the profile is an intrinsic crystal property: it uses the
        p = 0 exit field and does not depend on the source distance p.
        |D_h(theta)|^2 is the reflectivity-versus-angle diffraction profile
        (``method 2'' in Sanchez del Rio & Guigay).

        Parameters
        ----------
        THETA : ndarray
            Observation angles (radians), measured around the diffracted-beam
            direction, at which to evaluate the profile.

        Returns
        -------
        ndarray of complex
            The complex diffraction profile D_h(theta); take the squared
            modulus for the intensity (reflectivity) profile.

        References
        ----------
        Guigay & Ferrero, Acta Cryst. (2016) A72, 489-499, eq. 23 and eq. 24,
        http://dx.doi.org/10.1107/S2053273316006549
        """

        print("diffraction_profile_angle_scan(): FT of the exit field "
              "(GF2016 eq. 23; far-field limit of eq. 24) "
              "http://dx.doi.org/10.1107/S2053273316006549")

        AMPL = numpy.zeros_like(THETA, dtype=complex)

        kwds = self._calculate_constats_for_equation31_2016()
        print("a=%.3f mm..." % (kwds['a']))
        print("k=%.3g mm..." % (kwds['k']))

        ##
        # Phi = output_wavefront.get_complex_amplitude()
        # Phi_x = output_wavefront.get_abscissas() * 1e3
        ##
        # # interpolator
        # f_mag = interpolate.interp1d(Phi_x, numpy.abs(Phi), kind='linear', bounds_error=False, fill_value=0)
        # f_phase = interpolate.interp1d(Phi_x, numpy.angle(Phi), kind='linear', bounds_error=False, fill_value=0)

        if 0: # not vectorized
            print(f"Progress: 0%")
            for i, inclination in enumerate(THETA):
                progress = (i + 1) / THETA.size
                if (progress * 100) % 10 <= (100 / THETA.size):
                    print(f"Progress: {100 * progress:.0f}%")
                amplitude = self._equationXX_2016(inclination, **kwds)
                AMPL[i] = amplitude
            print(f"Progress: 100%")
        else: # vectorized
            AMPL = self._equationXX_2016_vectorized(THETA, **kwds)

        return AMPL
####################################
    #
    # private methods
    #

    # Guigay&Ferrero 2016: calculate equation 23, p=q=0
    def _equation23_2016(self, x,
                         a=None,
                         mu1=None,
                         mu2=None,
                         teta=None,
                         teta1=None,
                         teta2=None,
                         alfa=None,
                         acrist=None,
                         gamma=None,
                         lambda1=None,
                         omega=None,
                         t1=None,
                         a2=None,
                         g=None,
                         kap=None,
                         k=None,
                         pe=None,
                         acmax=None,
                         kiny=None,
                         att=None,
                         chizero=None,
                         t2=None,
                         chih2=None,
                         ):
        """
        Evaluate GF2016 eq. 23: the exit-surface amplitude at transverse
        position ``x`` for a point source on the entrance surface (p = q = 0).
        ``x`` is the TRUE exit-surface coordinate: at q = 0 there is no
        propagation-induced lateral shift (x_c = q*(...) = 0), so no x_c is
        subtracted here. Uses the Bessel form if alfa == 0, otherwise the Kummer
        function. The many keyword arguments are the precomputed constants
        returned by :meth:`_calculate_constats_for_equation23_2016`. Returns 0
        for |x| > a.

        Returns
        -------
        complex
            The complex amplitude D_h(x).
        """
        # FEATURE (2026): vectorized -- x may be a scalar (old behaviour) or an array.
        # The influence function comes from the cached grid evaluator.
        x_in = numpy.asarray(x, dtype=float)
        scalar_input = (x_in.ndim == 0)
        xx = numpy.atleast_1d(x_in)

        kum = self._influence_values(xx, a=a, alfa=alfa, k=k, teta=teta, chih2=chih2,
                                      acrist=acrist, gamma=gamma, kap=kap)

        # FEATURE (2026): split the (1j*Re - Im)*chizero factor into phase (kept) and normal
        # absorption (gated). sqrt(att) is the amplitude form of att=exp(-k*0.5*(t1+t2)*Im chi0),
        # so |.|^2 reproduces att; with apply_absorption=False, att=1 leaves only the phase.
        amp = numpy.exp(1j * k * chizero.real * 0.25 * (t1 + t2)) * numpy.sqrt(att) * \
              kum * \
              numpy.exp(-1j * xx ** 2 * k * mu1 / 2 / self._R) * \
              numpy.exp(1j * xx * k * (omega.real - t1 * numpy.sin(teta1) / 2 / self._R)) * \
              numpy.exp(- xx * k * omega.imag)
        amp = numpy.where(numpy.abs(xx) > a, 0.0, amp)

        if scalar_input: return complex(amp[0])
        return amp



    def _xc_equation24(self, q, omega=None, t1=None, teta1=None, **kwargs):
        """
        Lateral shift x_c (mm) of the pattern centre for eq. 24 (point source on
        the entrance surface, p = 0), as a function of q [Guigay & Ferrero 2016,
        eq. 25]:  x_c = q * (Re(omega) - t1*sin(theta_1)/(2R)).  Extra keyword
        arguments (the rest of the constants dict) are ignored.
        """
        return q * (omega.real - t1 * numpy.sin(teta1) / (2 * self._R))

    # Guigay&Ferrero 2016: calculate equation 24, p=0, finite q
    # 2026: written as a function of the TRUE observation coordinate x (x_c subtracted internally).
    def _equation24_2016(self, x, q,
                         a=None,
                         mu1=None,
                         mu2=None,
                         teta=None,
                         teta1=None,
                         teta2=None,
                         alfa=None,
                         acrist=None,
                         gamma=None,
                         lambda1=None,
                         omega=None,
                         t1=None,
                         a2=None,
                         g=None,
                         kap=None,
                         k=None,
                         pe=None,
                         acmax=None,
                         kiny=None,
                         att=None,
                         chizero=None,
                         t2=None,
                         chih2=None,
                         ):
        """
        Evaluate GF2016 eq. 24: the amplitude at the TRUE observation coordinate
        ``x`` and distance ``q`` for a point source on the entrance surface
        (p = 0). The lateral shift x_c is computed internally (eq. 25, via
        :meth:`_xc_equation24`) and subtracted in the cosine, so the pattern is
        centred at x = x_c. The integral over the influence region is folded onto
        [0, a] with the absorption inside the cosine (valid because the integrand
        is even in v for any fixed x - x_c). Keyword arguments are the precomputed
        constants. Returns the complex amplitude D_h(x, q).
        """
        #if numpy.abs(x) > a: return 0
        v = numpy.linspace(0, a, self._integration_points)
        invle = 1 / q - mu1 / self._R
        # 2026: function of the TRUE x; subtract the lateral shift x_c (eq. 25).
        xc = self._xc_equation24(q, omega=omega, t1=t1, teta1=teta1)

        # FEATURE (2026): vectorized over v; the influence function dispatches on use_fast_hyp1f1 (exact cached grid for mode 2).
        kum = self._influence_values(v, a=a, alfa=alfa, k=k, teta=teta, chih2=chih2,
                                      acrist=acrist, gamma=gamma, kap=kap)
        Q1 = 1j * k * 0.5 * v ** 2 * invle
        # 2026: lateral argument uses (x - xc) so the field is expressed vs the true x.
        Q2 = k * v * ((x - xc) / q - 1j * kiny)
        y = kum * numpy.exp(Q1) * numpy.cos(Q2)

        return 2 * trapezoid(y, x=v) * numpy.sqrt(att / numpy.abs(lambda1 * q))

########################################
    # Guigay&Ferrero 2016: calculate integral in equation 28 for q=0 with a given wavefront amplitude defined at p=0
    # note that the integral limits are gamma (u+-a) and the integrand is Phi(tau) P(u,tau) with Phi() the complex amplitude
    def _equation28_2016(self, x,
                        f_mag, f_phase, # interpolator
                        a       = None,
                        mu1     = None,
                        mu2     = None,
                        teta    = None,
                        teta1   = None,
                        teta2   = None,
                        alfa    = None,
                        acrist  = None,
                        gamma   = None,
                        lambda1 = None,
                        omega   = None,
                        t1      = None,
                        a2      = None,
                        g       = None,
                        kap     = None,
                        k       = None,
                        pe      = None,
                        acmax   = None,
                        kiny    = None,
                        att     = None,
                        chizero = None,
                        t2      = None,
                        chih2   = None,
                      ):
        """
        Evaluate GF2016 eq. 28 for q = 0 with an arbitrary incident field: the
        exit amplitude at ``x`` is the integral over the entrance coordinate tau
        in [gamma(x-a), gamma(x+a)] of Phi(tau) * P(x, tau), with the incident
        field supplied as the interpolators ``f_mag`` and ``f_phase`` (modulus
        and phase versus tau). Remaining keyword arguments are the precomputed
        constants. Returns the complex amplitude D_h(x).
        """
        tau = numpy.linspace(gamma * (x - a), gamma * (x + a), self._integration_points)

        # FEATURE (2026): vectorized over tau. As tau spans [gamma*(x-a), gamma*(x+a)],
        # nu = x - tau/gamma spans exactly [a, -a] INDEPENDENTLY of x, so the influence
        # function is computed on the fixed grid linspace(a, -a, N) and cached: one
        # Kummer-grid evaluation serves every x of the scan.
        nu = numpy.linspace(a, -a, self._integration_points)
        kum = self._influence_values(nu, a=a, alfa=alfa, k=k, teta=teta, chih2=chih2,
                                      acrist=acrist, gamma=gamma, kap=kap)

        Q1 = 1j * k * nu * omega
        Q2 = -1j * k * (mu1 * x**2 + x * t1 * numpy.sin(teta1)) / (2 * self._R)
        Q3 = -1j * k * (mu2 * (nu - x)**2 - a2 * gamma * (nu - x)) / (2 * self._R)
        Q4 = 1j * k * (g / self._R) * (a + x) * (nu - x)
        A = f_mag(tau) * numpy.exp(1j * f_phase(tau))
        y = A * kum * numpy.exp(Q1 + Q2 + Q3 + Q4)

        amplitude = trapezoid(y, x=tau)
        return amplitude

    # for rocking curve...
    def _equationXX_2016(self, inclination,
                        # f_mag, f_phase, # interpolator
                        a       = None,
                        mu1     = None,
                        mu2     = None,
                        teta    = None,
                        teta1   = None,
                        teta2   = None,
                        alfa    = None,
                        acrist  = None,
                        gamma   = None,
                        lambda1 = None,
                        omega   = None,
                        t1      = None,
                        a2      = None,
                        g       = None,
                        kap     = None,
                        k       = None,
                        pe      = None,
                        acmax   = None,
                        kiny    = None,
                        att     = None,
                        chizero = None,
                        t2      = None,
                        chih2   = None,
                      ):
        """
        Far-field diffraction profile at a SINGLE observation angle.

        Evaluates the Fourier component, at the angle ``inclination``, of the
        exit-surface field over the Borrmann fan x in [-a, a]:

            D_h(inclination) = integral_{-a}^{+a} D_h(x, q=0)
                                   exp(i k x inclination) dx.

        The integrand D_h(x, q=0) is the GF2016 eq. 23 exit field (Bessel
        form for alfa == 0, Kummer form otherwise) --- identical to the
        integrand of :meth:`_equation23_2016`. The keyword arguments are the
        precomputed constants (from
        :meth:`_calculate_constats_for_equation31_2016`). This non-vectorized
        form is kept for reference; :meth:`_equationXX_2016_vectorized` is
        equivalent but faster (it builds the exit field once and reuses it for
        all angles).

        Returns
        -------
        complex
            The complex diffraction profile at the single angle
            ``inclination``.
        """

        X = numpy.linspace(-a, a, self._integration_points)
        amplitude = numpy.zeros_like(X, dtype=complex)

        for i, x in enumerate(X):

            arg1 = a ** 2 - x ** 2
            if arg1 < 0: arg1 = 0

            if alfa == 0:
                Z = k * numpy.sqrt(chih2) / numpy.sin(2 * teta)
                kum = BesselJ(0, Z * numpy.sqrt(arg1))
                Q1 = - 1j * k * x * (x + a) / (2 * self._R * numpy.cos(teta))
                Q2 = 0
            else:
                yprime = acrist * gamma *  arg1 / (numpy.sin(2 * teta))**2

                if self._use_fast_hyp1f1:
                    kum = fast_hyp1f1(kap, yprime)
                else:
                    kum = mpmath.hyp1f1(1j * kap, 1, 1j * yprime)
                Q1 = 1j * k * x * omega
                Q2 = -1j * k * (mu1 * x**2 + x * t1 * numpy.sin(teta1)) / (2 * self._R)

            A = numpy.exp(Q1 + Q2)
            amplitude[i] = A * kum * numpy.exp(1j * k * x * inclination)

        amplitude = trapezoid(amplitude, x=X)
        return amplitude

    def _equationXX_2016_vectorized(self, THETA,
                        # f_mag, f_phase, # interpolator
                        a       = None,
                        mu1     = None,
                        mu2     = None,
                        teta    = None,
                        teta1   = None,
                        teta2   = None,
                        alfa    = None,
                        acrist  = None,
                        gamma   = None,
                        lambda1 = None,
                        omega   = None,
                        t1      = None,
                        a2      = None,
                        g       = None,
                        kap     = None,
                        k       = None,
                        pe      = None,
                        acmax   = None,
                        kiny    = None,
                        att     = None,
                        chizero = None,
                        t2      = None,
                        chih2   = None,
                      ):
        """
        Far-field diffraction profile over an ARRAY of observation angles
        (vectorized version of :meth:`_equationXX_2016`).

        The exit-surface field D_h(x, q=0) [GF2016 eq. 23; Bessel form for
        alfa == 0, Kummer form otherwise] is built once on the grid
        x in [-a, a], and its Fourier transform is then evaluated at every
        angle in ``THETA``:

            D_h(theta) = integral_{-a}^{+a} D_h(x, q=0) exp(i k x theta) dx.

        This is the far-field (q -> infinity) limit of the propagated field
        GF2016 eq. 24. The keyword arguments are the precomputed constants
        (from :meth:`_calculate_constats_for_equation31_2016`).

        Returns
        -------
        ndarray of complex
            The complex diffraction profile at each angle in ``THETA``.
        """

        X = numpy.linspace(-a, a, self._integration_points)

        # FEATURE (2026): vectorized over X; the influence function dispatches on use_fast_hyp1f1 (exact cached grid for mode 2).
        kum = self._influence_values(X, a=a, alfa=alfa, k=k, teta=teta, chih2=chih2,
                                      acrist=acrist, gamma=gamma, kap=kap)
        if alfa == 0:
            Q1 = - 1j * k * X * (X + a) / (2 * self._R * numpy.cos(teta))
            Q2 = numpy.zeros_like(X)
        else:
            Q1 = 1j * k * X * omega
            Q2 = -1j * k * (mu1 * X**2 + X * t1 * numpy.sin(teta1)) / (2 * self._R)

        amplitude = numpy.exp(Q1 + Q2) * kum

        AMPLITUDE_INTEGRATED = numpy.zeros_like(THETA, dtype=complex)
        for i, inclination in enumerate(THETA):
            AMPLITUDE_INTEGRATED[i] = trapezoid(amplitude * numpy.exp(1j * k * X * inclination), x=X)
        return AMPLITUDE_INTEGRATED
        # amplitude_integrated = numpy.trapz(amplitude, x=X)
        # return amplitude_integrated
########################################
    def _xc_equation30(self, pe=None, omega=None, g=None, a=None, gamma=None, a2=None, **kwargs):
        """
        Symmetry centre x_c (mm) of the q = 0 pattern for a source at finite p
        (eq. 30). It is the q -> 0 limit of the eq. 31 / eq. 32 lateral shift:

            x_c = pe * (Re(omega) + m),   m = (g/R)*a + gamma*a2/(2R)  (s = 0),

        which keeps the q-scan central-intensity curve continuous at q = 0.
        Requires the eq.-31 constants (in particular ``pe``); extra keyword
        arguments (the rest of the constants dict) are ignored.
        """
        m = g * a / self._R + gamma * (a2 / (2 * self._R))
        return pe * (omega.real + m)

    # Guigay&Ferrero 2016: calculate integral with limits -a,a in equation 30, finite p, q=0
    # 2026: x is the TRUE exit coordinate. This is the full complex integral over [-a, a]
    #       (not the folded/cosine form), so no x_c is subtracted; the pattern sits at its
    #       physical centre (generally != 0 for finite p; see _xc_equation30 for that centre).
    def _equation30_2016(self, x,
                        a       = None,
                        mu1     = None,
                        mu2     = None,
                        teta    = None,
                        teta1   = None,
                        teta2   = None,
                        alfa    = None,
                        acrist  = None,
                        gamma   = None,
                        lambda1 = None,
                        omega   = None,
                        t1      = None,
                        a2      = None,
                        g       = None,
                        kap     = None,
                        k       = None,
                        att     = None,
                        chih2   = None,
                      ):
        """
        Evaluate GF2016 eq. 30: the exit-surface amplitude (q = 0) at the TRUE
        exit coordinate ``x`` for a point source at a finite distance p. This is
        the full complex integral over v in [-a, a] of the influence function
        times the source/curvature phases (not the folded/cosine form), so no
        x_c is subtracted: the pattern sits at its physical centre, which for
        finite p is generally != 0 (see :meth:`_xc_equation30`). Keyword
        arguments are the precomputed constants. Returns the complex amplitude
        D_h(x).
        """
        v = numpy.linspace(-a, a, self._integration_points)

        # FEATURE (2026): vectorized over v; the influence function dispatches on use_fast_hyp1f1 (exact cached grid for mode 2).
        kum = self._influence_values(v, a=a, alfa=alfa, k=k, teta=teta, chih2=chih2,
                                      acrist=acrist, gamma=gamma, kap=kap)

        mfac = gamma / numpy.sqrt(lambda1 * self._p)
        pe = 1 / (1 / self._p - mu2 / self._R)
        Q1 = gamma**2 * (x - v)**2 / (2 * pe)  # quadratic
        Q2 = -(mu1 * x**2) / (2 * self._R)
        Q3 = -(x * t1 * numpy.sin(teta1) - a2 * gamma * (v - x)) / (2 * self._R)
        Q4 = v * omega + g * (a + x) * (v - x) / self._R

        y = mfac * kum * numpy.exp(1j * k * (Q1 + Q2 + Q3 + Q4))

        # FIX (bug #5, 2026): eq30 was missing the normal-absorption factor (it carries no chi0/att
        # term). Multiply by sqrt(att) [amplitude form of att = exp(-k*(t1+t2)/2*Im chi0)] so |.|^2
        # carries the absorption, consistent with eq23/24/31. att is gated by apply_absorption.
        return trapezoid(y, x=v) * numpy.sqrt(att)

    def _xc_equation31(self, q, mu1=None, g=None, pe=None, omega=None, t1=None, teta1=None,
                       gamma=None, a=None, a2=None, **kwargs):
        """
        Lateral shift x_c (mm) of the pattern centre for eq. 31, as a function of
        the observation distance q [Guigay & Ferrero 2016, eq. 32]:

            x_c = q * (Be*pe*Re(omega) + m*pe/qe - t1*sin(theta_1)/(2R)),

        with Be = 1/pe + 1/qe and m = (g/R)*a + gamma*(s/p + a2/(2R)) (s = 0).
        Extra keyword arguments (the remaining entries of the constants dict) are
        ignored. Used by :meth:`_equation31_2016` (to express the field versus the
        true x) and by :meth:`qscan` (to evaluate the central intensity at x_c).
        """
        qe = q * self._R / (self._R - q * mu1 - g * q)
        be = 1 / qe + 1 / pe
        s = 0
        m = g * a / self._R + gamma * (s / self._p + a2 / 2 / self._R)
        return q * (be * pe * omega.real + m * pe / qe - t1 * numpy.sin(teta1) / (2 * self._R))

    # Guigay&Ferrero 2016: calculate integral in equation 31 and add the corresponding phases, finite p, finite q
    # 2026: written as a function of the TRUE observation coordinate x; the lateral shift x_c
    #       (eq. 32) is computed internally and subtracted, so the pattern is centred at x = x_c.
    def _equation31_2016(self, x, q,
                        a       = None,
                        mu1     = None,
                        mu2     = None,
                        teta    = None,
                        teta1   = None,
                        teta2   = None,
                        alfa    = None,
                        acrist  = None,
                        gamma   = None,
                        lambda1 = None,
                        omega   = None,
                        t1      = None,
                        a2      = None,
                        g       = None,
                        kap     = None,
                        k       = None,
                        pe      = None,
                        acmax   = None,
                        kiny    = None,
                        att     = None,
                        chizero = None,
                        t2      = None,
                        chih2   = None,
                      ):
        """
        Evaluate GF2016 eq. 31: the amplitude at the TRUE observation coordinate
        ``x`` and distance ``q`` for a point source at a finite distance p (the
        general case). The lateral shift x_c is computed internally (eq. 32, via
        :meth:`_xc_equation31`) and subtracted in the cosine, so the pattern is
        centred at x = x_c (peak not at zero, as in GF2013 Fig. 4b/5b). The
        integral is folded onto [0, a] with the absorption inside the cosine
        (valid because the integrand is even in v for any fixed x - x_c); the
        global phases P_g, which use the true x, are restored afterwards.
        Keyword arguments are the precomputed constants. Returns the complex
        amplitude D_h(x, q).
        """
        # FIX (bug #1, 2026): fold the integral onto [0, a] (with the factor 2 added at the
        # trapz below) and put the absorption term -1j*kiny INSIDE the cosine argument.
        # The previous version integrated over [-a, a] with a SEPARATE exp(-k*v*kiny) factor and
        # a real-argument cosine. Because exp(-k*v*kiny) is odd in v, that is only correct when
        # kiny (= omega_im) = 0, i.e. the symmetric case (alfa == 0); it gave wrong results for
        # asymmetric crystals (alfa != 0). Correct folded form: see Guigay & Ferrero 2016 eq. 31
        # and _equation24_2016 (which was already written correctly this way).
        v = numpy.linspace(0, a, self._integration_points)
        qe = q * self._R / (self._R - q * mu1 - g * q)
        be = 1 / qe + 1 / pe
        invle = 1 / (pe + qe) + g / self._R
        s = 0
        # FIX (bug #2, 2026): geometric a2 (variable a2), NOT a**2 [GF2016, just after eq. 30].
        # m enters only a phase, so it does not change |amplitude|^2, but it corrects the phase of
        # the propagated complex wavefront.
        m = g * a / self._R + gamma * (s / self._p + a2 / 2 / self._R)
        # 2026: this routine is a function of the TRUE x; subtract the lateral shift x_c (eq. 32)
        # in the cosine argument. The cosine / [0, a] folding stays valid because the integrand is
        # even in v for any fixed (x - x_c).
        xc = self._xc_equation31(q, mu1=mu1, g=g, pe=pe, omega=omega, t1=t1, teta1=teta1,
                                 gamma=gamma, a=a, a2=a2)

        # FEATURE (2026): vectorized over v; for use_fast_hyp1f1=2 the influence function is cached per grid, so a
        # q-scan or a 2D map evaluates the Kummer function only once for all its points.
        kum = self._influence_values(v, a=a, alfa=alfa, k=k, teta=teta, chih2=chih2,
                                      acrist=acrist, gamma=gamma, kap=kap)
        Q1 = 1j * k * 0.5 * v ** 2 * invle
        # FIX (bug #1, 2026): absorption -1j*kiny INSIDE the cosine (folded [0, a] form).
        # 2026: the lateral argument uses (x - xc) so the field is expressed vs the true x.
        Q3 = k * v * ((x - xc) / (q * pe * be) - 1j * kiny)
        y = kum * numpy.exp(Q1) * numpy.cos(Q3)

        # FIX (bug #1, 2026): factor 2 because the integral is folded onto [0, a].
        amplitude = 2 * trapezoid(y, x=v)

        # FIX (bug #5, 2026): restore the gamma factor of GF2016 eq. 31 / paper eq. 25
        # [prefactor = gamma * sqrt(att/(lambda*q*p*Be)), with Be = 1/be]. It was missing, so the
        # asymmetric (gamma != 1) magnitude was off by gamma; symmetric (gamma = 1) was unaffected.
        amplitude *= gamma * numpy.sqrt(att / (lambda1 * q * self._p * be))
        # omitted phase (see just after equation 30); P_g uses the true x.
        amplitude *= numpy.exp(1j * k * x ** 2 / 2 / q) * \
                     numpy.exp(1j * k * s ** 2 / 2 / self._p) * \
                     numpy.exp(1j * k * chizero.real * (t1 + t2) / 4)
        # omitted phase (see just before equation 31)
        amplitude *= numpy.exp(- 1j * (k / 2 / be) * \
                               (x / q + t1 * numpy.sin(teta1) / 2 / self._R + m) ** 2)
        return amplitude

    #
    # pack constants
    #

    def _calculate_constats_for_equation23_2016(self):
        """
        Compute and pack the constants needed by :meth:`_equation23_2016` (and
        also reused for eq. 24): geometry (a, t1, t2, gamma, theta_1/2, mu1/2,
        a2), the strain-gradient parameter A, the Kummer parameter kap = beta,
        the propagator parameters omega and g, the absorption factor att, the
        susceptibilities and wavelength.

        Returns
        -------
        dict
            Keyword arguments consumed by the ``_equation*_2016`` evaluators.
        """
        photon_energy_in_keV = self._photon_energy_in_keV
        p = self._p
        alfa = self._alfa_deg * numpy.pi / 180
        R = self._R
        poisson_ratio = self._poisson_ratio
        thickness = self._thickness

        teta, chizero, chih, chimh = self.get_crystal_data()

        lambda1 = codata.h * codata.c / codata.e / (photon_energy_in_keV * 1e3) * 1e3  # in mm
        # CONVENTION (2026): chih2 = chi_h * chi_hbar directly, with chih = chi_h and chimh = chi_{-h}
        # the (un-conjugated) Fourier coefficients from get_crystal_data() -- chi_{-h} is the real (-h)
        # structure factor, NOT a "-1j*chi_h" shortcut. For Si this product has Im(chi_h*chi_hbar) > 0,
        # equal to GF2013's caption value (+3.26e-12) and to paper2013.py, with NO conjugate fix.
        # (Only chi0 is returned conjugated, so the normal-absorption factor att decays; that is
        # independent of this product. The sign of Im(chi_h*chi_hbar) sets the Borrmann branch / which
        # +/-q focus is enhanced -- visible only at strong absorption, e.g. GF2013 Fig. 6 at 8.3 keV;
        # GSdR2022 Table 1 uses the opposite sign convention.)
        chih2 = chih * chimh
        if self._chih2 is not None: chih2 = self._chih2  # FEATURE (2026): user-supplied override

        if self._verbose:
            print("photon_energy_in_keV:", photon_energy_in_keV)
            print("lambda1 in mm:", lambda1)
            print("lambda1 in m, A:", lambda1 * 1e-3, lambda1 * 1e-3 * 1e10)
            print("CrystalSi 111")
            print("teta_deg:", teta * 180 / numpy.pi)
            print("p:", p)
            print("R:", R)
            print("chizero:", chizero)
            print("chih:", chih)
            print("chimh:", chimh)
            print("chih*chihbar:", chih2)



        k = 2 * numpy.pi / lambda1
        h = 2 * k * numpy.sin(teta)


        u2 = 0.25 * chih2 * k ** 2
        raygam = R * numpy.cos(teta)
        kp = k * numpy.sin(2 * teta)
        kp2 = kp * numpy.sin(2 * teta)

        #
        # TODO: Not working for alfa_deg=0
        #


        teta1 = alfa + teta
        teta2 = alfa - teta
        SG = None
        fam1 = numpy.sin(teta1)
        fam2 = numpy.sin(teta2)
        gam1 = numpy.cos(teta1)
        gam2 = numpy.cos(teta2)


        t1 = thickness / gam1
        t2 = thickness / gam2
        qpoly = p * R * gam2 / (2 * p + R * gam1)
        att = numpy.exp(-k * 0.5 * (t1 + t2) * numpy.imag(chizero))
        # FEATURE (2026): disable the normal-absorption factor to reproduce paper2013.py (attsym=1).
        if not self._apply_absorption: att = 1.0
        s2max = 0.25 * t1 * t2
        u2max = u2 * s2max  # Omega = k**2 chi_h chi_hbar / 4 ? (end of pag 490)
        gamma = t2 / t1
        a = numpy.sin(2 * teta) * t1 * 0.5
        kin = 0.25 * (t1 - t2) * chizero / a
        kinx = numpy.real(kin)
        kiny = numpy.imag(kin)
        com = numpy.sin(alfa) * (1 + gam1 * gam2 * (1 + poisson_ratio))
        kp3 = 0.5 * k * (gamma * a) ** 2
        mu1 = (numpy.cos(alfa) * 2 * fam1 * gam1 + numpy.sin(alfa) * (fam1 ** 2 + poisson_ratio * gam1 ** 2)) / (
                    numpy.sin(2 * teta) * numpy.cos(teta))
        mu2 = (numpy.cos(alfa) * 2 * fam2 * gam2 + numpy.sin(alfa) * (fam2 ** 2 + poisson_ratio * gam2 ** 2)) / (
                    numpy.sin(2 * teta) * numpy.cos(teta))
        a1 = (thickness / numpy.cos(teta)) * (
                    numpy.cos(alfa) * numpy.sin(teta1) + poisson_ratio * numpy.sin(alfa) * numpy.cos(teta1))
        a2 = (thickness / numpy.cos(teta)) * (
                    numpy.cos(alfa) * numpy.sin(teta2) + poisson_ratio * numpy.sin(alfa) * numpy.cos(teta2))
        acrist = -h * com / R  # A in Eq 17


        acmax = acrist * s2max
        g = gamma * acrist * R / kp2
        # FIX (bug #3, 2026): guard the division when acmax == 0 (alfa == 0 => A == 0). kap (= beta
        # = Omega / A) is unused in the symmetric (alfa == 0) Bessel branch, so 0.0 is a safe
        # placeholder; this only avoids a 0/0 nan and its RuntimeWarning.
        kap = (u2max / acmax) if acmax != 0 else 0.0  # beta = Omega / A

        pe = p * R / (gamma ** 2 * (R - p * mu2) - g * p)

        # WARNING DIFFERNT FROM Fig 2 (+p)

        # pe = p * R / (gamma**2 * (R + p * mu2) - g * p)
        if self._verbose:
            print("alfa:", alfa)
            print("teta1, teta2, teta:", teta1, teta2, teta)
            print("t1, t2, t/cos(teta):", t1, t2, thickness / numpy.cos(teta))
            print("a1, a2:", a1, a2, +thickness * numpy.tan(teta), -thickness * numpy.tan(teta))
            print("mu1, mu2:", mu1, mu2, +1 / numpy.cos(teta), -1 / numpy.cos(teta))
            print("acrist, com:", acrist, 0)
            print("pe:", pe)
            print("a: ", a, thickness * numpy.sin(teta))
            print("pe:", pe)

        omega = 0.25 * (t1 - t2) * chizero / a  # omega following the definition found after eq 22
        omega_real = numpy.real(omega)
        omega_imag = numpy.imag(omega)
        xc_over_q = omega_real - t1 * numpy.sin(alfa + teta) / (2 * R)

        return {
            "a"       : a,
            "mu1"     : mu1,
            "mu2"     : mu2,
            "teta"    : teta,
            "teta1"   : teta1,
            "teta2"   : teta2,
            "alfa"    : alfa,
            "acrist"  : acrist,
            "gamma"   : gamma,
            "lambda1" : lambda1,
            "omega"   : omega,
            "t1"      : t1,
            "a2"      : a2,
            "g"       : g,
            "kap"     : kap,
            "k"       : k,
            "pe"      : pe,
            "acmax"   : acmax,
            "kiny"    : kiny,
            "att"     : att,
            "chizero" : chizero,
            "t2"      : t2,
            "chih2"   : chih2,
            }


    def _calculate_constats_for_equation30_2016(self):
        """
        Compute and pack the constants needed by :meth:`_equation30_2016` (q = 0,
        finite p). Same quantities as
        :meth:`_calculate_constats_for_equation23_2016` minus the few entries
        eq. 30 does not use.

        Returns
        -------
        dict
            Keyword arguments consumed by :meth:`_equation30_2016`.
        """
        photon_energy_in_keV = self._photon_energy_in_keV
        p = self._p
        alfa = self._alfa_deg * numpy.pi / 180
        R = self._R
        poisson_ratio = self._poisson_ratio
        thickness = self._thickness

        teta, chizero, chih, chimh = self.get_crystal_data()

        lambda1 = codata.h * codata.c / codata.e / (photon_energy_in_keV * 1e3) * 1e3  # in mm
        # CONVENTION (2026): chih2 = chi_h * chi_hbar directly, with chih = chi_h and chimh = chi_{-h}
        # the (un-conjugated) Fourier coefficients from get_crystal_data() -- chi_{-h} is the real (-h)
        # structure factor, NOT a "-1j*chi_h" shortcut. For Si this product has Im(chi_h*chi_hbar) > 0,
        # equal to GF2013's caption value (+3.26e-12) and to paper2013.py, with NO conjugate fix.
        # (Only chi0 is returned conjugated, so the normal-absorption factor att decays; that is
        # independent of this product. The sign of Im(chi_h*chi_hbar) sets the Borrmann branch / which
        # +/-q focus is enhanced -- visible only at strong absorption, e.g. GF2013 Fig. 6 at 8.3 keV;
        # GSdR2022 Table 1 uses the opposite sign convention.)
        chih2 = chih * chimh
        if self._chih2 is not None: chih2 = self._chih2  # FEATURE (2026): user-supplied override

        if self._verbose:
            print("photon_energy_in_keV:", photon_energy_in_keV)
            print("lambda1 in mm:", lambda1)
            print("lambda1 in m, A:", lambda1 * 1e-3, lambda1 * 1e-3 * 1e10)
            print("CrystalSi 111")
            print("teta_deg:", teta * 180 / numpy.pi)
            print("p:", p)
            print("R:", R)
            print("chizero:", chizero)
            print("chih:", chih)
            print("chimh:", chimh)
            print("chih*chihbar:", chih2)



        k = 2 * numpy.pi / lambda1
        h = 2 * k * numpy.sin(teta)


        u2 = 0.25 * chih2 * k ** 2
        raygam = R * numpy.cos(teta)
        kp = k * numpy.sin(2 * teta)
        kp2 = kp * numpy.sin(2 * teta)

        #
        # TODO: Not working for alfa_deg=0
        #


        teta1 = alfa + teta
        teta2 = alfa - teta
        SG = None
        fam1 = numpy.sin(teta1)
        fam2 = numpy.sin(teta2)
        gam1 = numpy.cos(teta1)
        gam2 = numpy.cos(teta2)


        t1 = thickness / gam1
        t2 = thickness / gam2
        qpoly = p * R * gam2 / (2 * p + R * gam1)
        att = numpy.exp(-k * 0.5 * (t1 + t2) * numpy.imag(chizero))
        # FEATURE (2026): disable the normal-absorption factor to reproduce paper2013.py (attsym=1).
        if not self._apply_absorption: att = 1.0
        s2max = 0.25 * t1 * t2
        u2max = u2 * s2max  # Omega = k**2 chi_h chi_hbar / 4 ? (end of pag 490)
        gamma = t2 / t1
        a = numpy.sin(2 * teta) * t1 * 0.5
        kin = 0.25 * (t1 - t2) * chizero / a
        kinx = numpy.real(kin)
        kiny = numpy.imag(kin)
        com = numpy.sin(alfa) * (1 + gam1 * gam2 * (1 + poisson_ratio))
        kp3 = 0.5 * k * (gamma * a) ** 2
        mu1 = (numpy.cos(alfa) * 2 * fam1 * gam1 + numpy.sin(alfa) * (fam1 ** 2 + poisson_ratio * gam1 ** 2)) / (
                    numpy.sin(2 * teta) * numpy.cos(teta))
        mu2 = (numpy.cos(alfa) * 2 * fam2 * gam2 + numpy.sin(alfa) * (fam2 ** 2 + poisson_ratio * gam2 ** 2)) / (
                    numpy.sin(2 * teta) * numpy.cos(teta))
        a1 = (thickness / numpy.cos(teta)) * (
                    numpy.cos(alfa) * numpy.sin(teta1) + poisson_ratio * numpy.sin(alfa) * numpy.cos(teta1))
        a2 = (thickness / numpy.cos(teta)) * (
                    numpy.cos(alfa) * numpy.sin(teta2) + poisson_ratio * numpy.sin(alfa) * numpy.cos(teta2))
        acrist = -h * com / R  # A in Eq 17


        acmax = acrist * s2max
        g = gamma * acrist * R / kp2
        # FIX (bug #3, 2026): guard the division when acmax == 0 (alfa == 0 => A == 0). kap (= beta
        # = Omega / A) is unused in the symmetric (alfa == 0) Bessel branch, so 0.0 is a safe
        # placeholder; this only avoids a 0/0 nan and its RuntimeWarning.
        kap = (u2max / acmax) if acmax != 0 else 0.0  # beta = Omega / A

        pe = p * R / (gamma ** 2 * (R - p * mu2) - g * p)

        # WARNING DIFFERNT FROM Fig 2 (+p)

        # pe = p * R / (gamma**2 * (R + p * mu2) - g * p)
        if self._verbose:
            print("alfa:", alfa)
            print("teta1, teta2, teta:", teta1, teta2, teta)
            print("t1, t2, t/cos(teta):", t1, t2, thickness / numpy.cos(teta))
            print("a1, a2:", a1, a2, +thickness * numpy.tan(teta), -thickness * numpy.tan(teta))
            print("mu1, mu2:", mu1, mu2, +1 / numpy.cos(teta), -1 / numpy.cos(teta))
            print("acrist, com:", acrist, 0)
            print("pe:", pe)
            print("a: ", a, thickness * numpy.sin(teta))

        omega = 0.25 * (t1 - t2) * chizero / a  # omega following the definition found after eq 22
        omega_real = numpy.real(omega)
        omega_imag = numpy.imag(omega)
        xc_over_q = omega_real - t1 * numpy.sin(alfa + teta) / (2 * R)

        return {
            "a"       : a,
            "mu1"     : mu1,
            "mu2"     : mu2,
            "teta"    : teta,
            "teta1"   : teta1,
            "teta2"   : teta2,
            "alfa"    : alfa,
            "acrist"  : acrist,
            "gamma"   : gamma,
            "lambda1" : lambda1,
            "omega"   : omega,
            "t1"      : t1,
            "a2"      : a2,
            "g"       : g,
            "kap"     : kap,
            "k"       : k,
            "att"     : att,          # FIX (bug #5, 2026): now returned so eq30 can apply sqrt(att)
            "chih2"   : chih2,
            }


    def _calculate_constats_for_equation31_2016(self):
        """
        Compute and pack the constants needed by :meth:`_equation31_2016` (finite
        p, finite q; the general case used by :meth:`qscan`). Includes the
        effective distances pe (qe and Be are derived inside the evaluator from
        the running q), omega, g, kap, att and the susceptibilities.

        Returns
        -------
        dict
            Keyword arguments consumed by :meth:`_equation31_2016`.
        """
        photon_energy_in_keV = self._photon_energy_in_keV
        p = self._p
        alfa = self._alfa_deg * numpy.pi / 180
        R = self._R
        poisson_ratio = self._poisson_ratio
        thickness = self._thickness

        teta, chizero, chih, chimh = self.get_crystal_data()

        lambda1 = codata.h * codata.c / codata.e / (photon_energy_in_keV * 1e3) * 1e3  # in mm
        # CONVENTION (2026): chih2 = chi_h * chi_hbar directly, with chih = chi_h and chimh = chi_{-h}
        # the (un-conjugated) Fourier coefficients from get_crystal_data() -- chi_{-h} is the real (-h)
        # structure factor, NOT a "-1j*chi_h" shortcut. For Si this product has Im(chi_h*chi_hbar) > 0,
        # equal to GF2013's caption value (+3.26e-12) and to paper2013.py, with NO conjugate fix.
        # (Only chi0 is returned conjugated, so the normal-absorption factor att decays; that is
        # independent of this product. The sign of Im(chi_h*chi_hbar) sets the Borrmann branch / which
        # +/-q focus is enhanced -- visible only at strong absorption, e.g. GF2013 Fig. 6 at 8.3 keV;
        # GSdR2022 Table 1 uses the opposite sign convention.)
        chih2 = chih * chimh
        if self._chih2 is not None: chih2 = self._chih2  # FEATURE (2026): user-supplied override

        if self._verbose:
            print("photon_energy_in_keV:", photon_energy_in_keV)
            print("lambda1 in mm:", lambda1)
            print("lambda1 in m, A:", lambda1 * 1e-3, lambda1 * 1e-3 * 1e10)
            print("CrystalSi 111")
            print("teta_deg:", teta * 180 / numpy.pi)
            print("p:", p)
            print("R:", R)
            print("chizero:", chizero)
            print("chih:", chih)
            print("chimh:", chimh)
            print("chih*chihbar:", chih2)



        k = 2 * numpy.pi / lambda1
        h = 2 * k * numpy.sin(teta)


        u2 = 0.25 * chih2 * k ** 2
        raygam = R * numpy.cos(teta)
        kp = k * numpy.sin(2 * teta)
        kp2 = kp * numpy.sin(2 * teta)

        #
        # TODO: Not working for alfa_deg=0
        #


        teta1 = alfa + teta
        teta2 = alfa - teta
        SG = None
        fam1 = numpy.sin(teta1)
        fam2 = numpy.sin(teta2)
        gam1 = numpy.cos(teta1)
        gam2 = numpy.cos(teta2)


        t1 = thickness / gam1
        t2 = thickness / gam2
        qpoly = p * R * gam2 / (2 * p + R * gam1)
        att = numpy.exp(-k * 0.5 * (t1 + t2) * numpy.imag(chizero))
        # FEATURE (2026): disable the normal-absorption factor to reproduce paper2013.py (attsym=1).
        if not self._apply_absorption: att = 1.0
        s2max = 0.25 * t1 * t2
        u2max = u2 * s2max  # Omega = k**2 chi_h chi_hbar / 4 ? (end of pag 490)
        gamma = t2 / t1
        a = numpy.sin(2 * teta) * t1 * 0.5
        kin = 0.25 * (t1 - t2) * chizero / a
        kinx = numpy.real(kin)
        kiny = numpy.imag(kin)
        com = numpy.sin(alfa) * (1 + gam1 * gam2 * (1 + poisson_ratio))
        kp3 = 0.5 * k * (gamma * a) ** 2
        mu1 = (numpy.cos(alfa) * 2 * fam1 * gam1 + numpy.sin(alfa) * (fam1 ** 2 + poisson_ratio * gam1 ** 2)) / (
                    numpy.sin(2 * teta) * numpy.cos(teta))
        mu2 = (numpy.cos(alfa) * 2 * fam2 * gam2 + numpy.sin(alfa) * (fam2 ** 2 + poisson_ratio * gam2 ** 2)) / (
                    numpy.sin(2 * teta) * numpy.cos(teta))
        a1 = (thickness / numpy.cos(teta)) * (
                    numpy.cos(alfa) * numpy.sin(teta1) + poisson_ratio * numpy.sin(alfa) * numpy.cos(teta1))
        a2 = (thickness / numpy.cos(teta)) * (
                    numpy.cos(alfa) * numpy.sin(teta2) + poisson_ratio * numpy.sin(alfa) * numpy.cos(teta2))
        acrist = -h * com / R  # A in Eq 17


        acmax = acrist * s2max
        g = gamma * acrist * R / kp2
        # FIX (bug #3, 2026): guard the division when acmax == 0 (alfa == 0 => A == 0). kap (= beta
        # = Omega / A) is unused in the symmetric (alfa == 0) Bessel branch, so 0.0 is a safe
        # placeholder; this only avoids a 0/0 nan and its RuntimeWarning.
        kap = (u2max / acmax) if acmax != 0 else 0.0  # beta = Omega / A

        pe = p * R / (gamma ** 2 * (R - p * mu2) - g * p)

        # WARNING DIFFERNT FROM Fig 2 (+p)

        # pe = p * R / (gamma**2 * (R + p * mu2) - g * p)
        if self._verbose:
            print("alfa:", alfa)
            print("teta1, teta2, teta:", teta1, teta2, teta)
            print("t1, t2, t/cos(teta):", t1, t2, thickness / numpy.cos(teta))
            print("a1, a2:", a1, a2, +thickness * numpy.tan(teta), -thickness * numpy.tan(teta))
            print("mu1, mu2:", mu1, mu2, +1 / numpy.cos(teta), -1 / numpy.cos(teta))
            print("acrist, com:", acrist, 0)
            print("pe:", pe)
            print("a: ", a, thickness * numpy.sin(teta))
            print("pe:", pe)

        omega = 0.25 * (t1 - t2) * chizero / a  # omega following the definition found after eq 22
        omega_real = numpy.real(omega)
        omega_imag = numpy.imag(omega)
        xc_over_q = omega_real - t1 * numpy.sin(alfa + teta) / (2 * R)

        return {
            "a"       : a,
            "mu1"     : mu1,
            "mu2"     : mu2,
            "teta"    : teta,
            "teta1"   : teta1,
            "teta2"   : teta2,
            "alfa"    : alfa,
            "acrist"  : acrist,
            "gamma"   : gamma,
            "lambda1" : lambda1,
            "omega"   : omega,
            "t1"      : t1,
            "a2"      : a2,
            "g"       : g,
            "kap"     : kap,
            "k"       : k,
            "pe"      : pe,           # used in eq 31
            "acmax"   : acmax,        # used in eq 31
            "kiny"    : kiny,         # used in eq 31
            "att"     : att,          # used in eq 31
            "chizero" : chizero,      # used in eq 31
            "t2"      : t2,           # used in eq 31
            "chih2"   : chih2,        # used in eq 31
            }

    #
    # q-scan % Guigay&Ferrero 2016 eq 31
    #
    def qscan(self, qmin=0.0, qmax=10000.0, npoints=10):
        """
        On-axis (symmetry-centre) amplitude versus crystal-to-observation
        distance q, i.e. the dynamical-focusing curve. For each q the field is
        evaluated at the centre of the pattern (x - x_c = 0) using GF2016 eq. 31
        (finite p) or eq. 24 (p = 0); the q = 0 point uses eq. 30 / eq. 23.

        Note: ``R`` (and hence the constants) is fixed for the instance, so a
        chromatic-focusing scan (R matched to each q) is done by creating one
        instance per q, as in GFBMP2013_figs.py.

        Parameters
        ----------
        qmin, qmax : float
            Range of q in mm.
        npoints : int
            Number of q samples.

        Returns
        -------
        qq : ndarray
            The q values in mm.
        yy_amplitude : ndarray of complex
            Complex on-axis amplitude at each q (intensity = |.|^2).
        """
        qq = numpy.linspace(qmin, qmax, npoints)
        yy_amplitude = numpy.zeros_like(qq, dtype=complex)

        kwds_eq30 = self._calculate_constats_for_equation30_2016()
        kwds_eq31 = self._calculate_constats_for_equation31_2016()
        a = kwds_eq31['a']

        print("Calculating q-scan at p=%.3f mm..." % self._p)
        t0 = time.time()
        print(f"Progress: 0%")
        for j in range(qq.size):
            progress = (j + 1) / qq.size * 100
            if progress % 10 < (1 / qq.size * 100):  print(f"Progress: {progress:.0f}%")

            if qq[j] == 0:
                if self._p == 0.0:
                    # p = q = 0: eq. 23 centre is x = 0 (x_c = q*(...) = 0).
                    amplitude = self._equation23_2016(0.0, **kwds_eq31)
                else:
                    # 2026: eq. 30 uses the true x; for finite p the q = 0 symmetry centre is the
                    # q -> 0 limit of the eq. 31 lateral shift (not 0). Evaluate eq. 30 there.
                    xcenter = self._xc_equation30(**kwds_eq31)
                    amplitude = self._equation30_2016(xcenter, **kwds_eq30)
                yy_amplitude[j] = amplitude
            else:
                # 2026: _equation24/31_2016 are now functions of the true x, so evaluate at the
                # pattern centre x = x_c to obtain the central (symmetry-centre) intensity.
                if self._p == 0.0:
                    xc = self._xc_equation24(qq[j], **kwds_eq31)
                    amplitude = self._equation24_2016(xc, qq[j], **kwds_eq31)
                else:
                    xc = self._xc_equation31(qq[j], **kwds_eq31)
                    amplitude = self._equation31_2016(xc, qq[j], **kwds_eq31)
                yy_amplitude[j] = amplitude
        print(f"Progress: 100%")
        print("Calculation time: ", time.time() - t0)

        return qq, yy_amplitude

    def info(self):
        """
        Return a multi-line string summarizing the configured parameters
        (crystal, hkl, R, Poisson ratio, energy, thickness, p, alfa, integration
        points, verbosity).
        """
        txt = ""
        txt += "\nself._crystal_descriptor    = %s" % (self._crystal_descriptor)
        txt += "\nself._hkl                   = " + repr(self._hkl)
        txt += "\nself._R                     = %f mm" % (self._R                   )
        txt += "\nself._poisson_ratio         = %f" % (self._poisson_ratio       )
        txt += "\nself._photon_energy_in_keV  = %f keV" % (self._photon_energy_in_keV)
        txt += "\nself._thickness             = %f mm" % (self._thickness           )
        txt += "\nself._p                     = %f mm" % (self._p                   )
        txt += "\nself._alfa_deg              = %f deg" % (self._alfa_deg            )
        txt += "\nself._integration_points    = %s " % (self._integration_points  )
        txt += "\nself._verbose               = %s " % (self._verbose             )
        txt += "\n"
        return txt


if __name__ == "__main__":

    from srxraylib.plot.gol import plot, set_qt, plot_show

    # Fig 5
    if False:
        a = LaueCrystalFocusing(
            R = 2000,
            poisson_ratio = 0.2201,
            photon_energy_in_keV = 20.0,
            thickness = 0.250,  # mm
            p = 29000.0,  # mm
            alfa_deg = 2.0,  # CAN BE POSITIVE OR NEGATIVE)
            use_fast_hyp1f1=0,
            verbose=0,
            )

        qq, yy_amplitude = a.qscan(qmin=0.0, qmax=10000.0, npoints=200)
        plot(qq, numpy.abs(yy_amplitude) ** 2, xtitle='q [mm]', ytitle="Intensity", title="", grid=1, show=1)


    #
    # fig 2
    #
    if False:
        a = LaueCrystalFocusing(
            R = 2000,
            poisson_ratio = 0.2201,
            photon_energy_in_keV = 80.0,
            thickness = 1.0,  # mm
            p = 0.0,  # mm
            alfa_deg = -0.05,  # CAN BE POSITIVE OR NEGATIVE)
            use_fast_hyp1f1=0,
            verbose=0,
            )

        # xx, yy_amplitude, _ = a.xscan_at_q0(npoints_x=500, a_factor=2, a_center=0.01511, filename="tmp2016_q0.h5")
        xx, yy_amplitude, _ = a.xscan(q=1671.1, npoints_x=500, a_factor=1.0, a_center=0.0, filename="tmp2016_q0.h5")  # same as before

        # xx, yy_amplitude, _ = a.xscan_at_q0(npoints_x=1000, a_factor=3, a_center=0.0, filename="tmp2016_q0.h5")

        print(a.info())
        # xx, yy_amplitude, _ = a.xscan_at_finite_q(q=437.275, npoints_x=200, a_factor=3, a_center=0.0, filename="tmp2016.h5")
        #
        plot(xx, numpy.abs(yy_amplitude) ** 2, xtitle='x [mm]', ytitle="Intensity", title="", grid=1, show=1)


    # external wavefront
    #
    # fig 2
    #
    if False:
        a = LaueCrystalFocusing(
            R = 2000,
            poisson_ratio = 0.2201,
            photon_energy_in_keV = 80.0,
            thickness = 1.0,  # mm
            p = 0.0,  # mm
            alfa_deg = -0.05,  # CAN BE POSITIVE OR NEGATIVE)
            use_fast_hyp1f1=0,
            verbose=0,
            )

        print(a.info())

        xx, yy_amplitude, _ = a.xscan_for_external_wavefront(npoints_x=500, a_factor=1.0, a_center=0.0, filename="")  # same as before

        plot(xx, numpy.abs(yy_amplitude) ** 2, xtitle='x [mm]', ytitle="Intensity", title="", grid=1, show=1)

    #
    # rocking curve
    #
    if True:
        a = LaueCrystalFocusing(
            crystal_descriptor='Diamond',
            hkl=[1, 1, 1],
            R = -1e3, #mm
            poisson_ratio = 0.2201,
            photon_energy_in_keV = 17.0,
            thickness = 155e-3,  # mm
            p = 0.0,  # mm
            alfa_deg = 5,  # CAN BE POSITIVE OR NEGATIVE)
            use_fast_hyp1f1=0,
            verbose=1,
            )

        print(a.info())

        THETA = numpy.linspace(-0.0001, 0.0001, 1000)
        AMPL = a.diffraction_profile_angle_scan(THETA)
        plot(THETA, numpy.abs(AMPL) ** 2, title="Diffraction Profile", grid=1, show=1)
