import numpy

from syned.beamline.optical_elements.crystals.crystal import Crystal, DiffractionGeometry
from wofry.beamline.decorators import OpticalElementDecorator
from wofryimpl.beamline.optical_elements.util.laue_crystal_focusing import LaueCrystalFocusing

class WOLaueCrystal1D(Crystal, OpticalElementDecorator):
    def __init__(self, name="",
                 crystal_descriptor="Si",
                 hkl=[1, 1, 1],
                 R=2.0, # m
                 poisson_ratio=0.2201,
                 photon_energy=20000.0,
                 thickness=250e-6,  # m
                 p=2.0,  # m
                 q=0.0,  # m
                 alfa_deg=2.0,  # CAN BE POSITIVE OR NEGATIVE)
                 integration_points=500,
                 npoints_x=100,
                 a_factor=1.0,
                 use_fast_hyp1f1=0,
                 apply_absorption=True,  # FEATURE (2026): pass-through; False reproduces paper2013.py (attsym=1)
                 chih2=None,  # FEATURE (2026): pass-through; if not None, overrides the computed chi_h*chi_hbar
                 source_flag=1,
                 verbose=0,
                 materials_library=None,
                 ):
        Crystal.__init__(self,
                         name,
                         material=crystal_descriptor,
                         diffraction_geometry=DiffractionGeometry.LAUE,
                         miller_index_h=int(hkl[0]),
                         miller_index_k=int(hkl[1]),
                         miller_index_l=int(hkl[2]),
                         asymmetry_angle=numpy.radians(90 - alfa_deg),
                         thickness=thickness,
                         )
        self._LaueCrystalFocusing = LaueCrystalFocusing(
            crystal_descriptor=crystal_descriptor,
            hkl=hkl,
            R=R*1e3, # mm
            poisson_ratio=poisson_ratio,
            photon_energy_in_keV=photon_energy*1e-3,
            thickness=thickness*1e3,  # mm
            p=p*1e3,  # mm
            alfa_deg=alfa_deg,  # CAN BE POSITIVE OR NEGATIVE)
            integration_points=integration_points,
            use_fast_hyp1f1=use_fast_hyp1f1,
            apply_absorption=apply_absorption,  # FEATURE (2026)
            chih2=chih2,  # FEATURE (2026)
            verbose=verbose,
            materials_library=materials_library,
        )
        if verbose: print(self._LaueCrystalFocusing.info())

        self._q = q*1e3 # mm
        self._npoints_x = npoints_x
        self._a_factor  = a_factor
        self._source_flag  = source_flag

    def applyOpticalElement(self, wavefront_in, parameters=None, element_index=None):
        if self._source_flag == 0: # external wavefront
            xx, yy_amplitude, wavefront = self._LaueCrystalFocusing.xscan_for_external_wavefront(
                                                                            Phi=wavefront_in.get_complex_amplitude(),
                                                                            Phi_tau=wavefront_in.get_abscissas() * 1e3,
                                                                            npoints_x=self._npoints_x,
                                                                            a_factor=self._a_factor,
                                                                            a_center=0.0,
                                                                            filename="")
        elif self._source_flag == 1: # point source
            xx, yy_amplitude, wavefront = self._LaueCrystalFocusing.xscan(self._q,
                                                                          npoints_x=self._npoints_x,
                                                                          a_factor=self._a_factor,
                                                                          a_center=0.0,
                                                                          filename="")
        return wavefront

    def qscan(self, qmin=0.0, qmax=10.0, qpoints=100):
        qq, amplitude = self._LaueCrystalFocusing.qscan(qmin=qmin*1e3, qmax=qmax*1e3, npoints=qpoints)
        return qq * 1e-3, amplitude

    def diffraction_profile_angle_scan(self, angle_min=0.0, angle_max=10.0, angle_points=100):
        THETA = numpy.linspace(angle_min, angle_max, angle_points)
        AMPLITUDE = self._LaueCrystalFocusing.diffraction_profile_angle_scan(THETA)
        return THETA, AMPLITUDE

    def to_python_code(self, do_plot=False, add_import_section=False):
        txt  = ""
        txt += "\n"
        txt += "\ntry:"
        txt += "\n    import xraylib as materials_library"
        txt += "\nexcept:"
        txt += "\n    from dabax.dabax_xraylib import DabaxXraylib"
        txt += "\n    materials_library = DabaxXraylib()"
        txt += "\n"
        txt += "\nfrom wofryimpl.beamline.optical_elements.crystals.laue_crystal import WOLaueCrystal1D"
        txt += "\n"
        txt += "\noptical_element = WOLaueCrystal1D(name='%s',"         % self.get_name()
        txt += "\n    crystal_descriptor = '%s',"                     % self._LaueCrystalFocusing._crystal_descriptor
        txt += "\n    hkl = %s,"                                      % self._LaueCrystalFocusing._hkl
        txt += "\n    R = %f, # m"                                    % (self._LaueCrystalFocusing._R * 1e-3)
        txt += "\n    poisson_ratio = %f,"                            % self._LaueCrystalFocusing._poisson_ratio
        txt += "\n    photon_energy = %f,"                            % (self._LaueCrystalFocusing._photon_energy_in_keV * 1e3)
        txt += "\n    thickness = %g,  # m,"                          % (self._LaueCrystalFocusing._thickness * 1e-3)
        txt += "\n    p = %f,  # m"                                   % (self._LaueCrystalFocusing._p * 1e-3)
        txt += "\n    q = %f,  # m"                                   % (self._q * 1e-3)
        txt += "\n    alfa_deg = %f,  # CAN BE POSITIVE OR NEGATIVE)" % self._LaueCrystalFocusing._alfa_deg
        txt += "\n    integration_points = %d,"                       % self._LaueCrystalFocusing._integration_points
        txt += "\n    npoints_x = %d,"                                % self._npoints_x
        txt += "\n    a_factor = %f,"                                 % self._a_factor
        txt += "\n    use_fast_hyp1f1 = %d,"                          % self._LaueCrystalFocusing._use_fast_hyp1f1
        txt += "\n    apply_absorption = %s,"                         % self._LaueCrystalFocusing._apply_absorption
        txt += "\n    chih2 = %s,"                                    % repr(self._LaueCrystalFocusing._chih2)
        txt += "\n    source_flag = %d,"                              % self._source_flag
        txt += "\n    verbose = %d,"                                  % self._LaueCrystalFocusing._verbose
        txt += "\n    materials_library = materials_library)"

        txt += "\n"
        if self._source_flag == 1:
            txt += "\ninput_wavefront = None"
            txt += "\noutput_wavefront = optical_element.applyOpticalElement(input_wavefront)"
        elif self._source_flag == 0:
            txt += "\noutput_wavefront = optical_element.applyOpticalElement(input_wavefront)"

        txt += "\n\n# qq, amplitude = optical_element.qscan(qmin=0.01, qmax=5, qpoints=500)"
        txt += "\n# plot(qq, numpy.abs(amplitude) ** 2, title='q [m]')"

        txt += "\n\n# angle, angle_amplitude = optical_element.diffraction_profile_angle_scan(angle_min=-50e-6, angle_max=50e-6, angle_points=1000)"
        txt += "\n# plot(angle, numpy.abs(angle_amplitude) ** 2, title='Diffraction Profile', xtitle='angle [rad]', ytitle='Intensity [a.u.]')"

        txt += "\n"
        return txt

    #
    # added
    #
    def get_dimension(self):
        return 1
