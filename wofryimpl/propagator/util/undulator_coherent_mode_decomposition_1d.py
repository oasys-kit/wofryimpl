import numpy as np

# needed by pySRU
from pySRU.ElectronBeam import ElectronBeam as PysruElectronBeam
from pySRU.MagneticStructureUndulatorPlane import MagneticStructureUndulatorPlane as PysruUndulator
from pySRU.Simulation import create_simulation
from pySRU.TrajectoryFactory import TRAJECTORY_METHOD_ANALYTIC
from pySRU.RadiationFactory import RADIATION_METHOD_APPROX_FARFIELD

# needed for backpropagation
import numpy
from syned.beamline.beamline_element import BeamlineElement
from syned.beamline.element_coordinates import ElementCoordinates
from wofry.propagator.propagator import PropagationManager, PropagationElements, PropagationParameters
from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
from wofryimpl.propagator.propagators1D.fresnel import Fresnel1D
from wofryimpl.propagator.propagators1D.fresnel_convolution import FresnelConvolution1D
from wofryimpl.propagator.propagators1D.fraunhofer import Fraunhofer1D
from wofryimpl.propagator.propagators1D.integral import Integral1D
from wofryimpl.propagator.propagators1D.fresnel_zoom import FresnelZoom1D
from wofryimpl.propagator.propagators1D.fresnel_zoom_scaling_theorem import FresnelZoomScaling1D

# needed for GSM approx
from syned.storage_ring.electron_beam import ElectronBeam
from syned.storage_ring.magnetic_structures.undulator import Undulator

class UndulatorCoherentModeDecomposition1D():
    def __init__(self,
                 electron_energy=6.04,
                 electron_current=0.2,
                 undulator_period=0.032,
                 undulator_nperiods=50,
                 K=0.25,
                 photon_energy=10490.0,
                 abscissas_interval=250e-6,
                 number_of_points=100,
                 distance_to_screen=100,
                 scan_direction="V",
                 magnification_x=None,
                 sigmaxx = 5e-6,
                 sigmaxpxp = 5e-6,
                 useGSMapproximation=False):

        # inputs
        self.electron_energy    = electron_energy
        self.electron_current   = electron_current
        self.undulator_period   = undulator_period
        self.undulator_nperiods = undulator_nperiods
        self.K                  = K
        self.photon_energy      = photon_energy
        self.abscissas_interval = abscissas_interval
        self.number_of_points   = number_of_points
        self.distance_to_screen = distance_to_screen
        if magnification_x is None:
            self.magnification_x = 1.0 / self.distance_to_screen
        else:
            self.magnification_x = magnification_x
        self.scan_direction     = scan_direction
        self.mxx                = 1.0 / sigmaxx**2
        self.mxpxp              = 1.0 / sigmaxpxp**2
        self.useGSMapproximation = useGSMapproximation

        # calculated
        self._abscissas_interval_in_far_field = self.abscissas_interval / self.magnification_x

        # development flags, use with care
        self._use_dirac_deltas = False
        self._use_vectorization = True

        # to store results
        self.far_field_wavefront = None
        self.output_wavefront = None
        self.CSD = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.abscissas = None

    def reset(self):
        self.far_field_wavefront = None
        self.output_wavefront = None
        self.CSD = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.abscissas = None

    def _WWW(self, x1, x2):
        # see Eq. 3.51 in Mark's thesis https://tel.archives-ouvertes.fr/tel-01664052/document
        k = self.output_wavefront.get_wavenumber()
        abscissas = self.output_wavefront.get_abscissas()
        Dx = x2-x1
        Dx1 = int( x1 / (abscissas[1] - abscissas[0]) )
        Dx2 = int( x2 / (abscissas[1] - abscissas[0]) )

        ca = self.output_wavefront.get_complex_amplitude()
        e01 = np.roll(ca.copy(), Dx1 )
        e02 = np.roll(ca.copy(), Dx2 )
        return np.sqrt(self.mxx) / (2 * np.pi) ** (1 / 2) * \
               np.exp(-k ** 2 * Dx ** 2 / 2 / self.mxpxp) * \
               (np.exp(-self.mxx * abscissas ** 2 / 2) * np.conjugate(e01) * e02).sum()


    def _WWW_vector(self, x1):
        # see Eq. 3.53 in Mark's thesis https://tel.archives-ouvertes.fr/tel-01664052/document
        k = self.output_wavefront.get_wavenumber()
        abscissas = self.output_wavefront.get_abscissas()
        Dx = abscissas-x1
        if self._use_dirac_deltas:
            c = self._H_x1(x1) * self.output_wavefront.get_complex_amplitude()
            return c
        else:
            c = np.convolve(self._H_x1(x1), self.output_wavefront.get_complex_amplitude(), mode='same')
            # TODO: check normalization factor
            return np.sqrt(self.mxx) / (2 * np.pi)**(1/2) * np.exp(-k**2 * Dx**2 / 2 / self.mxpxp) *  c

    def _H_x1(self, x1):
        # see Eq. 3.52 in Mark's thesis https://tel.archives-ouvertes.fr/tel-01664052/document
        abscissas = self.output_wavefront.get_abscissas()
        abscissas_step = abscissas[1] - abscissas[0]
        if self._use_dirac_deltas:
            return np.roll(np.conjugate(self.output_wavefront.get_complex_amplitude()), int(x1 // abscissas_step))
        else:
            return np.exp(-self.mxx * abscissas**2 / 2) * np.roll(np.conjugate(self.output_wavefront.get_complex_amplitude()), int(x1 // abscissas_step))

    def calculate(self):
        if not self.useGSMapproximation:
            print("Calculating far field emission using pySRU...")
            self._calculate_far_field()
            print("Calculating backpropagation to source position...")
            self._calculate_backpropagation()
            print("Computing Cross Spectral Density...")
            self._calculate_CSD()
            print("Diagonalizing CSD...")
            self._diagonalize()
            print("Done\n")
            return {"CSD":self.CSD,
                    "abscissas":self.output_wavefront.get_abscissas(),
                    "eigenvalues": self.eigenvalues,
                    "eigenvectors": self.eigenvectors}
        else:
            self._calculate_CSD_GSM()
            self._diagonalize()
            return {"CSD":self.CSD,
                    "abscissas":self.abscissas,
                    "eigenvalues": self.eigenvalues,
                    "eigenvectors": self.eigenvectors}



    def _calculate_far_field(self):
        #
        # undulator emission
        #
        out = self.calculate_undulator_emission(
            electron_energy    = self.electron_energy,
            electron_current   = self.electron_current,
            undulator_period   = self.undulator_period,
            undulator_nperiods = self.undulator_nperiods,
            K                  = self.K,
            photon_energy      = self.photon_energy,
            abscissas_interval_in_far_field = self._abscissas_interval_in_far_field,
            number_of_points   = self.number_of_points,
            distance_to_screen = self.distance_to_screen,
            scan_direction     = self.scan_direction,
        )

        #
        #
        #
        from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D

        input_wavefront = GenericWavefront1D.initialize_wavefront_from_arrays(out["abscissas"],
                                                                              out["electric_field"][:, 0])
        input_wavefront.set_photon_energy(photon_energy=self.photon_energy)

        self.far_field_wavefront = input_wavefront

    def _calculate_backpropagation(self):
        self.output_wavefront = self.backpropagate(input_wavefront=self.far_field_wavefront,
                                         distance=-self.distance_to_screen,
                                         magnification_x=self.magnification_x)

        self.abscissas = self.output_wavefront.get_abscissas()

    def _calculate_CSD(self):

        abscissas = self.output_wavefront.get_abscissas()

        CSD = np.zeros((abscissas.size, abscissas.size), dtype=complex)

        if self._use_vectorization:
            for i in range(abscissas.size):
                CSD[i, :] = self._WWW_vector(abscissas[i])
        else:
            for i in range(abscissas.size):
                for j in range(abscissas.size):
                    tmp = self._WWW(abscissas[i], abscissas[j])
                    CSD[i, j] = tmp

        self.CSD = CSD

    def _calculate_CSD_GSM(self):


        ebeam = ElectronBeam(energy_in_GeV=self.electron_energy, current=self.electron_current)
        su = Undulator.initialize_as_vertical_undulator(K=self.K, period_length=self.undulator_period,
                                                        periods_number=self.undulator_nperiods)

        sigma_u, sigma_up = su.get_sigmas_radiation(ebeam.gamma(), harmonic=1.0)

        self.abscissas = numpy.linspace(-0.5 * self.abscissas_interval,
                                   0.5 * self.abscissas_interval,
                                   self.number_of_points)

        X1 = numpy.outer(self.abscissas, numpy.ones_like(self.abscissas))
        X2 = numpy.outer(numpy.ones_like(self.abscissas), self.abscissas)

        CF = sigma_u * sigma_up / \
            numpy.sqrt(sigma_up ** 2 + 1 / self.mxpxp) / \
            numpy.sqrt(sigma_u ** 2 + 1 / self.mxx)
        sigmaI = numpy.sqrt(sigma_u**2 + 1/self.mxx)
        beta = CF / numpy.sqrt(1.0-CF)
        sigmaMu = beta * sigmaI
        CSD = numpy.exp(-(X1**2+X2**2)/4/sigmaI**2) * numpy.exp(-(X2-X1)**2/2/sigmaMu**2)
        self.CSD = CSD

    def _diagonalize(self, normalize_eigenvectors=False):
        #
        # diagonalize the CSD
        #
        w, v = np.linalg.eig(self.CSD)
        print(w.shape, v.shape)
        idx = w.argsort()[::-1]  # large to small
        self.eigenvalues = np.real(w[idx])
        eigenvectors = v[:, idx].T

        if normalize_eigenvectors:
            abscissas = self.output_wavefront.get_abscissas()
            for i in range(eigenvectors.shape[0]):
                y1 = eigenvectors[i, :]
                y1integral = (np.conjugate(y1) * y1).sum() * (abscissas[1] - abscissas[0])
                eigenvectors[i, :] = y1 / np.sqrt(y1integral)

        self.eigenvectors = eigenvectors
        print("Coherence Fraction (from modes): ", self.eigenvalues[0] / self.eigenvalues.sum())


    def get_abscissas(self):
        return self.abscissas

    def get_eigenvectors(self):
        return self.eigenvectors

    def get_eigenvector_wavefront(self, mode):
        complex_amplitude = self.get_eigenvectors()[mode,:] * np.sqrt(self.get_eigenvalue(mode))
        wf = GenericWavefront1D.initialize_wavefront_from_arrays(
            self.abscissas, complex_amplitude)
        wf.set_photon_energy(self.photon_energy)
        return wf

    def get_eigenvalues(self):
        return self.eigenvalues

    def get_eigenvalue(self, mode):
        return self.eigenvalues[mode]

    @classmethod
    def calculate_undulator_emission(cls,
            electron_energy=6.04,
            electron_current=0.2,
            undulator_period=0.032,
            undulator_nperiods=50,
            K=0.25,
            photon_energy=10490.0,
            abscissas_interval_in_far_field=250e-6,
            number_of_points=100,
            distance_to_screen=100,
            scan_direction="V"):


        myelectronbeam = PysruElectronBeam(Electron_energy=electron_energy, I_current=electron_current)
        myundulator = PysruUndulator(K=K, period_length=undulator_period, length=undulator_period * undulator_nperiods)

        abscissas = np.linspace(-0.5 * abscissas_interval_in_far_field,
                                0.5 * abscissas_interval_in_far_field,
                                number_of_points)

        if scan_direction == "H":
            X = abscissas
            Y = np.zeros_like(abscissas)
        elif scan_direction == "V":
            X = np.zeros_like(abscissas)
            Y = abscissas

        print("   photon energy %g eV" % photon_energy)
        simulation_test = create_simulation(magnetic_structure=myundulator, electron_beam=myelectronbeam,
                                            magnetic_field=None, photon_energy=photon_energy,
                                            traj_method=TRAJECTORY_METHOD_ANALYTIC, Nb_pts_trajectory=None,
                                            rad_method=RADIATION_METHOD_APPROX_FARFIELD, initial_condition=None,
                                            distance=distance_to_screen,
                                            X=X, Y=Y, XY_are_list=True)

        # TODO: this is not nice: I redo the calculations because I need the electric vectors to get polarization
        #       this should be avoided after refactoring pySRU to include electric field in simulations!!
        electric_field = simulation_test.radiation_fact.calculate_electrical_field(
            simulation_test.trajectory, simulation_test.source, X, Y, distance_to_screen)

        E = electric_field._electrical_field
        pol_deg1 = (np.abs(E[:,0]) / (np.abs(E[:,0]) + np.abs(E[:,1]))).flatten() # SHADOW definition!!

        intens1 = simulation_test.radiation.intensity.copy()

        #  Conversion from pySRU units (photons/mm^2/0.1%bw) to SHADOW units (photons/rad^2/eV)
        intens1 *= (distance_to_screen * 1e3) ** 2 # photons/mm^2 -> photons/rad^2
        intens1 /= 1e-3 * photon_energy # photons/o.1%bw -> photons/eV

        # unpack trajectory
        T0 = simulation_test.trajectory
        T = np.vstack((T0.t,T0.x,T0.y,T0.z,T0.v_x,T0.v_y,T0.v_z,T0.a_x,T0.a_y,T0.a_z))

        return {'intensity':intens1,
                'polarization':pol_deg1,
                'electric_field':E,
                'trajectory':T,
                'photon_energy': photon_energy,
                "abscissas":abscissas,
                "D":distance_to_screen,
                "theta": abscissas / distance_to_screen,
                }

    @classmethod
    def backpropagate(cls,
                      input_wavefront,
                      distance=-100.0,
                      handler_name='FRESNEL_ZOOM_1D', # 'INTEGRAL_1D', #
                      magnification_x=1.0,  # used for handler_name='FRESNEL_ZOOM_1D' or 'INTEGRAL_1D',
                      magnification_N=10.0, # only used if handler_name='INTEGRAL_1D',
                      ):
        #
        plot_from_oe = 100  # set to a large number to avoid plots

        ##########  OPTICAL ELEMENT NUMBER 1 ##########

        # input_wavefront = output_wavefront.duplicate()
        from wofryimpl.beamline.optical_elements.ideal_elements.screen import WOScreen1D

        optical_element = WOScreen1D()

        # drift_before 35 m
        #
        # propagating
        #
        #
        propagation_elements = PropagationElements()
        beamline_element = BeamlineElement(optical_element=optical_element,
                                           coordinates=ElementCoordinates(p=distance, q=0.000000,
                                                                          angle_radial=numpy.radians(0.000000),
                                                                          angle_azimuthal=numpy.radians(0.000000)))
        propagation_elements.add_beamline_element(beamline_element)
        propagation_parameters = PropagationParameters(wavefront=input_wavefront, propagation_elements=propagation_elements)
        # self.set_additional_parameters(propagation_parameters)
        #

        if handler_name == 'FRESNEL_ZOOM_1D':
            propagation_parameters.set_additional_parameters('magnification_x', magnification_x)
            #
            propagator = PropagationManager.Instance()
            try:
                propagator.add_propagator(FresnelZoom1D())
            except:
                pass
            output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,
                                                         handler_name='FRESNEL_ZOOM_1D')
        elif handler_name == 'INTEGRAL_1D':
            propagation_parameters.set_additional_parameters('magnification_x', magnification_x)
            propagation_parameters.set_additional_parameters('magnification_N', magnification_N)
            #
            propagator = PropagationManager.Instance()
            try:
                propagator.add_propagator(Integral1D())
            except:
                pass
            output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,
                                                         handler_name='INTEGRAL_1D')
        else:
            raise Exception("Unknown propagator % s" % handler_name)

        #
        # ---- plots -----
        #
        if plot_from_oe <= 1: plot(output_wavefront.get_abscissas()*1e6, output_wavefront.get_intensity(),
                                   title='OPTICAL ELEMENT NR 1',xtitle="x [um]", show=0)

        return output_wavefront



if __name__ == "__main__":
    from srxraylib.plot.gol import plot, plot_image, plot_table, set_qt
    from syned.storage_ring.electron_beam import ElectronBeam
    from syned.storage_ring.magnetic_structures.undulator import Undulator

    set_qt()

    # definitions with syned compatibility
    ebeam = ElectronBeam(energy_in_GeV=6.0, current = 0.2)
    su = Undulator.initialize_as_vertical_undulator(K=1.191085, period_length=0.02, periods_number=100)
    photon_energy = su.resonance_energy(ebeam.gamma(),harmonic=1)
    print("Resonance energy: ", photon_energy)
    # other inputs
    distance_to_screen = 100.0
    number_of_points = 200
    # Electron beam values: sigma_h : 30.184 um, sigma_v:  3.636 um
    sigmaxx = 3.01836e-05
    sigmaxpxp = 4.36821e-06


    # set parameters
    co = UndulatorCoherentModeDecomposition1D(
                                    electron_energy=ebeam.energy(),
                                    electron_current=ebeam.current(),
                                    undulator_period=su.period_length(),
                                    undulator_nperiods=su.number_of_periods(),
                                    K=su.K(),
                                    photon_energy=photon_energy,
                                    abscissas_interval=250e-6,
                                    number_of_points=number_of_points,
                                    distance_to_screen=distance_to_screen,
                                    scan_direction="V",
                                    sigmaxx=sigmaxx,
                                    sigmaxpxp=sigmaxpxp,
                                    useGSMapproximation=False)
    # make calculation
    out = co.calculate()

    CSD = out["CSD"]
    abscissas = out["abscissas"]

    plot_image(np.abs(CSD), abscissas*1e6, abscissas*1e6, title="Cross spectral density", xtitle="x1 [um]", ytitle="x2 [um]")

    SD = np.zeros_like(abscissas)
    for i in range(SD.size):
        SD[i] = CSD[i,i]

    try:
        radiation_intensity = co.output_wavefront.get_intensity()
    except:
        radiation_intensity = abscissas * 0
    weight = np.exp(-abscissas**2 / 2 / sigmaxx**2)
    conv = np.convolve(radiation_intensity, weight, mode='same')
    plot(
         abscissas, 1.01 * SD / SD.max(),
         abscissas, weight,
         abscissas, radiation_intensity / radiation_intensity.max(),
         abscissas, conv / conv.max(),
         legend=["1.01 * Spectral density", "Gaussian with sigmaxx","radiation intensity","rad in x Gaussian with sigmaxx"],
         title="CSD", show=1)

    #
    # plot decomposition
    #
    eigenvalues = out["eigenvalues"]
    eigenvectors = out["eigenvectors"]

    # test normalizetion
    for i in range(10):
        y1 = eigenvectors[i, :]
        print(i, (np.conjugate(y1) * y1).sum() * (abscissas[1] - abscissas[0]),
              (np.conjugate(y1) * eigenvectors[i+1, :]).sum() * (abscissas[1] - abscissas[0]))


    # plot occupation
    nmodes = 100
    plot(np.arange(nmodes), eigenvalues[0:nmodes]/(eigenvalues.sum()),
         title="mode occupation")

    plot(np.arange(nmodes), np.cumsum(eigenvalues[0:nmodes]/(eigenvalues.sum())),
         title="mode cumulated occupation")

    # plot eigenvectors
    plot(abscissas, eigenvectors[0:10,:].T, title="eigenvectors")

    # restore spectral density from modes
    y = np.zeros_like(abscissas, dtype=complex)
    nmodes = 100
    for i in range(nmodes):
        y += eigenvalues[i] * np.conjugate(eigenvectors[i, :]) * eigenvectors[i, :]

    y = np.real(y)
    plot(abscissas, SD,
         abscissas, y, legend=["SD", "SD From modes"])

    print(">>>>>>", co.get_abscissas().shape, co.get_eigenvectors().shape)
    wf0 = co.get_eigenvector_wavefront(0)
    plot(wf0.get_abscissas(), wf0.get_intensity())