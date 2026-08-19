"""
FresnelConvolution1D — 1-D near-field Fresnel propagator using direct spatial-domain convolution (numpy.convolve),
with analytical normalization (amplitude factor e^{ikz}/sqrt(i lambda z) and integration measure delta_x).
"""
import numpy

from srxraylib.util.data_structures import ScaledArray
from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
from wofry.propagator.propagator import Propagator1D

class FresnelConvolution1D(Propagator1D):

    HANDLER_NAME = "FRESNEL_CONVOLUTION_1D"

    def get_handler_name(self):
        return self.HANDLER_NAME

    def do_specific_progation_after(self, wavefront, propagation_distance, parameters=None, element_index=None):
        return self.do_specific_progation(wavefront, propagation_distance, parameters=parameters, element_index=element_index)

    def do_specific_progation_before(self, wavefront, propagation_distance, parameters=None, element_index=None):
        return self.do_specific_progation( wavefront, propagation_distance, parameters=parameters, element_index=element_index)

    def do_specific_progation(self, wavefront, propagation_distance, parameters=None, element_index=None):
        """
        Propagate a 1-D wavefront using direct spatial-domain convolution with the Fresnel kernel.

        Parameters
        ----------
        wavefront : GenericWavefront1D
            Input wavefront.
        propagation_distance : float
            Propagation distance [m].
        parameters : PropagationParameters, optional
            Propagation parameter container.
        element_index : int, optional
            Index of the beamline element being propagated through.

        Returns
        -------
        GenericWavefront1D
            Propagated wavefront with analytical normalization applied.
        """
        # instead of numpy.convolve, this can be used:
        # from scipy.signal import fftconvolve
        return self.propagate_wavefront(wavefront,propagation_distance)

    @classmethod
    def propagate_wavefront(cls,wavefront,propagation_distance):

        kernel = numpy.exp(1j*2*numpy.pi/wavefront.get_wavelength() * wavefront.get_abscissas()**2 / 2 / propagation_distance)
        kernel *= numpy.exp(1j*2*numpy.pi/wavefront.get_wavelength() * propagation_distance)
        # 1D amplitude factor 1/sqrt(i lambda z) (the 2D one is 1/(i lambda z))
        kernel /=  numpy.sqrt(1j * wavefront.get_wavelength() * propagation_distance)
        tmp = numpy.convolve(wavefront.get_complex_amplitude(),kernel,mode='same')
        # integration measure delta_x of the discretized convolution integral
        tmp *= wavefront.delta()

        wavefront_out =  GenericWavefront1D(wavefront.get_wavelength(), ScaledArray.initialize_from_steps(tmp,
                                    wavefront.offset(), wavefront.delta()))

        return wavefront_out