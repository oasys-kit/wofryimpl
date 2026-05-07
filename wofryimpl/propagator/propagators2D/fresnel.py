"""
Fresnel2D — 2-D near-field Fresnel propagator using FFT-based transfer-function convolution.

Recommendations: use an even number of bins and ``shift_half_pixel=1`` (the default) for
best agreement with SRW results.
"""
import numpy

from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
from wofry.propagator.propagator import Propagator2D

class Fresnel2D(Propagator2D):

    HANDLER_NAME = "FRESNEL_2D"

    def get_handler_name(self):
        return self.HANDLER_NAME

    def do_specific_progation_after(self, wavefront, propagation_distance, parameters, element_index=None):
        return self.do_specific_progation(wavefront, propagation_distance, parameters, element_index=element_index)

    def do_specific_progation_before(self, wavefront, propagation_distance, parameters, element_index=None):
        return self.do_specific_progation( wavefront, propagation_distance, parameters, element_index=element_index)

    def do_specific_progation(self, wavefront, propagation_distance, parameters, element_index=None):
        """
        Propagate a 2-D wavefront using the Fresnel (near-field) FFT transfer-function method.

        Parameters
        ----------
        wavefront : GenericWavefront2D
            Input wavefront.
        propagation_distance : float
            Propagation distance [m].
        parameters : PropagationParameters
            Propagation parameter container (may include ``shift_half_pixel``).
        element_index : int, optional
            Index of the beamline element being propagated through.

        Returns
        -------
        GenericWavefront2D
            Propagated wavefront on the same spatial grid.
        """

        shift_half_pixel = self.get_additional_parameter("shift_half_pixel",False,parameters,element_index=element_index)

        return self.propagate_wavefront(wavefront,propagation_distance,shift_half_pixel=shift_half_pixel)

    @classmethod
    def propagate_wavefront(cls,wavefront,propagation_distance,shift_half_pixel=False):
        wavelength = wavefront.get_wavelength()

        #
        # convolving with the Fresnel kernel via FFT multiplication
        #
        fft = numpy.fft.fft2(wavefront.get_complex_amplitude())

        # frequency for axis 1
        shape = wavefront.size()
        delta = wavefront.delta()

        pixelsize = delta[0] # p_x[1] - p_x[0]
        npixels = shape[0]
        freq_nyquist = 0.5/pixelsize
        freq_n = numpy.linspace(-1.0,1.0,npixels)
        freq_x = freq_n * freq_nyquist

        # frequency for axis 2
        pixelsize = delta[1]
        npixels = shape[1]
        freq_nyquist = 0.5/pixelsize
        freq_n = numpy.linspace(-1.0,1.0,npixels)
        freq_y = freq_n * freq_nyquist

        if shift_half_pixel:
            freq_x = freq_x - 0.5 * numpy.abs(freq_x[1] - freq_x[0])
            freq_y = freq_y - 0.5 * numpy.abs(freq_y[1] - freq_y[0])

        freq_xy = numpy.array(numpy.meshgrid(freq_y,freq_x))
        fft *= numpy.exp((-1.0j) * numpy.pi * wavelength * propagation_distance *
                      numpy.fft.fftshift(freq_xy[0]*freq_xy[0] + freq_xy[1]*freq_xy[1]) )

        wf_propagated = GenericWavefront2D.initialize_wavefront_from_arrays(x_array=wavefront.get_coordinate_x(),
                                                                            y_array=wavefront.get_coordinate_y(),
                                                                            z_array=numpy.fft.ifft2(fft),
                                                                            wavelength=wavelength)

        return wf_propagated
