"""
FresnelConvolution2D — 2-D near-field Fresnel propagator using direct spatial-domain convolution (scipy.signal.fftconvolve).

Prefer ``Fresnel2D`` (FFT transfer-function method) for better accuracy.
"""
import numpy

from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
from wofry.propagator.propagator import Propagator2D

class FresnelConvolution2D(Propagator2D):

    HANDLER_NAME = "FRESNEL_CONVOLUTION_2D"

    def get_handler_name(self):
        return self.HANDLER_NAME


    def do_specific_progation_after(self, wavefront, propagation_distance, parameters, element_index=None):
        return self.do_specific_progation(wavefront, propagation_distance, parameters, element_index=element_index)

    def do_specific_progation_before(self, wavefront, propagation_distance, parameters, element_index=None):
        return self.do_specific_progation( wavefront, propagation_distance, parameters, element_index=element_index)

    def do_specific_progation(self, wavefront, propagation_distance, parameters, element_index=None):
        """
        Propagate a 2-D wavefront using direct spatial-domain convolution with the Fresnel kernel.

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

        is_generic_wavefront = isinstance(wavefront, GenericWavefront2D)

        if is_generic_wavefront:
            pass
        else:
            wavefront = wavefront.toGenericWavefront()

        return self.propagate_wavefront(wavefront,propagation_distance,shift_half_pixel=shift_half_pixel)

    @classmethod
    def propagate_wavefront(cls,wavefront,propagation_distance,shift_half_pixel=False):

        from scipy.signal import fftconvolve

        wavelength = wavefront.get_wavelength()

        X = wavefront.get_mesh_x()
        Y = wavefront.get_mesh_y()

        if shift_half_pixel:
            x = wavefront.get_coordinate_x()
            y = wavefront.get_coordinate_y()
            X += 0.5 * numpy.abs( x[0] - x[1] )
            Y += 0.5 * numpy.abs( y[0] - y[1] )

        kernel = numpy.exp(1j*2*numpy.pi/wavefront.get_wavelength() *
                           (X**2 + Y**2) / 2 / propagation_distance)
        kernel *= numpy.exp(1j*2*numpy.pi/wavefront.get_wavelength() * propagation_distance)
        kernel /=  1j * wavefront.get_wavelength() * propagation_distance

        wavefront_out = GenericWavefront2D.initialize_wavefront_from_arrays(x_array=wavefront.get_coordinate_x(),
                                                                            y_array=wavefront.get_coordinate_y(),
                                                                            z_array=fftconvolve(wavefront.get_complex_amplitude(),
                                                                                                kernel,
                                                                                                mode='same'),
                                                                            wavelength=wavelength)
        # added srio@esrf.eu 2018-03-23 to conserve energy - TODO: review method!
        wavefront_out.rescale_amplitude( numpy.sqrt(wavefront.get_intensity().sum() /
                                                    wavefront_out.get_intensity().sum()))

        return wavefront_out
