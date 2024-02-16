#   propagate_2D_integral: Simplification of the Kirchhoff-Fresnel integral. TODO: Very slow and give some problems

import numpy

from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
from wofry.propagator.propagator import Propagator2D

# TODO: check resulting amplitude normalization (fft and srw likely agree, convolution gives too high amplitudes, so needs normalization)

class Integral2D(Propagator2D):

    HANDLER_NAME = "INTEGRAL_2D"

    def get_handler_name(self):
        return self.HANDLER_NAME

    def do_specific_progation_after(self, wavefront, propagation_distance, parameters, element_index=None):
        return self.do_specific_progation(wavefront, propagation_distance, parameters, element_index=element_index)

    def do_specific_progation_before(self, wavefront, propagation_distance, parameters, element_index=None):
        return self.do_specific_progation( wavefront, propagation_distance, parameters, element_index=element_index)

    """
    2D Fresnel-Kirchhoff propagator via simplified integral

    NOTE: this propagator is experimental and much less performant than the ones using Fourier Optics
          Therefore, it is not recommended to use.

    :param wavefront:
    :param propagation_distance: propagation distance
    :param shuffle_interval: it is known that this method replicates the central diffraction spot
                            The distace of the replica is proportional to 1/pixelsize
                            To avoid that, it is possible to change a bit (randomly) the coordinates
                            of the wavefront. shuffle_interval controls this shift: 0=No shift. A typical
                             value can be 1e5.
                             The result shows a diffraction pattern without replica but with much noise.
    :param calculate_grid_only: if set, it calculates only the horizontal and vertical profiles, but returns the
                             full image with the other pixels to zero. This is useful when calculating large arrays,
                             so it is set as the default.
    :return: a new 2D wavefront object with propagated wavefront
    """

    def do_specific_progation(self, wavefront, propagation_distance, parameters, element_index=None):

        shuffle_interval = self.get_additional_parameter("shuffle_interval",False,parameters,element_index=element_index)
        calculate_grid_only = self.get_additional_parameter("calculate_grid_only",True,parameters,element_index=element_index)
        m_x = self.get_additional_parameter("magnification_x",1.0,parameters,element_index=element_index)
        m_y = self.get_additional_parameter("magnification_y",1.0,parameters,element_index=element_index)

        if calculate_grid_only:
            return self.propagate_wavefront_calculate_grid_only(wavefront,
                                                                propagation_distance,
                                                                magnification_x=m_x,
                                                                magnification_y=m_y)
        else:
            return self.propagate_wavefront(wavefront,
                                            propagation_distance,
                                            shuffle_interval=shuffle_interval,
                                            magnification_x=m_x,
                                            magnification_y=m_y)

    @classmethod
    def propagate_wavefront(cls, wavefront, propagation_distance, shuffle_interval=False,
                            magnification_x=1.0, magnification_y=1.0):
        #
        # Fresnel-Kirchhoff integral (neglecting inclination factor)
        #

        #
        # calculation over the whole detector area
        #
        p_x = wavefront.get_coordinate_x()
        p_y = wavefront.get_coordinate_y()
        wavelength = wavefront.get_wavelength()
        amplitude = wavefront.get_complex_amplitude()

        det_x = p_x.copy() * magnification_x
        det_y = p_y.copy() * magnification_y

        p_X = wavefront.get_mesh_x()
        p_Y = wavefront.get_mesh_y()

        det_X = numpy.outer(det_x, numpy.ones_like(det_y))
        det_Y = numpy.outer(numpy.ones_like(det_x), det_y)

        amplitude_propagated = numpy.zeros_like(amplitude,dtype='complex')

        wavenumber = 2 * numpy.pi / wavelength

        for i in range(det_x.size):
            for j in range(det_y.size):
                if not shuffle_interval:
                    rd_x = 0.0
                    rd_y = 0.0
                else:
                    rd_x = (numpy.random.rand(p_x.size,p_y.size)-0.5)*shuffle_interval
                    rd_y = (numpy.random.rand(p_x.size,p_y.size)-0.5)*shuffle_interval

                r = numpy.sqrt( numpy.power(p_X + rd_x - det_X[i,j],2) +
                                numpy.power(p_Y + rd_y - det_Y[i,j],2) +
                                numpy.power(propagation_distance,2) )

                amplitude_propagated[i,j] = (amplitude / r * numpy.exp(1.j * wavenumber *  r)).sum()

        output_wavefront = GenericWavefront2D.initialize_wavefront_from_arrays(det_x,det_y,amplitude_propagated)

        # added srio@esrf.eu 2018-03-23 to conserve energy - TODO: review method!
        output_wavefront.rescale_amplitude( numpy.sqrt(wavefront.get_intensity().sum() /
                                                    output_wavefront.get_intensity().sum()))

        return output_wavefront

    @classmethod
    def propagate_wavefront_calculate_grid_only(cls, wavefront, propagation_distance,
                            magnification_x=1.0, magnification_y=1.0):
        #
        # Fresnel-Kirchhoff integral (neglecting inclination factor)
        #

        x = wavefront.get_coordinate_x()
        y = wavefront.get_coordinate_y()

        X = numpy.outer(x, numpy.ones_like(y))
        Y = numpy.outer(numpy.ones_like(x), y)

        X_flatten = X.flatten()
        Y_flatten = Y.flatten()

        det_x = x.copy() * magnification_x
        det_y = y.copy() * magnification_y

        det_X = numpy.outer(det_x, numpy.ones_like(det_y))
        det_Y = numpy.outer(numpy.ones_like(det_x), det_y)

        det_X_flatten = det_X.flatten()
        det_Y_flatten = det_Y.flatten()

        wavenumber = 2 * numpy.pi / wavefront.get_wavelength()
        amplitude = wavefront.get_complex_amplitude()

        indices_x = numpy.outer(numpy.arange(0, wavefront.size()[0]), numpy.ones(wavefront.size()[1]))
        indices_y = numpy.outer(numpy.ones(wavefront.size()[0]), numpy.arange(0, wavefront.size()[1]))

        fla_x_indices = indices_x.flatten()
        fla_y_indices = indices_y.flatten()

        weights = numpy.zeros_like(X)
        weights[weights.shape[0] // 2, :] = 1
        weights[:, weights.shape[1] // 2] = 1
        fla_weights = weights.flatten()

        good_indices = numpy.argwhere(fla_weights == 1)
        ngood = good_indices.size

        det_X_flatten_good = det_X_flatten[good_indices]
        det_Y_flatten_good = det_Y_flatten[good_indices]

        print("propagate_2D_integral: Calculating %d points from a total of %d x %d = %d"%(
            ngood, amplitude.shape[0], amplitude.shape[1], amplitude.shape[0] * amplitude.shape[1]))

        externalize_loop = 1
        if externalize_loop:
            fla_complex_amplitude_propagated = cls.propagate_complex_amplitude_from_arrays(
                amplitude.flatten(),
                X_flatten,
                Y_flatten,
                det_X_flatten=det_X_flatten_good,
                det_Y_flatten=det_Y_flatten_good,
                wavelength=wavefront.get_wavelength(),
                propagation_distance=propagation_distance)
        else:
            Propagation_distance = numpy.ones_like(X) * propagation_distance
            fla_complex_amplitude_propagated = numpy.zeros(ngood, dtype=complex)
            for i in range(ngood):
                r = numpy.sqrt( (X - det_X_flatten_good[i])**2 +
                                (Y - det_Y_flatten_good[i])**2 +
                                Propagation_distance**2 )

                fla_complex_amplitude_propagated[i] = (amplitude / r * numpy.exp(1.j * wavenumber *  r)).sum()

            # added srio@esrf.eu 2018-03-23 to conserve energy - TODO: review method!
            i0 = numpy.abs(amplitude) ** 2
            i1 = numpy.abs(fla_complex_amplitude_propagated) ** 2
            fla_complex_amplitude_propagated *= i0.sum() / i1.sum()

        complex_amplitude_propagated = numpy.zeros_like(amplitude, dtype=complex)

        for i in range(ngood):
            ix = int(fla_x_indices[good_indices][i])
            iy = int(fla_y_indices[good_indices][i])
            complex_amplitude_propagated[ix, iy] = fla_complex_amplitude_propagated[i]

        output_wavefront = GenericWavefront2D.initialize_wavefront_from_arrays(x_array=det_x,
                                                                               y_array=det_y,
                                                                               z_array=complex_amplitude_propagated,
                                                                               wavelength=wavefront.get_wavelength())

        return output_wavefront


    @classmethod
    def propagate_complex_amplitude_from_arrays(cls,
                                                amplitude_flatten,
                                                X_flatten,
                                                Y_flatten,
                                                det_X_flatten=None,
                                                det_Y_flatten=None,
                                                wavelength=1e-10,
                                                propagation_distance=100):
        #
        # Fresnel-Kirchhoff integral (neglecting inclination factor)
        #
        if det_X_flatten is None: det_X_flatten = X_flatten
        if det_Y_flatten is None: det_Y_flatten = Y_flatten

        ngood = det_X_flatten.size

        fla_complex_amplitude_propagated = numpy.zeros(ngood, dtype=complex)

        Propagation_distance = numpy.ones_like(X_flatten) * propagation_distance

        wavenumber = 2 * numpy.pi / wavelength

        for i in range(ngood):
            r = numpy.sqrt( (X_flatten - det_X_flatten[i])**2 +
                            (Y_flatten - det_Y_flatten[i])**2 +
                            Propagation_distance**2 )

            fla_complex_amplitude_propagated[i] = (amplitude_flatten / r * numpy.exp(1.j * wavenumber *  r)).sum()


        # added srio@esrf.eu 2018-03-23 to conserve energy - TODO: review method!
        i0 = numpy.abs(amplitude_flatten)**2
        i1 = numpy.abs(fla_complex_amplitude_propagated)**2

        fla_complex_amplitude_propagated *= i0.sum() / i1.sum()

        return fla_complex_amplitude_propagated