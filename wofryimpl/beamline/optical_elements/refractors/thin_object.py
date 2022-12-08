import numpy
from scipy.interpolate import interp2d
import scipy.constants as codata
import xraylib

from oasys.util.oasys_util import write_surface_file, read_surface_file
from oasys.util.oasys_objects import OasysSurfaceData


from syned.beamline.optical_element import OpticalElement
from syned.widget.widget_decorator import WidgetDecorator

from wofry.beamline.decorators import OpticalElementDecorator


# mimics a syned element
class ThinObject(OpticalElement):
    def __init__(self,
                 name="Undefined",
                 file_with_thickness_mesh="",
                 material=""):

        super().__init__(name=name,
                         boundary_shape=None)
        self._material = material
        self._file_with_thickness_mesh = file_with_thickness_mesh

        # support text contaning name of variable, help text and unit. Will be stored in self._support_dictionary
        self._set_support_text([
                    ("name",                       "Name" ,                                "" ),
                    ("boundary_shape",             "Boundary shape",                       "" ),
                    ("material",                   "Material (element, compound or name)", "" ),
                    ("file_with_thickness_mesh",   "File with thickness mesh",             "" ),
            ] )


    def get_material(self):
        return self._material

    def get_file_with_thickness_mesh(self):
        return self._file_with_thickness_mesh

# the wofry element
class WOThinObject(ThinObject, OpticalElementDecorator):
    def __init__(self,
                 name="Undefined",
                 file_with_thickness_mesh="",
                 material="",
                 refraction_index_delta=1e-07,
                 att_coefficient=0.0,
                 verbose=1,
                 ):
        ThinObject.__init__(self,
                      name=name,
                      file_with_thickness_mesh=file_with_thickness_mesh,
                      material=material)

        self._refraction_index_delta = refraction_index_delta
        self._att_coefficient = att_coefficient
        self._verbose = verbose

    def get_refraction_index(self, photon_energy=10000.0):

        wave_length = codata.h * codata.c / codata.e / photon_energy

        if self.get_material() == "External": # Be
             return self._refraction_index_delta, \
                    self._att_coefficient

        if self.get_material() == "Be": # Be
            element = "Be"
            density = xraylib.ElementDensity(4)
        elif self.get_material() == "Al": # Al
            element = "Al"
            density = xraylib.ElementDensity(13)
        elif self.get_material() == "Diamond": # Diamond
            element = "C"
            density = 3.51
        else:
            raise Exception("Bad material: " + self.get_material())

        refraction_index = xraylib.Refractive_Index(element, photon_energy/1000, density)
        refraction_index_delta = 1 - refraction_index.real
        att_coefficient = 4*numpy.pi * (xraylib.Refractive_Index(element, photon_energy/1000, density)).imag / wave_length

        return refraction_index_delta, att_coefficient

    def get_surface_thickness_mesh(self, wavefront):
        xx, yy, zz = read_surface_file(self.get_file_with_thickness_mesh())

        if zz.min() < 0: zz -= zz.min()

        f = interp2d(xx, yy, zz, kind='linear', bounds_error=False, fill_value=0)
        x = wavefront.get_coordinate_x()
        y = wavefront.get_coordinate_y()
        interpolated_profile = f(x, y)
        return x, y, interpolated_profile

    def applyOpticalElement(self, wavefront, parameters=None, element_index=None):

        if self._verbose:
            print("\n\n\n ==========  parameters from optical element : ")
            print(self.info())

        photon_energy = wavefront.get_photon_energy()

        refraction_index_delta, att_coefficient = self.get_refraction_index(photon_energy)

        x, y, interpolated_profile = self.get_surface_thickness_mesh(wavefront)

        amp_factors = numpy.exp(-1.0 * att_coefficient * interpolated_profile / 2) # factor of 2 because it is amplitude
        phase_shifts = -1.0 * wavefront.get_wavenumber() * refraction_index_delta * interpolated_profile

        output_wavefront = wavefront.duplicate()
        output_wavefront.rescale_amplitudes(amp_factors.T)
        output_wavefront.add_phase_shifts(phase_shifts.T)

        return output_wavefront
    def to_python_code(self, data=None):
        txt  = ""
        txt += "\nfrom wofryimpl.beamline.optical_elements.refractors.thin_object import WOThinObject"
        txt += "\n"
        if self.get_material() == "External":
            txt += "\noptical_element = WOThinObject(name='%s',file_with_thickness_mesh='%s',material='%s',refraction_index_delta=%g,att_coefficient=%g,verbose=%d)" % \
                   (self.get_name(), self.get_file_with_thickness_mesh(), self.get_material(), self._refraction_index_delta, self._att_coefficient, self._verbose)
        else:
            txt += "\noptical_element = WOThinObject(name='%s',file_with_thickness_mesh='%s',material='%s',verbose=%d)" % \
                   (self.get_name(), self.get_file_with_thickness_mesh(), self.get_material(), self._verbose)
        txt += "\n"
        return txt


class WOThinObject1D(ThinObject, OpticalElementDecorator):
    def __init__(self,
                 name="Undefined",
                 file_with_thickness_mesh="",
                 material="",
                 refraction_index_delta=1e-07,
                 att_coefficient=0.0,
                 verbose=1,
                 ):
        super().__init__(
                      name=name,
                      file_with_thickness_mesh=file_with_thickness_mesh,
                      material=material)

        self._refraction_index_delta = refraction_index_delta
        self._att_coefficient = att_coefficient
        self._verbose = verbose

    def get_surface_thickness_mesh(self, wavefront):
        a = numpy.loadtxt(self.get_file_with_thickness_mesh())
        xx = a[:,0].copy()
        zz = a[:,1].copy()

        if zz.min() < 0: zz -= zz.min()
        x = wavefront.get_abscissas()
        interpolated_profile = numpy.interp(x, xx, zz)
        return x, interpolated_profile

    def get_refraction_index(self, photon_energy=10000.0):

        wave_length = codata.h * codata.c / codata.e / photon_energy

        if self.get_material() == "External": # Be
             return self._refraction_index_delta, \
                    self._att_coefficient

        if self.get_material() == "Be": # Be
            element = "Be"
            density = xraylib.ElementDensity(4)
        elif self.get_material() == "Al": # Al
            element = "Al"
            density = xraylib.ElementDensity(13)
        elif self.get_material() == "Diamond": # Diamond
            element = "C"
            density = 3.51
        else:
            raise Exception("Bad material: " + self.get_material())

        refraction_index = xraylib.Refractive_Index(element, photon_energy/1000, density)
        refraction_index_delta = 1 - refraction_index.real
        att_coefficient = 4*numpy.pi * (xraylib.Refractive_Index(element, photon_energy/1000, density)).imag / wave_length

        if False:
            print("\n\n\n ==========  parameters recovered from xraylib : ")
            print("Element: %s" % element)
            print("        density = %g " % density)
            print("Photon energy = %g eV" % (photon_energy))
            print("Refracion index delta = %g " % (refraction_index_delta))
            print("Attenuation coeff mu = %g m^-1" % (att_coefficient))

        return refraction_index_delta, att_coefficient

    def applyOpticalElement(self, wavefront, parameters=None, element_index=None):

        if self._verbose:
            print("\n\n\n ==========  parameters from optical element : ")
            print(self.info())

        photon_energy = wavefront.get_photon_energy()

        refraction_index_delta, att_coefficient = self.get_refraction_index(photon_energy)

        x, interpolated_profile = self.get_surface_thickness_mesh(wavefront)
        #
        amp_factors = numpy.exp(-1.0 * att_coefficient * interpolated_profile / 2) # factor of 2 because it is amplitude
        phase_shifts = -1.0 * wavefront.get_wavenumber() * refraction_index_delta * interpolated_profile

        output_wavefront = wavefront.duplicate()
        output_wavefront.rescale_amplitudes(amp_factors)
        output_wavefront.add_phase_shifts(phase_shifts)

        return output_wavefront

    def to_python_code(self, data=None):
        txt  = ""
        txt += "\nfrom wofryimpl.beamline.optical_elements.refractors.thin_object import WOThinObject1D"
        txt += "\n"
        if self.get_material() == "External":
            txt += "\noptical_element = WOThinObject1D(name='%s',file_with_thickness_mesh='%s',material='%s',refraction_index_delta=%g,att_coefficient=%g)" % \
                   (self.get_name(), self.get_file_with_thickness_mesh(), self.get_material(), self._refraction_index_delta, self._att_coefficient)
        else:
            txt += "\noptical_element = WOThinObject1D(name='%s',file_with_thickness_mesh='%s',material='%s')" % \
                   (self.get_name(), self.get_file_with_thickness_mesh(), self.get_material())

        txt += "\n"
        return txt

if __name__ == "__main__":

    import numpy
    from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
    from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
    from srxraylib.plot.gol import plot, plot_image
    import requests

    #
    # 2D
    #
    if True:
        url = 'https://raw.githubusercontent.com/oasys-esrf-kit/dabam2d/main/data/dabam2d-001.h5'
        response = requests.get(url)
        open("dabam2d-001.h5", "wb").write(response.content)

        input_wavefront = GenericWavefront2D.initialize_wavefront_from_range(x_min=-0.0003, x_max=0.0003, y_min=-0.0003,
                                                                              y_max=0.0003, number_of_points=(400, 200))
        input_wavefront.set_photon_energy(10000)
        input_wavefront.set_plane_wave_from_complex_amplitude(complex_amplitude=complex(1, 0))

        optical_element = WOThinObject(name='ThinObject',
                                       file_with_thickness_mesh='dabam2d-001.h5',
                                       material='Be', verbose=1)

        # no drift in this element
        output_wavefront = optical_element.applyOpticalElement(input_wavefront)

        #
        # ---- plots -----
        #
        plot_image(output_wavefront.get_intensity(), output_wavefront.get_coordinate_x(),
                   output_wavefront.get_coordinate_y(), aspect='auto', title='OPTICAL ELEMENT NR 1')


    #
    # 1D
    #

    if True:
        url = 'https://raw.githubusercontent.com/oasys-esrf-kit/dabam2d/main/data/dabam2d-001.h5'
        response = requests.get(url)
        open("dabam2d-001.h5", "wb").write(response.content)

        input_wavefront = GenericWavefront1D.initialize_wavefront_from_range(x_min=-0.0003, x_max=0.0003,
                                                                             number_of_points=400)
        input_wavefront.set_photon_energy(10000)
        input_wavefront.set_plane_wave_from_complex_amplitude(complex_amplitude=complex(1, 0))

        xx, yy, zz = read_surface_file('dabam2d-001.h5')
        if zz.min() < 0: zz -= zz.min()
        nx, ny = zz.shape
        x = yy
        z = zz[nx//2,:]
        plot(x,z,title="x,z")

        f = open('profile.dat', 'w')
        for i in range(ny):
            f.write("%g %g\n" % (x[i], z[i]))
        f.close()
        print(">> File profile.dat written to disk.")

        optical_element = WOThinObject1D(name='ThinObject1D',
                                       file_with_thickness_mesh='profile.dat',
                                       material='Be',
                                       verbose=1)


        # print(optical_element.info())

        # no drift in this element
        output_wavefront = optical_element.applyOpticalElement(input_wavefront)

        #
        # ---- plots -----
        #
        plot(output_wavefront.get_abscissas(), output_wavefront.get_intensity(),
                   title='OPTICAL ELEMENT NR 1')