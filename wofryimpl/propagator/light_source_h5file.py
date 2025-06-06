import numpy
from syned.storage_ring.light_source import LightSource

from wofry.beamline.decorators import LightSourceDecorator

from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D

class WOH5FileLightSource(LightSource, LightSourceDecorator):
    def __init__(self,
                 name                = "Undefined",
                 electron_beam       = None,
                 magnetic_structure  = None,
                 h5file              = "",
                 filepath            = "wfr",
                 ):

        LightSource.__init__(self, name=name, electron_beam=electron_beam, magnetic_structure=magnetic_structure)

        self.__source_wavefront_parameters = {
            'h5file'  : h5file,
            'filepath': filepath,
            }


        self.dimension =  0
        self._h5file = h5file
        self._filepath = filepath
        self._set_support_text([
                    ("name"     ,      "to define ", "" ),
                    ("h5file"  ,       "h5file ",    ""),
                    ("filepath",       "filepath ",  ""),
            ] )


    def get_dimension(self):
        return self.dimension

    def get_source_wavefront_parameters(self):
        return self.__source_wavefront_parameters

    # from Wofry Decorator
    def get_wavefront(self):
        try:
            wf = GenericWavefront1D.load_h5_file(self._h5file, filepath=self._filepath)
        except:
            try:
                wf = GenericWavefront2D.load_h5_file(self._h5file, filepath=self._filepath)
            except:
                raise Exception("Cannot load oasys/wofry wavefront from file %s." % self._h5file)

        self.dimension = int(wf.get_dimension())
        return wf

    def to_python_code(self, do_plot=True, add_import_section=False):
        txt = ""

        txt += "#\n# create output_wavefront\n#"

        txt += "\nfrom wofry.propagator.wavefront%dD.generic_wavefront import GenericWavefront%dD" % \
               (self.dimension, self.dimension,)
        txt += "\noutput_wavefront = GenericWavefront%dD.load_h5_file('%s', filepath='%s')" % \
               (self.dimension, self._h5file, self._filepath)

        return txt



if __name__ == "__main__":


    for h5file in ["/users/srio/OASYS1.2/paper-guigay-resources/scripts_new/crystal_amplitude_8300.h5",
                   "/users/srio/Oasys/tmp.h5"]:
        pp = WOH5FileLightSource(
            name="",
            h5file=h5file,
        )

        wf = pp.get_wavefront()
        print(">>>>> Dimension, Intensity: ", wf.get_dimension(), wf.get_intensity().shape)

        from srxraylib.plot.gol import plot, plot_image
        if pp.get_dimension() == 1:
            plot(wf.get_abscissas(), wf.get_intensity())
        else:
            plot_image(wf.get_intensity(), wf.get_coordinate_x(), wf.get_coordinate_y())
        print(pp.to_python_code())