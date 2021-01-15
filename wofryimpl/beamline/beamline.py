from syned.beamline.beamline import Beamline


class WOBeamline(Beamline):

    def __init__(self,
                 light_source=None,
                 beamline_elements_list=None,
                 propagation_info_list=None):
        super().__init__(light_source=light_source, beamline_elements_list=beamline_elements_list)

        if propagation_info_list is None:
            self._propagation_info_list = [{}] * self.get_beamline_elements_number()
        else:
            self._propagation_info_list = propagation_info_list

    def duplicate(self):
        beamline_elements_list = []
        for beamline_element in self._beamline_elements_list:
            beamline_elements_list.append(beamline_element)

        propagation_info_list  = []
        for propagation_info_list_element in self._propagation_info_list:
            propagation_info_list.append(propagation_info_list_element)

        return WOBeamline(light_source=self._light_source,
                        beamline_elements_list=beamline_elements_list,
                        propagation_info_list=propagation_info_list)

    def append_beamline_element(self, beamline_element, propagation_info=None):
        super().append_beamline_element(beamline_element)
        if propagation_info is None:
            self._propagation_info_list.append({})
        else:
            self._propagation_info_list.append(propagation_info)

    def get_propagation_info_list(self):
        return self._propagation_info_list

    def get_propagation_info_at(self, i):
        return self.get_propagation_info_list()[i]

    def to_python_code(self,do_plot=True):

        text_code = ""

        text_code += "\n#"
        text_code += "\n# Import section"
        text_code += "\n#"
        text_code += "\nimport numpy"
        text_code += "\n\nfrom syned.beamline.beamline_element import BeamlineElement"
        text_code += "\nfrom syned.beamline.element_coordinates import ElementCoordinates"
        text_code += "\nfrom wofry.propagator.propagator import PropagationManager, PropagationElements, PropagationParameters"

        if self.get_light_source() is None:
            source_dimension = 0
            text_code += "\n\n#\n# UNDEFINED SOURCE (please complete...)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n#"
            text_code += "\noutput_wavefront = None"
            text_code += "\n#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        else:
            source_dimension = self.get_light_source().get_dimension()
            if source_dimension == 1:
                text_code += "\n\nfrom wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D"
                text_code += "\n\nfrom wofryimpl.propagator.propagators1D.fresnel import Fresnel1D"
                text_code +=   "\nfrom wofryimpl.propagator.propagators1D.fresnel_convolution import FresnelConvolution1D"
                text_code +=   "\nfrom wofryimpl.propagator.propagators1D.fraunhofer import Fraunhofer1D"
                text_code +=   "\nfrom wofryimpl.propagator.propagators1D.integral import Integral1D"
                text_code +=   "\nfrom wofryimpl.propagator.propagators1D.fresnel_zoom import FresnelZoom1D"
                text_code +=   "\nfrom wofryimpl.propagator.propagators1D.fresnel_zoom_scaling_theorem import FresnelZoomScaling1D"
            elif source_dimension == 2:
                text_code += "\n\nfrom wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D"
                text_code += "\n\nfrom wofryimpl.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D"
                text_code +=   "\nfrom wofryimpl.propagator.propagators2D.fresnel import Fresnel2D"
                text_code +=   "\nfrom wofryimpl.propagator.propagators2D.fresnel_convolution import FresnelConvolution2D"
                text_code +=   "\nfrom wofryimpl.propagator.propagators2D.fraunhofer import Fraunhofer2D"
                text_code +=   "\nfrom wofryimpl.propagator.propagators2D.integral import Integral2D"
                text_code +=   "\nfrom wofryimpl.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D"

            if do_plot:
                text_code += "\n\nfrom srxraylib.plot.gol import plot, plot_image"
                text_code += "\nplot_from_oe = 0 # set to a large number to avoid plots"


            text_code  +=  "\n\n\n##########  SOURCE ##########\n\n\n"
            text_code += self.get_light_source().to_python_code(do_plot=False, add_import_section=False)


            if do_plot:
                if source_dimension == 1:
                    text_code += "\n\n\nif plot_from_oe <= 0: plot(output_wavefront.get_abscissas(),output_wavefront.get_intensity(),title='SOURCE')"
                elif source_dimension == 2:
                    text_code += "\n\n\nif plot_from_oe <= 0: plot_image(output_wavefront.get_intensity(),output_wavefront.get_coordinate_x(),output_wavefront.get_coordinate_y(),aspect='auto',title='SOURCE')"


        if self.get_beamline_elements_number() > 0:
            text_code += "\n\n\n##########  OPTICAL SYSTEM ##########\n\n\n"


            for index in range(self.get_beamline_elements_number()):
                text_code += "\n\n\n##########  OPTICAL ELEMENT NUMBER %i ##########\n\n\n" % (index+1)
                oe_name = "oe_" + str(index)
                beamline_element = self.get_beamline_element_at(index)
                optical_element = beamline_element.get_optical_element()
                coordinates = beamline_element.get_coordinates()

                text_code += "\ninput_wavefront = output_wavefront.duplicate()"

                # OPTICAL ELEMENT ----------------
                text_code += optical_element.to_python_code()

                propagation_info = self.get_propagation_info_at(index)

                if (coordinates.p() == 0.0) and (coordinates.q() == 0.0): # NO DRIFT
                    text_code += "\n# no drift in this element "
                    text_code += "\noutput_wavefront = optical_element.applyOpticalElement(input_wavefront)"
                else:
                    if coordinates.p() != 0.0:
                        text_code += "\n# drift_before %g m" % coordinates.p()
                    if coordinates.q() != 0.0:
                        text_code += "\n# drift_after %g m " % coordinates.q()

                    ##########################
                    # 1D
                    # ==
                    #
                    # propagators_list = ["Fresnel",    "Fresnel (Convolution)",  "Fraunhofer",    "Integral",    "Fresnel Zoom",    "Fresnel Zoom Scaled"]
                    # class_name       = ["Fresnel1D",  "FresnelConvolution1D",   "Fraunhofer1D",  "Integral1D",  "FresnelZoom1D",   "FresnelZoomScaling1D"]
                    # handler_name     = ["FRESNEL_1D", "FRESNEL_CONVOLUTION_1D", "FRAUNHOFER_1D", "INTEGRAL_1D", "FRESNEL_ZOOM_1D", "FRESNEL_ZOOM_SCALING_1D"]
                    #
                    # 2D
                    # ==
                    # propagators_list = ["Fresnel",   "Fresnel (Convolution)",  "Fraunhofer",    "Integral",    "Fresnel Zoom XY"   ]
                    # class_name       = ["Fresnel2D", "FresnelConvolution2D",   "Fraunhofer2D",  "Integral2D",  "FresnelZoomXY2D"   ]
                    # handler_name     = ["FRESNEL_2D","FRESNEL_CONVOLUTION_2D", "FRAUNHOFER_2D", "INTEGRAL_2D", "FRESNEL_ZOOM_XY_2D"]

                    propagator_class_name                   = propagation_info["propagator_class_name"]
                    propagator_handler_name                 = propagation_info["propagator_handler_name"]
                    propagator_additional_parameters_names  = propagation_info["propagator_additional_parameters_names"]
                    propagator_additional_parameters_values = propagation_info["propagator_additional_parameters_values"]

                    text_code += "\n#"
                    text_code += "\n# propagating\n#"
                    text_code += "\n#"
                    text_code += "\npropagation_elements = PropagationElements()"
                    text_code += "\nbeamline_element = BeamlineElement(optical_element=optical_element,"
                    text_code += "    coordinates=ElementCoordinates(p=%f," % (coordinates.p())
                    text_code += "    q=%f," % (coordinates.q())
                    text_code += "    angle_radial=numpy.radians(%f)," % (coordinates.angle_radial())
                    text_code += "    angle_azimuthal=numpy.radians(%f)))" % (coordinates.angle_azimuthal())
                    text_code += "\npropagation_elements.add_beamline_element(beamline_element)"
                    text_code += "\npropagation_parameters = PropagationParameters(wavefront=input_wavefront,"
                    text_code += "    propagation_elements = propagation_elements)"
                    text_code += "\n#self.set_additional_parameters(propagation_parameters)"
                    text_code += "\n#"

                    for i in range(len(propagator_additional_parameters_names)):
                        text_code += "\npropagation_parameters.set_additional_parameters('%s', %s)" % \
                        (propagator_additional_parameters_names[i], str(propagator_additional_parameters_values[i]))

                    text_code += "\n#"
                    text_code += "\npropagator = PropagationManager.Instance()"
                    text_code += "\ntry:"
                    text_code += "\n    propagator.add_propagator(%s())" % propagator_class_name
                    text_code += "\nexcept:"
                    text_code += "\n    pass"
                    text_code += "\noutput_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,"
                    text_code += "    handler_name='%s')" % (propagator_handler_name)

                if do_plot:
                    text_code += "\n\n\n#\n#---- plots -----\n#"
                    if source_dimension == 1:
                        text_code += "\nif plot_from_oe <= %d: plot(output_wavefront.get_abscissas(),output_wavefront.get_intensity(),title='OPTICAL ELEMENT NR %d')" % (index+1, index+1)
                    elif source_dimension == 2:
                        text_code += "\nif plot_from_oe <= %d: plot_image(output_wavefront.get_intensity(),output_wavefront.get_coordinate_x(),output_wavefront.get_coordinate_y(),aspect='auto',title='OPTICAL ELEMENT NR %d')" % (index+1, index+1)

        return text_code

    # TODO move to syned beamline
    def get_beamline_elements_names(self):
        els = self.get_beamline_elements()
        NAMES = []
        for i, eli in enumerate(els):
            NAMES.append(eli.get_optical_element().get_name())
        return NAMES

    def get_p_and_q(self):
        els = self.get_beamline_elements()
        P = []
        Q = []
        for i, eli in enumerate(els):
            P.append(eli.get_coordinates().p())
            Q.append(eli.get_coordinates().q())
        return P, Q

    def distances(self,source_at=0.0):
        txt = "%5s %15s %10s %10s %10s %10s\n" % ("O.E.#","O.E. Name","p [m]", "q [m]", "src-oe", "src-screen")
        NAMES = self.get_beamline_elements_names()
        P,Q = self.get_p_and_q()
        oe_accumulated = source_at
        image_accumulated = source_at
        for i in range(len(NAMES)):
            oe_accumulated = image_accumulated + P[i]
            image_accumulated += P[i] + Q[i]
            txt += "%5d %15s %10g %10g %10g %10g\n" % (i+1,NAMES[i], P[i], Q[i], oe_accumulated, image_accumulated)
        return txt