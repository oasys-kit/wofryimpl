API Reference
=============

Beamline
--------

.. autosummary::
   :toctree: api

   wofryimpl.beamline.beamline

Optical Elements — Absorbers
-----------------------------

.. autosummary::
   :toctree: api

   wofryimpl.beamline.optical_elements.absorbers.slit
   wofryimpl.beamline.optical_elements.absorbers.beam_stopper

Optical Elements — Ideal Elements
-----------------------------------

.. autosummary::
   :toctree: api

   wofryimpl.beamline.optical_elements.ideal_elements.ideal_lens
   wofryimpl.beamline.optical_elements.ideal_elements.screen

Optical Elements — Mirrors
---------------------------

.. autosummary::
   :toctree: api

   wofryimpl.beamline.optical_elements.mirrors.mirror

Optical Elements — Refractors
-------------------------------

.. autosummary::
   :toctree: api

   wofryimpl.beamline.optical_elements.refractors.lens
   wofryimpl.beamline.optical_elements.refractors.thin_object
   wofryimpl.beamline.optical_elements.refractors.thin_object_corrector

Optical Elements — Utilities
------------------------------

.. autosummary::
   :toctree: api

   wofryimpl.beamline.optical_elements.util.s4_optical_surface
   wofryimpl.beamline.optical_elements.util.s4_conic
   wofryimpl.beamline.optical_elements.util.arrayofvectors

Light Sources
-------------

.. autosummary::
   :toctree: api

   wofryimpl.propagator.light_source
   wofryimpl.propagator.light_source_h5file
   wofryimpl.propagator.light_source_cmd
   wofryimpl.propagator.light_source_pysru

Propagators 1D
--------------

.. autosummary::
   :toctree: api

   wofryimpl.propagator.propagators1D.fraunhofer
   wofryimpl.propagator.propagators1D.fresnel
   wofryimpl.propagator.propagators1D.fresnel_convolution
   wofryimpl.propagator.propagators1D.fresnel_zoom
   wofryimpl.propagator.propagators1D.fresnel_zoom_scaling_theorem
   wofryimpl.propagator.propagators1D.integral

Propagators 2D
--------------

.. autosummary::
   :toctree: api

   wofryimpl.propagator.propagators2D.fraunhofer
   wofryimpl.propagator.propagators2D.fresnel
   wofryimpl.propagator.propagators2D.fresnel_convolution
   wofryimpl.propagator.propagators2D.fresnel_zoom_xy
   wofryimpl.propagator.propagators2D.integral

Propagator Utilities
--------------------

.. autosummary::
   :toctree: api

   wofryimpl.propagator.util.tally
   wofryimpl.propagator.util.undulator_coherent_mode_decomposition_1d
