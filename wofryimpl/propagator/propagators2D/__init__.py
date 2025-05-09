
from wofryimpl.propagator.propagators2D.fraunhofer import Fraunhofer2D
from wofryimpl.propagator.propagators2D.fresnel import Fresnel2D
from wofryimpl.propagator.propagators2D.fresnel_convolution import FresnelConvolution2D
from wofryimpl.propagator.propagators2D.integral import Integral2D
from wofryimpl.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D
from wofry.propagator.propagator import PropagationManager

def initialize_default_propagator_2D():
    propagator = PropagationManager.Instance()

    try: propagator.add_propagator(Fraunhofer2D())
    except Exception as e: print(e)
    try: propagator.add_propagator(Fresnel2D())
    except Exception as e: print(e)
    try: propagator.add_propagator(FresnelConvolution2D())
    except Exception as e: print(e)
    try: propagator.add_propagator(Integral2D())
    except Exception as e: print(e)
    try: propagator.add_propagator(FresnelZoomXY2D())
    except Exception as e: print(e)
