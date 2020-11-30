from importlib import import_module
from pyFAI.detectors import Detector


def grab(name, *args, **kwargs):

    if isinstance(name, Detector):
        return name

    try:
        if "." in name:
            module_name, class_name = name.rsplit(".", 1)
        else:
            module_name = name
            class_name = name.capitalize()

        detector_module = import_module("." + module_name, package="Xana.detectors")

        detector_class = getattr(detector_module, class_name)

        instance = detector_class(*args, **kwargs)

    except (AttributeError, ModuleNotFoundError):
        raise ImportError("{} is not part of our detector collection!".format(name))
    else:
        if not issubclass(detector_class, Detector):
            raise ImportError(
                "We currently don't have {}, but you are welcome to send in the request for it!".format(
                    detector_class
                )
            )

    return instance
