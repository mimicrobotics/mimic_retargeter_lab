from abc import abstractmethod

from mimic_retargeter_lab.interfaces.objects.base_object import BaseObject


class MujocoBaseObject(BaseObject):
    def __init__(self):
        pass

    @abstractmethod
    def get_xml(self) -> str:
        """
        Return an MJCF/XML fragment (or full MJCF string) that represents this
        object for visualization in MuJoCo. Concrete subclasses should return
        a string.
        """
        raise NotImplementedError
