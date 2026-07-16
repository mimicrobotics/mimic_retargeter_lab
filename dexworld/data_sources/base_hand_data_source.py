from abc import ABC, abstractmethod


class BaseHandDataSource(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_iter(self):
        pass
