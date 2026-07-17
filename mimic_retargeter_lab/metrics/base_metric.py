from abc import ABC, abstractmethod


class BaseMetric(ABC):
    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def compute(self):
        pass

    def generate_report(self):
        pass
