from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional
from abc import ABCMeta


class MetaModel(metaclass=ABCMeta):
    """The necessary functions for the base model.
    This class defines the necessary functions for the base model, in the base model, we have implemented them.
    """

    @abstractmethod
    def load_models(self, model_cfg):
        """Based on the given data_cfg, we get the data loader."""
        raise NotImplementedError

    @abstractmethod
    def inputs_pretreatment(self, inputs):
        """Transform the input data based on transform setting."""
        raise NotImplementedError

    @abstractmethod
    def inference(self, inputs):
        """Do inference (calculate features.)."""
        raise NotImplementedError
