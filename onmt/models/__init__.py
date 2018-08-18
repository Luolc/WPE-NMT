"""Module defining models."""
from onmt.models.model_saver import build_model_saver, ModelSaver
from onmt.models.model import NMTModel, WPEModel

__all__ = ["build_model_saver", "ModelSaver",
           "NMTModel", 'WPEModel', "check_sru_requirement"]
