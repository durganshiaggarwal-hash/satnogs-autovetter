"""The application's tabs. Each is a :class:`Page`."""

from .base import Page
from .benchmark import BenchmarkPage
from .experiment import ExperimentPage
from .models import ModelsPage
from .stages import StagesPage

__all__ = ["BenchmarkPage", "ExperimentPage", "ModelsPage", "Page", "StagesPage"]
