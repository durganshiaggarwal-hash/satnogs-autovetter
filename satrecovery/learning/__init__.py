"""Learned components: features, datasets, training, and the models.

Phase 4 adds the AI half of the comparison. Everything here follows the
same two rules as the rest of the project: a model must satisfy an
interface the receiver already understands, and it must be gradable
against the classical baseline on identical frames.
"""

from .base import TrainableModel, TrainingHistory
from .datasets import (
    ClassificationDataset,
    DatasetBuilder,
    DemappingDataset,
    EndToEndDataset,
    generate_frames,
    split_rows,
    training_conditions,
)
from .features import (
    CombinedFeatures,
    CumulantFeatures,
    FeatureExtractor,
    SpectralFeatures,
    default_extractor,
)
from .forest import ForestModulationClassifier
from .neural import CNNDemodulator, MLPDemapper
from .trainer import (
    EarlyStopping,
    ProgressPrinter,
    Trainer,
    TrainingCallback,
    pick_device,
)

__all__ = [
    "CNNDemodulator",
    "ClassificationDataset",
    "CombinedFeatures",
    "CumulantFeatures",
    "DatasetBuilder",
    "DemappingDataset",
    "EarlyStopping",
    "EndToEndDataset",
    "FeatureExtractor",
    "ForestModulationClassifier",
    "MLPDemapper",
    "ProgressPrinter",
    "SpectralFeatures",
    "TrainableModel",
    "Trainer",
    "TrainingCallback",
    "TrainingHistory",
    "default_extractor",
    "generate_frames",
    "pick_device",
    "split_rows",
    "training_conditions",
]
