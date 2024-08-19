# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config

# dataset loading

from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)


# models
from .maskformer_model import MaskFormer
from .weighted_maskformer_model import WeightedMaskFormer
from .test_time_augmentation import SemanticSegmentorWithTTA
from .groupformer_model import GroupFormer

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator