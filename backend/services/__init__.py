"""__init__ for services package.

``training_service`` is intentionally not imported here: it pulls in torch,
transformers and torch_geometric at module level. Import it directly where
training is needed so the API can start without those dependencies loaded.
"""
from . import dataset_service
from . import features
from . import external_datasets

__all__ = ["dataset_service", "features", "external_datasets"]
