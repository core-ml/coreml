"""Defines Factory object to register dimensionality reduction methods"""
from typing import Any
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from coreml.factory import Factory

factory = Factory()
factory.register_builder('PCA', PCA)
factory.register_builder('TSNE', TSNE)
