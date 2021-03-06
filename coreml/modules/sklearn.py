"""Defines Factory object to register various sklearn methods"""
from typing import Any
from sklearn.svm import SVC as SVM
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from coreml.factory import Factory

factory = Factory()
factory.register_builder('SVM', SVM)
factory.register_builder(
    'GradientBoostingClassifier', GradientBoostingClassifier)
factory.register_builder('XGBClassifier', XGBClassifier)
