"""
QRT Leukemia Data Challenge - ML Pipeline
=========================================

Package structure:
- config: Constants and configuration
- data_loader: Data loading and validation
- features: Feature engineering (clinical + molecular)
- preprocessing: sklearn pipelines for data transformation
- models: Survival models and baselines
- evaluation: IPCW C-index and cross-validation
- optimization: Numba-accelerated computations
"""

from . import config
from . import data_loader
from . import features
from . import preprocessing
from . import models
from . import evaluation
from . import optimization

__version__ = "1.0.0"
__author__ = "QRT Leukemia Challenge Team"
