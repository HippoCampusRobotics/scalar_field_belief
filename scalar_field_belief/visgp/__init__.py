# Visibility-graph GP (visGP) after Gilbert and Datta (2024), Biometrics:
# https://academic.oup.com/biometrics/article/doi/10.1093/biomtc/ujae089/7753413
# Authors' R implementation: https://github.com/bjg345/visGP

"""Visibility-graph GP (visGP) belief for domains with known straight walls.

Dense reference implementation after Gilbert and Datta (2024): a fixed
reference set, covariance selection on its visibility graph, the
nearest-clique process extension to arbitrary positions, and independent
Gaussian sensor noise. Needs NumPy and SciPy only (no ROS, no PyTorch).
"""

from scalar_field_belief.visgp.belief import MarginalPrediction, VisGPBelief
from scalar_field_belief.visgp.config import VisGPConfig
from scalar_field_belief.visgp.covariance import (
    CovarianceSelectionError,
    CovarianceSelectionResult,
    covariance_selection,
    matern32,
)
from scalar_field_belief.visgp.extension import (
    ExtensionRows,
    ProcessExtension,
    location_key,
)

__all__ = [
    'CovarianceSelectionError',
    'CovarianceSelectionResult',
    'ExtensionRows',
    'MarginalPrediction',
    'ProcessExtension',
    'VisGPBelief',
    'VisGPConfig',
    'covariance_selection',
    'location_key',
    'matern32',
]
