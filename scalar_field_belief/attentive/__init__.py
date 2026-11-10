# Copyright (C) 2026 Nathalie Bauschmann
# SPDX-License-Identifier: GPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Exact GP belief with the Attentive Kernel (AK).

Attentive Kernel after Chen, Khardon and Liu, "AK: Attentive Kernel for
Information Gathering", RSS 2022 (doi:10.15607/RSS.2022.XVIII.047), with a
shared attention network over normalized positions, fixed primitive RBF
lengthscales and a learned amplitude, sensor-noise variance and constant
mean (see PROVENANCE.md). Needs NumPy, PyTorch and GPyTorch (no ROS, no GUI,
no geometry).
"""

from scalar_field_belief.attentive.belief import (
    VARIANCE_ROUNDOFF_TOLERANCE,
    AttentiveGPBelief,
    AttentiveGPError,
    MarginalPrediction,
    exact_gpytorch_settings,
    torch_threads,
)
from scalar_field_belief.attentive.config import AttentiveGPConfig
from scalar_field_belief.attentive.kernel import (
    AttentiveKernel,
    TwoHiddenLayerTanhNN,
)

__all__ = [
    'VARIANCE_ROUNDOFF_TOLERANCE',
    'AttentiveGPBelief',
    'AttentiveGPConfig',
    'AttentiveGPError',
    'AttentiveKernel',
    'MarginalPrediction',
    'TwoHiddenLayerTanhNN',
    'exact_gpytorch_settings',
    'torch_threads',
]
