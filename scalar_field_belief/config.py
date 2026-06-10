"""Configuration object for the scalar field belief node.

The `BeliefConfig` dataclass collects all parameters that affect the GP belief
model, input normalization, refitting behavior, and visualization grid.

The ROS node is responsible for reading parameters from ROS and constructing a
`BeliefConfig`. The core belief code then uses this plain Python object without
depending on ROS parameter APIs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class BeliefConfig:
    """Configuration for the scalar field GP belief.

    Parameters
    ----------
    frame_id
        ROS frame in which measurements and belief queries are interpreted.

    x_min, x_max, y_min, y_max
        Physical domain bounds used to normalize `(x, y)` positions before they
        are passed to the GP.

    kernel_type
        Spatial GP kernel type. Supported values are `'rbf'`, `'matern32'`, and
        `'matern52'`.

    training_iter
        Number of optimizer iterations used whenever the exact GP is refitted.

    learning_rate
        Adam optimizer learning rate used during GP fitting.

    init_lengthscale_x, init_lengthscale_y
        Initial ARD kernel lengthscales in normalized input units.

    init_outputscale
        Initial kernel outputscale in standardized target units.

    init_noise
        Initial Gaussian likelihood noise in standardized target units.

    refit_policy
        Policy that decides when the GP is refitted. Supported values are
        `'every_measurement'` and `'every_k_measurements'`.

    refit_every_k
        Number of new measurements required before refitting when
        `refit_policy == 'every_k_measurements'`.

    publish_visualization
        Whether the node should publish belief visualization point clouds.

    visualization_grid_step
        Physical grid spacing used for visualization queries.

    visualization_z_mode
        Visualization z mode. Supported values are `'flat'` and `'height'`.

    visualization_height_scale
        Height scale used when `visualization_z_mode == 'height'`.

    device
        Torch device used for GP tensors and model parameters.

    dtype
        Torch floating-point dtype used for GP tensors.

    Notes
    -----
    The dataclass is frozen so that the belief object sees a stable
    configuration after construction. Parameters that should change at runtime
    are handled separately by the ROS node.
    """

    frame_id: str

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    kernel_type: str
    training_iter: int
    learning_rate: float

    init_lengthscale_x: float
    init_lengthscale_y: float
    init_outputscale: float
    init_noise: float

    refit_policy: str
    refit_every_k: int

    publish_visualization: bool
    visualization_grid_step: float
    visualization_z_mode: str
    visualization_height_scale: float
    device: str = 'cpu'
    dtype: torch.dtype = torch.float64

    @property
    def x_range(self) -> tuple[float, float]:
        return (self.x_min, self.x_max)

    @property
    def y_range(self) -> tuple[float, float]:
        return (self.y_min, self.y_max)

    def validate(self) -> None:
        if not self.frame_id:
            raise ValueError('frame_id must not be empty.')
        if self.x_max <= self.x_min:
            raise ValueError('x_max must be greater than x_min.')
        if self.y_max <= self.y_min:
            raise ValueError('y_max must be greater than y_min.')

        if self.kernel_type not in {'rbf', 'matern32', 'matern52'}:
            raise ValueError(
                f"Unsupported kernel_type '{self.kernel_type}'. "
                "Expected one of {'rbf', 'matern32', 'matern52'}."
            )

        if self.training_iter <= 0:
            raise ValueError('training_iter must be positive.')
        if self.learning_rate <= 0.0:
            raise ValueError('learning_rate must be positive.')

        if self.init_lengthscale_x <= 0.0:
            raise ValueError('init_lengthscale_x must be positive.')
        if self.init_lengthscale_y <= 0.0:
            raise ValueError('init_lengthscale_y must be positive.')
        if self.init_outputscale <= 0.0:
            raise ValueError('init_outputscale must be positive.')
        if self.init_noise <= 0.0:
            raise ValueError('init_noise must be positive.')

        if self.refit_policy not in {
            'every_measurement',
            'every_k_measurements',
        }:
            raise ValueError(
                f"Unsupported refit_policy '{self.refit_policy}'. "
                "Expected 'every_measurement' or 'every_k_measurements'."
            )
        if self.refit_every_k <= 0:
            raise ValueError('refit_every_k must be positive.')

        if self.visualization_grid_step <= 0.0:
            raise ValueError('visualization_grid_step must be positive.')
        if self.visualization_z_mode not in {'flat', 'height'}:
            raise ValueError(
                f"Unsupported visualization_z_mode '{self.visualization_z_mode}'. "
                "Expected 'flat' or 'height'."
            )
        if self.visualization_height_scale <= 0.0:
            raise ValueError('visualization_height_scale must be positive.')
