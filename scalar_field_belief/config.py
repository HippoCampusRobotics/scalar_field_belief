from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class BeliefConfig:
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
