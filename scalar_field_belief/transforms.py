from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class TargetStandardizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, y: torch.Tensor, eps: float = 1e-12) -> "TargetStandardizer":
        if y.ndim != 1:
            raise ValueError(f"Expected y to have shape (N,), got {tuple(y.shape)}")
        if not y.is_floating_point():
            raise TypeError("Expected floating-point tensor for y")

        mean = y.mean()
        std = y.std(correction=0)
        std = torch.where(std < eps, torch.ones_like(std), std)
        return cls(mean=mean.detach(), std=std.detach())

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / self.std

    def inverse_transform_mean_var(
        self, mean_norm: torch.Tensor, var_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = mean_norm * self.std + self.mean
        var = var_norm * (self.std**2)
        return mean, var


@dataclass(frozen=True)
class InputNormalizer2d:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min.")
        if self.y_max <= self.y_min:
            raise ValueError("y_max must be greater than y_min.")

    def transform(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"Expected xy to have shape (N, 2), got {tuple(xy.shape)}")
        xy_norm = np.empty_like(xy, dtype=float)
        xy_norm[:, 0] = (xy[:, 0] - self.x_min) / (self.x_max - self.x_min)
        xy_norm[:, 1] = (xy[:, 1] - self.y_min) / (self.y_max - self.y_min)
        return xy_norm
