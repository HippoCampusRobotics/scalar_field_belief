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

"""Tests of the GP belief core, baseline and wall-aware (no ROS needed).

Run from the package root:  python3 -m pytest test/test_wall_aware_gp.py
"""

import dataclasses

import gpytorch
import numpy as np
import pytest
import torch

from scalar_field_belief.belief import ScalarFieldBelief
from scalar_field_belief.config import BeliefConfig
from scalar_field_belief.geometry import (
    WallGeometry,
    WallSegment,
    load_wall_geometry_from_toml,
)
from scalar_field_belief.models import (
    build_covar_module,
    get_spatial_kernel,
    get_wall_kernel,
)
from scalar_field_belief.transforms import InputNormalizer2d, build_gp_inputs
from scalar_field_belief.wall_features import WallFeatureMap

# Domain 2 m x 4 m with one vertical wall and one horizontal wall.
GEOMETRY = WallGeometry(
    name='test',
    x_range=(0.0, 2.0),
    y_range=(0.0, 4.0),
    walls=(
        WallSegment(start=(1.0, 0.6), end=(1.0, 3.4)),
        WallSegment(start=(1.0, 2.1), end=(1.5, 2.1)),
    ),
)


def make_config(**overrides) -> BeliefConfig:
    kwargs = {
        'frame_id': 'map',
        'x_min': 0.0,
        'x_max': 2.0,
        'y_min': 0.0,
        'y_max': 4.0,
        'kernel_type': 'rbf',
        'training_iter': 10,
        'learning_rate': 0.1,
        'init_lengthscale_x': 0.2,
        'init_lengthscale_y': 0.1,
        'init_outputscale': 1.0,
        'init_noise': 0.01,
        'refit_policy': 'every_measurement',
        'refit_every_k': 1,
        'optimize_hyperparameters': False,
        'publish_visualization': False,
        'visualization_grid_step': 0.1,
        'visualization_z_mode': 'flat',
        'visualization_height_scale': 0.5,
    }
    kwargs.update(overrides)
    return BeliefConfig(**kwargs)


def measurements(n: int = 12, seed: int = 0):
    """Random positions and a field that jumps across the wall at x = 1 m."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform((0.05, 0.05), (1.95, 3.95), size=(n, 2))
    values = 0.002 * (xy[:, 0] < 1.0) + 1e-4 * rng.standard_normal(n)
    return xy, values


def fitted_belief(variant: str, optimize: bool = False) -> ScalarFieldBelief:
    torch.manual_seed(0)
    config = make_config(
        model_variant=variant, optimize_hyperparameters=optimize
    )
    belief = ScalarFieldBelief(config, wall_geometry=GEOMETRY)
    for (x, y), value in zip(*measurements(), strict=True):
        belief.add_measurement(x, y, value)
    return belief


# ---------------------------------------------------------------- geometry


GEOMETRY_TOML = """
[general]
name = "two_walls"

[bounds]
x_min = 0.0
x_max = 2.0
y_min = 0.0
y_max = 4.0

[objects.walls]
wall_vertices = [[1.0, 0.6, 1.0, 3.4], [1.0, 2.1, 1.5, 2.1]]
"""

SELF_CONTAINED_SCENARIO_TOML = """
[general]
name = "inline_scenario"

[bounds]
x_min = 0.0
x_max = 1.0
y_min = 0.0
y_max = 2.0

[objects.walls]
wall_vertices = [[0.5, 0.2, 0.5, 1.8]]

[objects.sources]
source_positions = [[0.2, 1.0]]
"""


def test_load_wall_geometry_file(tmp_path):
    path = tmp_path / 'geometry.toml'
    path.write_text(GEOMETRY_TOML)
    geometry = load_wall_geometry_from_toml(path)
    assert geometry.name == 'two_walls'
    assert geometry.x_range == (0.0, 2.0)
    assert geometry.y_range == (0.0, 4.0)
    assert len(geometry.walls) == 2
    assert geometry.walls[1].start == (1.0, 2.1)
    assert geometry.walls[1].end == (1.5, 2.1)


def test_load_scenario_with_geometry_reference(tmp_path):
    (tmp_path / 'geometry').mkdir()
    (tmp_path / 'geometry' / 'tank.toml').write_text(GEOMETRY_TOML)
    scenario = tmp_path / 'scenario.toml'
    # field_file and [general] of the scenario are ignored.
    scenario.write_text(
        'geometry_file = "geometry/tank.toml"\n'
        'field_file = "does_not_exist.toml"\n'
        '[general]\nname = "scenario"\n'
    )
    geometry = load_wall_geometry_from_toml(scenario)
    assert geometry.name == 'two_walls'
    assert geometry.x_range == (0.0, 2.0)
    assert geometry.y_range == (0.0, 4.0)
    assert len(geometry.walls) == 2


def test_load_self_contained_scenario(tmp_path):
    path = tmp_path / 'scenario.toml'
    path.write_text(SELF_CONTAINED_SCENARIO_TOML)
    geometry = load_wall_geometry_from_toml(path)
    assert geometry.name == 'inline_scenario'
    assert geometry.x_range == (0.0, 1.0)
    assert geometry.y_range == (0.0, 2.0)
    assert [(w.start, w.end) for w in geometry.walls] == [
        ((0.5, 0.2), (0.5, 1.8))
    ]


def test_nested_geometry_reference_rejected(tmp_path):
    (tmp_path / 'inner.toml').write_text(GEOMETRY_TOML)
    (tmp_path / 'outer.toml').write_text('geometry_file = "inner.toml"\n')
    scenario = tmp_path / 'scenario.toml'
    scenario.write_text('geometry_file = "outer.toml"\n')
    with pytest.raises(ValueError, match='geometry_file'):
        load_wall_geometry_from_toml(scenario)


def test_field_file_without_geometry_file_rejected(tmp_path):
    scenario = tmp_path / 'scenario.toml'
    scenario.write_text(
        'field_file = "field.toml"\n' + SELF_CONTAINED_SCENARIO_TOML
    )
    with pytest.raises(ValueError, match='geometry_file'):
        load_wall_geometry_from_toml(scenario)


def test_missing_referenced_geometry_file(tmp_path):
    scenario = tmp_path / 'scenario.toml'
    scenario.write_text('geometry_file = "missing.toml"\n')
    with pytest.raises(FileNotFoundError, match='missing.toml'):
        load_wall_geometry_from_toml(scenario)


@pytest.mark.parametrize(
    'text',
    [
        # missing [bounds]
        '[objects.walls]\nwall_vertices = [[0, 0, 1, 1]]\n',
        # wall with three numbers
        '[bounds]\nx_min = 0\nx_max = 1\ny_min = 0\ny_max = 1\n'
        '[objects.walls]\nwall_vertices = [[0, 0, 1]]\n',
        # zero-length wall
        '[bounds]\nx_min = 0\nx_max = 1\ny_min = 0\ny_max = 1\n'
        '[objects.walls]\nwall_vertices = [[0.5, 0.5, 0.5, 0.5]]\n',
    ],
    ids=['missing bounds', 'short wall', 'zero-length wall'],
)
def test_invalid_wall_file_rejected(tmp_path, text):
    path = tmp_path / 'walls.toml'
    path.write_text(text)
    with pytest.raises(ValueError):
        load_wall_geometry_from_toml(path)


# ----------------------------------------------------------- wall features


def expected_features(xy, geometry, beta, rho):
    """Wall feature formula, written out per point and wall."""
    out = np.zeros((len(xy), len(geometry.walls)))
    for i, p in enumerate(np.asarray(xy, dtype=float)):
        for m, wall in enumerate(geometry.walls):
            a, b = np.array(wall.start), np.array(wall.end)
            t = (b - a) / np.linalg.norm(b - a)
            n = np.array([-t[1], t[0]])
            s = np.dot(p - a, n)
            u = np.clip(np.dot(p - a, t), 0.0, np.linalg.norm(b - a))
            d = np.linalg.norm(p - (a + u * t))
            out[i, m] = np.tanh(beta * s) * np.exp(-(d**2) / (2 * rho**2))
    return out


def test_wall_features_match_formula():
    xy = np.random.default_rng(1).uniform((0, 0), (2, 4), size=(50, 2))
    fmap = WallFeatureMap(GEOMETRY, beta=1.0, rho=0.35)
    features = fmap.transform_xy_to_wall_features(xy)
    assert features.shape == (50, 2)
    np.testing.assert_allclose(
        features, expected_features(xy, GEOMETRY, 1.0, 0.35), atol=1e-12
    )


def test_wall_features_properties():
    fmap = WallFeatureMap(GEOMETRY, beta=2.0, rho=0.35)
    # Mirror images across the vertical wall (x = 1) have opposite features.
    left, right = [[0.8, 1.0]], [[1.2, 1.0]]
    h_left = fmap.transform_xy_to_wall_features(left)[0, 0]
    h_right = fmap.transform_xy_to_wall_features(right)[0, 0]
    assert h_left == pytest.approx(-h_right)
    assert h_left != 0.0
    # Zero on the wall line, close to zero far away from the wall.
    on_wall = fmap.transform_xy_to_wall_features([[1.0, 1.0]])[0, 0]
    assert on_wall == pytest.approx(0.0, abs=1e-12)
    far = fmap.transform_xy_to_wall_features([[0.0, 1.0]])[0, 0]
    assert abs(far) < 0.05
    # Bounded by 1.
    xy = np.random.default_rng(2).uniform((0, 0), (2, 4), size=(100, 2))
    assert np.all(np.abs(fmap.transform_xy_to_wall_features(xy)) <= 1.0)


@pytest.mark.parametrize('beta, rho', [(0.0, 0.35), (1.0, -1.0), (np.nan, 1)])
def test_wall_features_reject_invalid_parameters(beta, rho):
    with pytest.raises(ValueError):
        WallFeatureMap(GEOMETRY, beta=beta, rho=rho)


# -------------------------------------------------------------- GP inputs


def test_gp_inputs_per_variant():
    xy = np.array([[0.5, 1.0], [1.5, 3.0]])
    normalizer = InputNormalizer2d(x_min=0.0, x_max=2.0, y_min=0.0, y_max=4.0)
    fmap = WallFeatureMap(GEOMETRY, beta=1.0, rho=0.35)

    baseline = build_gp_inputs(xy, normalizer).to_model_array('baseline')
    np.testing.assert_allclose(baseline, [[0.25, 0.25], [0.75, 0.75]])

    inputs = build_gp_inputs(xy, normalizer, fmap)
    augmented = inputs.to_model_array('wall_aware')
    assert augmented.shape == (2, 4)
    np.testing.assert_allclose(augmented[:, :2], baseline)
    np.testing.assert_allclose(
        augmented[:, 2:], fmap.transform_xy_to_wall_features(xy)
    )

    with pytest.raises(ValueError):  # wall-aware needs wall features
        build_gp_inputs(xy, normalizer).to_model_array('wall_aware')


# ---------------------------------------------------------------- kernels


@pytest.mark.parametrize('kernel_type', ['rbf', 'matern32', 'matern52'])
def test_covar_module_structure(kernel_type):
    baseline = build_covar_module(kernel_type)
    assert get_wall_kernel(baseline) is None
    assert get_spatial_kernel(baseline).ard_num_dims == 2

    wall_aware = build_covar_module(
        kernel_type, model_variant='wall_aware', wall_feature_dim=3
    )
    spatial = get_spatial_kernel(wall_aware)
    wall = get_wall_kernel(wall_aware)
    assert isinstance(wall_aware.base_kernel, gpytorch.kernels.ProductKernel)
    assert spatial.active_dims.tolist() == [0, 1]
    assert wall.active_dims.tolist() == [2, 3, 4]
    assert wall.ard_num_dims == 3


@pytest.mark.parametrize(
    'variant, dim',
    [('baseline', 1), ('wall_aware', 0), ('unknown', 0)],
)
def test_covar_module_rejects_invalid_arguments(variant, dim):
    with pytest.raises(ValueError):
        build_covar_module('rbf', model_variant=variant, wall_feature_dim=dim)


# ------------------------------------------------------------------ config


def test_config_defaults_to_baseline():
    assert make_config().model_variant == 'baseline'


def test_config_wall_settings_are_keyword_only():
    positional = dataclasses.astuple(make_config())
    with pytest.raises(TypeError):
        BeliefConfig(*positional[:21], 'wall_aware')


@pytest.mark.parametrize(
    'overrides',
    [
        {'model_variant': 'walls'},
        {'wall_beta': 0.0},
        {'wall_rho': -0.1},
        {'init_wall_lengthscale': float('inf')},
    ],
)
def test_config_rejects_invalid_wall_settings(overrides):
    with pytest.raises(ValueError):
        make_config(**overrides).validate()


# ------------------------------------------------------- belief: geometry


def test_wall_aware_belief_requires_geometry():
    config = make_config(model_variant='wall_aware')
    with pytest.raises(ValueError):
        ScalarFieldBelief(config)
    with pytest.raises(TypeError):
        ScalarFieldBelief(config, wall_geometry='walls.toml')
    no_walls = dataclasses.replace(GEOMETRY, walls=())
    with pytest.raises(ValueError):
        ScalarFieldBelief(config, wall_geometry=no_walls)
    other_bounds = dataclasses.replace(GEOMETRY, x_range=(0.0, 3.0))
    with pytest.raises(ValueError):
        ScalarFieldBelief(config, wall_geometry=other_bounds)


def test_baseline_belief_ignores_geometry():
    belief = ScalarFieldBelief(make_config(), wall_geometry=GEOMETRY)
    assert belief.wall_feature_dim == 0
    assert belief.wall_feature_map is None


def test_reset_keeps_wall_geometry():
    belief = fitted_belief('wall_aware')
    belief.reset()
    assert not belief.has_model()
    assert belief.wall_feature_dim == 2


# ------------------------------------------------- belief: posterior math


def oracle_posterior(belief: ScalarFieldBelief, xq: np.ndarray):
    """Exact GP posterior in NumPy, using the belief's hyperparameters."""
    model, likelihood = belief.model, belief.likelihood
    outputscale = model.covar_module.outputscale.item()
    spatial_ls = get_spatial_kernel(model.covar_module).lengthscale
    spatial_ls = spatial_ls.detach().numpy().ravel()
    wall_kernel = get_wall_kernel(model.covar_module)
    noise = likelihood.noise.item()
    const = model.mean_module.constant.item()

    def inputs(xy):
        xn = (xy - [0.0, 0.0]) / [2.0, 4.0]
        if wall_kernel is None:
            return xn, None
        return xn, belief.wall_feature_map.transform_xy_to_wall_features(xy)

    def kernel(p, q):
        (xp, hp), (xq_, hq) = p, q
        d2 = (((xp[:, None] - xq_[None]) / spatial_ls) ** 2).sum(-1)
        k = outputscale * np.exp(-0.5 * d2)
        if wall_kernel is not None:
            wall_ls = wall_kernel.lengthscale.detach().numpy().ravel()
            h2 = (((hp[:, None] - hq[None]) / wall_ls) ** 2).sum(-1)
            k = k * np.exp(-0.5 * h2)
        return k

    y = belief.y_train
    mu, sd = y.mean(), y.std()
    train, query = inputs(belief.xy_train_phys), inputs(xq)
    A = kernel(train, train) + noise * np.eye(len(y))
    Ks = kernel(train, query)
    mean = const + Ks.T @ np.linalg.solve(A, (y - mu) / sd - const)
    var = outputscale - np.sum(Ks * np.linalg.solve(A, Ks), axis=0)
    return mu + sd * mean, sd**2 * var


@pytest.mark.parametrize('optimize', [False, True], ids=['fixed', 'fitted'])
@pytest.mark.parametrize('variant', ['baseline', 'wall_aware'])
def test_posterior_matches_exact_gp(variant, optimize):
    belief = fitted_belief(variant, optimize)
    xq = np.random.default_rng(3).uniform((0, 0), (2, 4), size=(20, 2))
    mean, var = belief.query(xq)
    mean_ref, var_ref = oracle_posterior(belief, xq)
    scale = belief.y_train.std()
    np.testing.assert_allclose(mean, mean_ref, rtol=1e-6, atol=1e-6 * scale)
    np.testing.assert_allclose(var, var_ref, rtol=1e-5, atol=1e-6 * scale**2)
    assert np.all(var > 0)


@pytest.mark.parametrize('variant', ['baseline', 'wall_aware'])
def test_query_with_covariance_consistent(variant):
    belief = fitted_belief(variant)
    xq = np.array([[0.5, 1.0], [0.9, 2.0], [1.1, 2.0], [1.5, 3.0]])
    mean, var = belief.query(xq)
    cmean, cov = belief.query_with_covariance(xq)
    assert cov.shape == (4, 4)
    np.testing.assert_allclose(cmean, mean, rtol=1e-9)
    np.testing.assert_allclose(np.diag(cov), var, rtol=1e-6)
    np.testing.assert_allclose(cov, cov.T, rtol=1e-9)


def test_physical_hyperparameters():
    baseline = fitted_belief('baseline').fitted_hyperparameters_physical()
    assert set(baseline) == {
        'lengthscale_x_m',
        'lengthscale_y_m',
        'outputscale',
        'noise_variance',
    }
    # Normalized initial lengthscales times the domain extent (2 m, 4 m).
    assert baseline['lengthscale_x_m'] == pytest.approx(0.4, rel=1e-6)
    assert baseline['lengthscale_y_m'] == pytest.approx(0.4, rel=1e-6)

    wall_aware = fitted_belief('wall_aware').fitted_hyperparameters_physical()
    assert set(wall_aware) == set(baseline) | {
        'wall_feature_lengthscale_0',
        'wall_feature_lengthscale_1',
    }
    assert wall_aware['wall_feature_lengthscale_0'] == pytest.approx(
        1.0, rel=1e-6
    )
