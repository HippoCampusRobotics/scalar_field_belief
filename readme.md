# scalar_field_belief

`scalar_field_belief` is a ROS 2 package for maintaining a simple Gaussian-process belief over a 2D scalar field.

The node subscribes to scalar measurements, fits an exact GPyTorch GP over physical `(x, y)` positions, provides a batch query service for posterior mean and variance, and publishes RViz point clouds for visualization.

The current version is intentionally simple:

- 2D field belief over physical `x/y`
- exact GP using GPyTorch
- fixed physical input normalization bounds
- target standardization from observed measurement values
- query service for latent posterior mean and variance
- query service for latent posterior mean and full covariance matrix
- RViz point clouds for mean and variance visualization
- optional wall-aware mode with smooth wall features (see [Wall-aware belief](#wall-aware-belief))
- a separate, standalone visibility-graph GP belief (`scalar_field_belief.visgp`, no ROS node yet; see [visGP belief](#visgp-belief))

The numerical core (`scalar_field_belief.belief`, `config`, `models`, `transforms`, `geometry`, `wall_features`, and the `visgp` subpackage) does not depend on ROS 2 and can also be used as an ordinary Python library (see [Standalone Python use](#standalone-python-use-without-ros-2)).

It is designed to work together with:

- `scalar_field_interfaces`
- `scalar_field_sim`


## Installation

The package is intended to be built in a ROS 2 Jazzy workspace.

The examples below assume `~/ros2` as the workspace and `~/ros2/venv` as the workspace-local Python virtual environment.

### 1. Clone the package

```bash
cd ~/ros2/src
git clone git@github.com:HippoCampusRobotics/scalar_field_belief.git
```

Make sure the required interface package is also available in the workspace:

```bash
cd ~/ros2/src
git clone git@github.com:HippoCampusRobotics/scalar_field_interfaces.git
```

For simulation-based testing, also clone `scalar_field_sim`:

```bash
cd ~/ros2/src
git clone git@github.com:HippoCampusRobotics/scalar_field_sim.git
```

### 2. Create the Python virtual environment

The GP dependencies `torch` and `gpytorch` are installed in a workspace-local virtual environment.

Use `--system-site-packages` so that the virtual environment can still access ROS Python packages installed by apt.

```bash
cd ~/ros2
python3 -m venv --system-site-packages venv
source ~/ros2/venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch gpytorch matplotlib
```

### 3. Install ROS dependencies

```bash
cd ~/ros2
rosdep install --from-paths src --ignore-src -r -y
```

### 4. Build the workspace

Build the workspace.

The virtual environment must be active when running the node, because `torch`
and `gpytorch` are installed there. Activating it during build should not be required for this package.

## Standalone Python use (without ROS 2)

This is independent of the ROS installation above. The numerical core only needs NumPy, PyTorch, and GPyTorch (Python >= 3.11); the visGP belief needs NumPy and SciPy only. Install it in editable mode into an ordinary virtual environment, without sourcing ROS 2:

```bash
python3 -m venv ~/venvs/field-belief
source ~/venvs/field-belief/bin/activate
python -m pip install -e ~/ros2/src/scalar_field_belief
```

The `pyproject.toml` is only used for this pip install; the ROS build still uses `CMakeLists.txt`/`package.xml`. Do not import `scalar_field_belief.visualization` in standalone use, since it needs ROS message packages and Matplotlib.

Minimal wall-aware example with directly constructed geometry. All required `BeliefConfig` arguments are listed explicitly; the visualization settings are only used by the ROS node. With `refit_policy='every_measurement'`, the GP is fitted after the first measurement, so the belief can be queried afterwards:

```python
import numpy as np

from scalar_field_belief.belief import ScalarFieldBelief
from scalar_field_belief.config import BeliefConfig
from scalar_field_belief.geometry import WallGeometry, WallSegment

config = BeliefConfig(
    frame_id='map',
    x_min=0.0,
    x_max=2.0,
    y_min=0.0,
    y_max=4.0,
    kernel_type='rbf',
    training_iter=25,
    learning_rate=0.1,
    init_lengthscale_x=0.09,
    init_lengthscale_y=0.045,
    init_outputscale=1.0,
    init_noise=0.01,
    refit_policy='every_measurement',
    refit_every_k=1,
    optimize_hyperparameters=False,
    publish_visualization=False,
    visualization_grid_step=0.1,
    visualization_z_mode='flat',
    visualization_height_scale=0.5,
    model_variant='wall_aware',
    wall_beta=1.0,
    wall_rho=0.35,
    init_wall_lengthscale=1.0,
)

geometry = WallGeometry(
    name='simple_wall_case',
    x_range=(0.0, 2.0),
    y_range=(0.0, 4.0),
    walls=(
        WallSegment(start=(1.0, 0.6), end=(1.0, 3.4)),
        WallSegment(start=(1.0, 2.1), end=(1.5, 2.1)),
    ),
)

belief = ScalarFieldBelief(config, wall_geometry=geometry)

# (x [m], y [m], measured value)
readings = [
    (0.5, 1.0, 0.012),
    (0.6, 2.0, 0.010),
    (1.4, 1.5, 0.004),
    (1.4, 3.0, 0.003),
]
for x, y, value in readings:
    belief.add_measurement(x=x, y=y, value=value)

query_xy = np.array([[0.9, 2.0], [1.1, 2.0], [1.4, 2.5]])
mean, variance = belief.query(query_xy)  # shapes (3,), (3,)
mean, covariance = belief.query_with_covariance(query_xy)  # (3,), (3, 3)
print(mean, variance, covariance)
print(belief.fitted_hyperparameters_physical())
```

For the baseline GP, omit `model_variant` (default `'baseline'`) and the geometry: `ScalarFieldBelief(config)`.

To load the geometry from a TOML file instead (a geometry file or a `scalar_field_sim` scenario, see [Geometry files](#geometry-files); the path is used as given, relative to the current working directory):

```python
from scalar_field_belief.geometry import load_wall_geometry_from_toml

geometry = load_wall_geometry_from_toml(
    'src/scalar_field_sim/geometry/tank_2x4_two_walls.toml'
)
belief = ScalarFieldBelief(config, wall_geometry=geometry)
```


## Running the node

After pulling changes that add Python modules or config files, rebuild the package with `colcon build --packages-select scalar_field_belief`, since an existing (symlink) install does not contain the new modules or files.

Activate the virtual environment:

```bash
cd ~/ros2
source ~/ros2/venv/bin/activate
```

A test launch file is provided:

```bash
ros2 launch scalar_field_belief test_belief.launch.py vehicle_name:=uuv00
```

This launch file starts only the belief node. It does not start the scalar field simulator or publish measurements. To see belief clouds, publish measurements to `/<vehicle_name>/ir_measurement` or run it together with a simulator/sensor node.


## Wall-aware belief

The wall-aware mode ports the wall-feature GP of the research experiments. For each wall segment `m` (start `a`, end `b`, unit tangent `t`, normal `n = (-t_y, t_x)`), a dimensionless feature is computed from the **physical** position `x` in meters:

```text
s_m(x) = dot(x - a, n)                          signed distance to the wall line
d_m(x) = distance from x to the finite segment
h_m(x) = tanh(wall_beta * s_m(x)) * exp(-d_m(x)^2 / (2 * wall_rho^2))
```

The GP inputs become `[x_norm, y_norm, h_0(x), ..., h_(M-1)(x)]` and the kernel becomes `ScaleKernel(spatial_kernel * RBF_wall)`, where the spatial kernel (RBF/Matern as configured) acts on the normalized coordinates and the ARD RBF wall kernel acts on the wall features. The public API (measurements, queries, service responses) still uses physical `(x, y)` positions and the same units as the baseline.

> **Caveat:** this is a smooth, feature-based attenuation of correlation across walls. It does **not** guarantee zero correlation between every pair of points separated by a wall, in particular near wall endpoints or far away from a wall, where the features decay towards zero.
>
> The effect is small with the shipped defaults (`wall_beta: 1.0`, `wall_rho: 0.35`, `init_wall_lengthscale: 1.0`, and `optimize_hyperparameters: false` in the example YAML): then `|h| <= ~0.2`, so the wall kernel reduces cross-wall correlation by at most roughly 8%. Stronger effects need smaller or learned wall lengthscales. Because the feature takes its sign from the infinite supporting line within the decay envelope, it can also reduce correlation between points that are *not* separated by the wall, e.g. around segment endpoints and at wall junctions, and between a point near a wall and a point far from it on the same side. These figures follow from the feature formula; they have not been validated by tests.

### Selecting the mode

`model_variant` selects the mode:

- `baseline` (default): the unchanged GP over normalized `x/y`. No geometry is needed and the `wall_*` parameters are not used (they must still be valid positive numbers).
- `wall_aware`: requires a wall geometry with at least one wall whose bounds match `x_min/x_max/y_min/y_max`, selected with `belief_geometry`. The node refuses to start if `belief_geometry` is empty, if `belief_geometry: scenario` is set without `scenario_path`, or if the geometry is missing, invalid, or contains no walls; there is no fallback to baseline.

### Parameters

| Parameter | Default | Unit | Meaning |
| --- | --- | --- | --- |
| `model_variant` | `baseline` | - | `baseline` or `wall_aware` |
| `wall_beta` | `1.0` | 1/m | Fixed sharpness of the sign change across the wall line |
| `wall_rho` | `0.35` | m | Fixed decay length of the feature away from the segment |
| `init_wall_lengthscale` | `1.0` | dimensionless | Initial lengthscale of every wall feature |
| `belief_geometry` | `''` | - | Node only: `scenario`, or a geometry or scenario TOML file, absolute or relative to the installed `scalar_field_sim` share directory (only read in `wall_aware` mode) |
| `scenario_path` | `''` | - | Node only: `scalar_field_sim` scenario used with `belief_geometry: scenario`, absolute or relative to the installed `scalar_field_sim` share directory. The launch files always pass their `scenario_path` argument (default `''` in `test_belief.launch.py`), which overrides a value in the parameter YAML |

`wall_beta` and `wall_rho` are fixed; they are not learned. Like the spatial lengthscales, the wall lengthscales are optimized only if `optimize_hyperparameters: true`; otherwise they stay at `init_wall_lengthscale`. Use float literals (e.g. `1.0`, not `1`) for these parameters in YAML files.

`fitted_hyperparameters_physical()` additionally reports `wall_feature_lengthscale_0`, `wall_feature_lengthscale_1`, ... in wall-aware mode. These are dimensionless and are not scaled by the domain size.

### Geometry files

The belief chooses its geometry independently of the simulated field. The geometry TOML schema is the one of the `scalar_field_sim` geometry files (`geometry/*.toml` in that package); other sections, such as sources, are ignored:

```toml
[general]
name = "tank_2x4_two_walls"   # optional, defaults to the file name stem

[bounds]
x_min = 0.0
x_max = 2.0
y_min = 0.0
y_max = 4.0

[objects.walls]
wall_vertices = [[1.0, 0.6, 1.0, 3.4], [1.0, 2.1, 1.5, 2.1]]  # [x1, y1, x2, y2]
```

`load_wall_geometry_from_toml(path)` accepts:

- a geometry file as above, or a self-contained `scalar_field_sim` scenario (bounds and walls inline),
- a `scalar_field_sim` scenario with a top-level `geometry_file` reference: bounds, walls, and name are then read from the referenced file, resolved relative to the scenario's directory (absolute paths are used as given). The referenced file must not contain `geometry_file` itself (`ValueError`); a missing referenced file raises `FileNotFoundError`. All other content of the scenario (`field_file`, `[general]`, ...) is ignored.

`belief_geometry` selects the file in the node:

- `scenario`: the scenario in `scenario_path`, i.e. the walls of the simulated scenario. The launch files pass the same `scenario_path` to the belief as to the simulator.
- any other non-empty value: a geometry or scenario file, e.g. `geometry/tank_2x4_two_walls.toml`.

Both follow the path rule of the `scalar_field_sim` nodes (`scalar_field_sim.config.resolve_package_path`): absolute paths are used as given; relative paths are resolved against the installed `scalar_field_sim` share directory (`$(ros2 pkg prefix --share scalar_field_sim)`), e.g. `scenarios/tests/fields/01_single_wall.toml`. Simulator and belief therefore read the same file for the same `scenario_path`. Files are only found there after `colcon build` installed them; rebuild after adding or changing them.

The node logs the file the geometry was loaded from and whether it was selected through the scenario. The pure Python loader `load_wall_geometry_from_toml(path)` does not perform the share-directory lookup; it uses the path it is given.

### Frames

Walls, measurements, and queries are all interpreted in the configured `frame_id` (usually `map`). The node performs no TF transformation, so the geometry file must already be expressed in that frame.

### Launching the wall-aware configuration

Reuse the `belief_params` argument of the existing launch file and pass the scenario whose walls the belief should use:

```bash
ros2 launch scalar_field_belief test_belief.launch.py vehicle_name:=uuv00 \
  belief_params:=$(ros2 pkg prefix --share scalar_field_belief)/config/belief_params_wall_aware.yaml \
  scenario_path:=$(ros2 pkg prefix --share scalar_field_sim)/scenarios/tests/fields/01_single_wall.toml
```

`config/belief_params_wall_aware.yaml` uses the same settings as `belief_params_default.yaml`, except `model_variant: wall_aware` and `belief_geometry: scenario` (the scenario bounds must be `0..2 x 0..4`). To use a geometry that differs from the simulated scenario, override `belief_geometry` with a geometry file instead, e.g. in a copy of the YAML file.

`top_field_sim_stack.launch.py` passes its `scenario_path` to both the simulator and the belief; its default scenario (`simple_field_1`) has no walls, so give a scenario with walls when using the wall-aware parameters:

```bash
ros2 launch scalar_field_belief top_field_sim_stack.launch.py vehicle_name:=uuv00 \
  belief_params:=$(ros2 pkg prefix --share scalar_field_belief)/config/belief_params_wall_aware.yaml \
  scenario_path:=$(ros2 pkg prefix --share scalar_field_sim)/scenarios/tests/fields/01_single_wall.toml
```


## visGP belief

`scalar_field_belief.visgp` is a small, dense reference implementation of a
visibility-graph Gaussian process (visGP) after Gilbert and Datta (2024) for
domains with known straight walls. It uses NumPy and SciPy (float64) only: no
ROS, no PyTorch. There is no ROS node yet; it is used directly from Python or
through the [belief playground](tools/belief_playground/README.md)
(`--backends gp,visgp`).

| Module | Content |
| --- | --- |
| `visgp/config.py` | `VisGPConfig` (fixed parameters, `validate()`) |
| `visgp/visibility.py` | contact tolerance, on-wall test, segment visibility, reference grid, connected components |
| `visgp/covariance.py` | Matérn-3/2 parent covariance, maximal cliques, covariance selection |
| `visgp/extension.py` | location identity, nearest-clique neighbour sets, rows `b(x)` and residual variances `F(x)` |
| `visgp/belief.py` | `VisGPBelief`: `reset`, `add_measurement`, `query`, `predict_marginal`, `query_with_covariance`, `valid_mask`, `status` |

### Model

Latent field and independent sensor noise, all in physical units (meters;
"units" = scenario measurement unit, e.g. V; no target standardisation):

```text
f ~ GP(prior_mean, C_vis)
y_i = f(x_i) + e_i,   e_i ~ N(0, measurement_noise_variance), independent
C(d) = signal_variance * (1 + sqrt(3) d / lengthscale) * exp(-sqrt(3) d / lengthscale)
```

Nothing is fitted: kernel parameters, noise variance, reference set and
covariance selection are fixed when the belief is constructed.

**Reference set V.** By default a cell-centred grid built from the domain
bounds and walls only: `nx = max(1, round((x_max - x_min) / reference_spacing))`
(same for y), node `(i, j)` at the cell centre
`(x_min + (i + 0.5) dx, y_min + (j + 0.5) dy)` with `dx = (x_max - x_min) / nx`.
Nodes within the contact tolerance of a wall are dropped (the number is
recorded). With the default 0.25 m spacing a 2 m x 4 m tank has 8 x 16 = 128
nodes. A grid larger than `max_reference_nodes` is rejected with an error
naming its size; the resolution is never changed silently. Explicit reference
locations (`VisGPBelief(config, geometry, reference_locations=...)`) must be
finite, distinct, inside the domain and off walls. V never depends on
measurements, queries, the display grid or source information, and
measurements are never added to it.

**Visibility.** Walls are zero-width finite segments (no buffering, no
thickness). With the contact tolerance `delta = 1e-9 * max(x extent, y extent)`
[m], a position is on a wall if it is within `delta` of one, and two positions
see each other if the segment between them stays farther than `delta` from
every wall. Crossing, touching (including contact at a wall endpoint) and
collinear overlap are blocked. The predicate is exactly symmetric and does not
depend on the order of the walls or of their endpoints; diagonal walls,
isolated nodes and disconnected components are supported. The domain boundary
is not a wall.

**Exact construction on V.** With `K = C(V, V)` and the visibility graph `G`
on V, `L = CovSel(K, G)` is the unique symmetric positive definite matrix with

```text
L_ii = K_ii,   L_ij = K_ij for visible pairs,   inv(L)_ij = 0 for non-visible pairs
```

(the maximum-determinant completion; Dempster 1972). It is solved per
connected component of `G`: a complete component keeps `L = K` (so without
walls `L = K` exactly), `L` is exactly zero between components, and any other
component is solved by the iterative proportional scaling of Speed and Kiiveri
(1986) over its maximal cliques (Bron-Kerbosch), in covariance form, starting
from `diag(K)`. It stops when `max |L - K|` on the diagonal and the visible
pairs is at most `covsel_tolerance * signal_variance` and the normalised
precision `|P_ij| / sqrt(P_ii P_jj)` (`P = inv(L)`) of every non-visible pair
is at most `covsel_precision_tolerance`. Non-convergence within
`covsel_max_iterations` sweeps, more than `max_cliques` cliques or a failed
Cholesky factorisation raise `CovarianceSelectionError` with these residuals.
No graph edges are added (no chordal completion), and nothing is masked,
clipped or jittered. Sweeps, residuals, the smallest eigenvalue and the
condition number of `L` are reported (`belief.covariance_selection`).

**Finite-reference process extension.** A position `x` gets a neighbour set
`N(x)` in V and

```text
b_N(x) = C(x, N) inv(C(N, N))         (row b(x) over V, zero outside N)
F(x)   = C(x, x) - C(x, N) inv(C(N, N)) C(N, x)
C_vis(x, z) = b(x) L b(z)^T + I[x = z] F(x)
```

A position exactly in V has the unit row and `F = 0`; a position without
usable neighbours has `b = 0` and `F = C(x, x)`. `I[x = z]` means the same
spatial location (exact float64 equality of both coordinates; `-0.0 == 0.0`),
not the same observation or array index. Positions are never rounded or
snapped to the reference or display grid. The same `C_vis` is used for
training/training, training/query and query/query blocks; queries never
rebuild covariance selection.

Neighbour sets use the nearest-clique rule (Algorithm 2 in the supplement of
the paper): the reference nodes visible from `x` are sorted by (squared
distance, reference index); among the first `max_neighbors` of them,
candidates are added in that order while the set stays a clique of `G`, and
the scan **stops at the first incompatible candidate**.

The authors' R code (`functions.R`) is not a reference implementation of this
construction. It never runs iterative proportional scaling on `G`: its
likelihood (`cov_select_loglik`) uses the cliques and separators (`rip()`) of
a chordal completion of `G` (`is_chordal(newgraph = TRUE)`), with the nugget
inside every clique and separator covariance, and V = the data locations. Its
prediction (`pred_neighbor`) kriges from the observations at the neighbours
only and includes the nugget (`Sig11`), whereas this implementation conditions
on all observations through `C_vis` and returns the latent variance. Its
nearest-clique variant also differs: it skips incompatible candidates instead
of stopping, and with ties at its distance cutoff it keeps the lowest indices
among all candidates within the cutoff (which can drop a nearer candidate).

**Sensor noise and noise order.** Sensor noise is added *after* the latent
process: `Cov(y) = C_vis + measurement_noise_variance * I`. Repeated
observations at one off-reference location share its residual `F` and get
independent noise. The paper's data model (Sec. 3) applies covariance
selection to a nugget-inclusive covariance on the data locations,
`CovSel(C + tau^2 I, G)`. The two orders are not interchangeable: for a path
1-2-3 with all parent correlations 0.5, `signal_variance = 1` and
`tau^2 = 1`, `CovSel(C) + tau^2 I` gives `Cov(y_1, y_3) = 0.25`, while
`CovSel(C + tau^2 I)` gives `0.125`.

**What is exact and what is not.**

- Exact on V: the diagonal and visible pairs of `L` equal `K`, non-visible
  pairs are conditionally independent, and without walls `L = K`.
- Off V, `C_vis` is a finite-reference construction and need not reproduce
  the Euclidean `C`, even without walls (e.g. for two distinct off-reference
  positions `C_vis = b(x) K b(z)^T`). The marginal prior variance equals
  `signal_variance` everywhere (`C_vis(x, x) = C(x, x)`, because `N(x)` is a
  clique and `L = K` on cliques up to the covariance-selection tolerance).
- `F(x)` is an independent per-location residual: the standard deviation is
  discontinuous at a measured off-reference position (a position 1e-9 m
  away does not share its residual), and the mean can jump where the
  neighbour set changes, most visibly next to wall ends and openings. With
  the default settings, `F / signal_variance` on the default playground
  display grid (0.05 m) is about 0.03 on average, about 0.04 at most between
  reference nodes away from walls and the boundary, about 0.14 in the corner
  display cells and about 0.20 at the exact domain corner.
- `F = 0` exactly at the reference nodes. With the default 0.05 m display
  step and 0.25 m reference spacing on a 2 m x 4 m tank, 128 display cells
  coincide with reference nodes, so posterior standard deviation maps show
  small dips at the lattice nodes.
- Information propagates around finite wall ends and through openings, so
  the covariance between two points on opposite sides of a finite wall is in
  general not zero. It is exactly zero between parts of the domain that the
  walls disconnect, and also across an opening that the reference grid does
  not resolve: if no two reference nodes see each other through it, the
  parts on both sides are separate components of `G` even though they are
  physically connected. Choose `reference_spacing` fine enough for every
  opening.

**Conditioning and queries.** After every `add_measurement` the belief
re-conditions on all stored measurements (Cholesky of the noisy training
covariance); this is a straightforward batch update, not an efficient
streaming one. `query` returns the latent posterior mean and marginal
variance (without sensor noise), computed in blocks without forming an
`(N, N)` matrix; `query_with_covariance` returns the full latent covariance
for at most `max_joint_query_size` positions. Roundoff-negative values of `F`
or of a marginal variance down to `-1e-12 * signal_variance` are returned as
0 and counted (`predict_marginal`, `status()`); more negative values raise.
The playground adapter keeps the counts of its last grid query and saves them
with the session metadata (`last_query`, next to the covariance selection
diagnostics).
The prior (no measurement) is available immediately. `reset()` removes all
measurements and keeps the geometry, V, `G` and `L`. Queries do not change
the belief.

**Invalid positions.** Non-finite input raises `ValueError`. Positions outside
the closed domain rectangle or within the contact tolerance of a wall are
invalid: `query` returns NaN mean and variance there (as the GMRF adapter does
for cells it cannot represent), `add_measurement` and `query_with_covariance`
raise `ValueError` and leave the belief unchanged.

### Configuration

`VisGPConfig` fields; the playground reads them from
`config/belief_params_visgp.yaml` (ROS parameter layout, no ROS runtime
needed; domain bounds and walls always come from the scenario).

| Parameter | YAML value | Unit | Meaning |
| --- | --- | --- | --- |
| `prior_mean` | `0.0` | units | constant prior mean (default 0.0) |
| `signal_variance` | `1.0e-4` | units^2 | variance of the parent covariance (required; a variance, not a std) |
| `lengthscale` | `0.5` | m | Matérn-3/2 lengthscale (required) |
| `measurement_noise_variance` | `2.5e-7` | units^2 | independent sensor noise variance, > 0 (required) |
| `reference_spacing` | `0.25` | m | target spacing of the generated reference grid (required) |
| `max_reference_nodes` | `200` | - | dense size limit of the reference set (default 200) |
| `max_neighbors` | `8` | - | nearest-clique neighbour cap `k` (default 8) |
| `covsel_tolerance` | `1.0e-10` | - | diagonal/edge residual of `L`, relative to `signal_variance` (default 1e-10) |
| `covsel_precision_tolerance` | `1.0e-9` | - | normalised non-edge precision of `L` (default 1e-9) |
| `covsel_max_iterations` | `500` | - | bound on the number of IPS sweeps (default 500) |
| `max_cliques` | `2000` | - | bound on the number of maximal cliques (default 2000) |
| `max_joint_query_size` | `2000` | - | largest `query_with_covariance` batch (default 2000) |

The demonstration values match the GMRF's prior and noise standard
deviations (0.01 and 5e-4 units). Prefer YAML floats with a decimal point
and a signed exponent (`1.0e-4`): PyYAML reads `1e-4` as a string. The
playground converts numeric strings, so both work there, but only the first
is a number for other YAML readers.

### Usage without ROS

From Python (paths relative to the workspace root, with
`src/scalar_field_belief` on `PYTHONPATH` or the package installed with pip):

```python
import numpy as np

from scalar_field_belief.geometry import load_wall_geometry_from_toml
from scalar_field_belief.visgp import VisGPBelief, VisGPConfig

geometry = load_wall_geometry_from_toml(
    'src/scalar_field_sim/scenarios/tests/fields/01_single_wall.toml'
)
config = VisGPConfig(
    signal_variance=1.0e-4,  # [V^2], a variance
    lengthscale=0.5,  # [m]
    measurement_noise_variance=2.5e-7,  # [V^2]
    reference_spacing=0.25,  # [m] -> 8 x 16 = 128 reference nodes
)
belief = VisGPBelief(config, geometry)  # builds V, G and L once
belief.add_measurement(x=0.5, y=1.0, value=0.0106)
belief.add_measurement(x=0.7, y=2.0, value=0.0111)

xy = np.array([[0.6, 1.5], [1.5, 1.5], [1.0, 2.0]])  # (1.0, 2.0) is on the wall
mean, variance = belief.query(xy)  # latent; NaN on the wall
mean, covariance = belief.query_with_covariance(xy[:2])  # (2,), (2, 2)
print(belief.covariance_selection.summary())
```

Playground comparison and the numerical diagnostic, from the workspace root
without sourcing ROS (the Python interpreter needs NumPy, SciPy, PyYAML,
PySide6 and pyqtgraph, plus PyTorch and GPyTorch for `gp`):

```bash
env -u PYTHONPATH -u AMENT_PREFIX_PATH \
  PYTHONPATH=src/scalar_field_belief:src/scalar_field_belief/tools/belief_playground:src/scalar_field_sim \
  python3 -m belief_playground --scenario 01_single_wall --backends gp,visgp

env -u PYTHONPATH -u AMENT_PREFIX_PATH \
  PYTHONPATH=src/scalar_field_belief:src/scalar_field_belief/tools/belief_playground:src/scalar_field_sim \
  python3 -m belief_playground.visgp_diagnostics
```

The diagnostic checks the model invariants on small hand-made geometries and
on the scenario reference grids, and reports the reconstruction quality of the
baseline GP and visGP after the same measurements on one side of the wall
(reported only, not judged). Every check ends as passed, failed or not run
(with the reason, e.g. a default scenario left out with `--scenarios`). Exit
code: 0 if everything passed, 1 if a check failed, 3 if none failed but some
did not run; 2 is a usage error, including `--export-image` without a
display. `--visgp-config YAML` selects the visGP parameter file of the
scenario runs (default: the checkout's `config/belief_params_visgp.yaml`,
else the installed one). It writes `report.md`, `report.json`, `arrays.npz`
and the playground sessions to
`./belief_playground_sessions/visgp_diagnostics/<UTC time>/` (or `--out-dir`);
`--export-image` then adds a PNG of the playground window per scenario (Qt;
it needs a working display, e.g. `xvfb-run` on machines without a desktop).
See the [playground README](tools/belief_playground/README.md#visgp).

### Limitations and deferred work

- Dense matrices: memory and time grow with the number of reference nodes
  (`m^2`) and measurements (`n^3` per re-conditioning).
- Covariance selection is sensitive to the graph and the kernel. Its cost
  grows with the number of maximal cliques, which is much larger next to
  openings, and its convergence slows down for long lengthscales. Examples
  with the default tolerances (times approximate, with an unoptimised
  reference BLAS):
  - `01_single_wall`, default settings: 22 cliques, 24 sweeps, about 0.3 s.
  - `02_wall_with_gap`, default settings: 1295 cliques, 19 sweeps, about
    6.5 s.
  - `02_wall_with_gap`, lengthscale 1.0 m: 54 sweeps, about 20 s.
  - `01_single_wall`, lengthscale 2.0 m: no convergence within 500 sweeps
    (diagonal/edge residual 1.9e-10 against the tolerance 1e-10), so
    `CovarianceSelectionError` is raised.
  - `02_wall_with_gap`, 0.2 m spacing (200 nodes): 4502 cliques, more than
    `max_cliques = 2000`, so construction is refused. With a raised limit it
    converges in 18 sweeps but takes over a minute.

  The playground builds the belief before its window opens, so this time
  delays the start.
- Fixed parameters; no hyperparameter optimisation. Only the Matérn-3/2
  parent covariance and the nearest-clique neighbour rule.
- Deferred: hyperparameter optimisation, sparse/variational (SVGP)
  approximations, GPU support, chordal completion, graph sparsification and
  distance cutoffs, incremental (streaming) solvers, a ROS 2 wrapper and a
  C++ implementation.

### References

- B. Gilbert and A. Datta (2024), "Visibility graph-based covariance
  functions for scalable spatial analysis in non-convex partially Euclidean
  domains", *Biometrics*, ujae089,
  [doi:10.1093/biomtc/ujae089](https://doi.org/10.1093/biomtc/ujae089);
  manuscript and supplement:
  [arXiv:2307.11941v3](https://arxiv.org/abs/2307.11941v3).
- Author R implementation:
  [github.com/bjg345/visGP](https://github.com/bjg345/visGP), `functions.R`.

The construction (covariance selection on a fixed reference set, the process
extension and the nearest-clique rule) follows the arXiv v3 text (Sec. 2.1-2.2
and Algorithm 2 in Sec. S3.4); Sec. 3 is the source of the nugget-inclusive
data model it is compared with. `functions.R` was read for comparison only
(see the differences above); this is not a port of it. The published
Biometrics version was not compared.


## Full Simulation Setup

For a fully functional demo, run the following launch files. 

Start the simulation with one robot (HippoCampus in this case):
```bash
ros2 launch hippo_sim  top_hippocampus_complete.launch.py vehicle_name:=uuv00
```

As an example, start a path follower:
```bash
ros2 launch hippo_control top_path_following_intra_process.launch.py vehicle_name:=uuv00 use_sim_time:=true
```

And get the robot moving: 
```bash
ros2 topic pub -r 50 /uuv00/thrust_setpoint hippo_control_msgs/msg/ActuatorSetpoint "{x: 0.8}"
```

Finally, within the virtual environment, start a full field estimation setup for testing:

```bash
ros2 launch scalar_field_belief top_field_sim_stack.launch.py vehicle_name:=uuv00
```



## Additional Information

### Model conventions

The public API of the belief uses physical coordinates in meters.

Internally, the GP uses normalized inputs:

```text
x_norm = (x - x_min) / (x_max - x_min)
y_norm = (y - y_min) / (y_max - y_min)
```

Measurement values are standardized during GP fitting. This is separate from input normalization:

- input normalization uses fixed physical domain bounds
- target standardization uses the observed measurement values

The query service returns the latent GP posterior mean and variance in the original measurement units. The returned variance is in measurement-units squared and does not include additional observation noise.


### Public API

#### Subscribed topics

```text
/<vehicle_name>/ir_measurement
```

Type:

```text
scalar_field_interfaces/msg/ScalarMeasurement
```

Only the following fields are used by the current 2D belief:

```text
header.frame_id
pose.position.x
pose.position.y
value
```

The pose `z` coordinate and orientation are currently ignored. The message frame must match the configured `frame_id`, usually `map`. The node does not transform measurements between frames.

#### Services

```text
/<vehicle_name>/query_scalar_field_belief
```

Type:

```text
scalar_field_interfaces/srv/QueryScalarFieldBelief
```

Queries the current GP belief at a batch of poses and returns posterior mean and variance.

```text
/<vehicle_name>/query_scalar_field_belief_covariance
```

Type:

```text
scalar_field_interfaces/srv/QueryScalarFieldBeliefCovariance
```

Queries the current GP belief at a batch of poses and returns posterior mean and full covariance matrix instead of just the diagonal (variance).

> `covariance_matrix` is flattened because ROS messages can only hold 1D arrays, not a real N x N grid. For two points, the 4 numbers come back in this order: point 1 vs point 1, point 1 vs point 2, point 2 vs point 1, point 2 vs point 2. With more points, the same pattern continues: the whole first row comes first, then the whole second row, and so on.

> The response also includes `measurement_noise_variance`, the GP's likelihood noise (sigma_n^2) in the same physical units as `covariance_matrix`. Callers simulating a new noisy measurement at a queried pose should add this to the relevant diagonal entries before conditioning, instead of assuming their own noise value.

```text
/<vehicle_name>/reset_scalar_field_belief
```

Type:

```text
std_srvs/srv/Trigger
```

Clears all stored measurements and fitted GP state.

#### Published topics

For visualization only:

```text
/<vehicle_name>/belief/mean_cloud
/<vehicle_name>/belief/variance_cloud
```

Both are `sensor_msgs/PointCloud2` on the visualization grid, in `frame_id`,
latched (`transient_local`, depth 1). The mean cloud has the fields `x`, `y`,
`z`, `rgb` (mean mapped to colour, for RViz) and `intensity` (posterior mean
value). The variance cloud has `x`, `y`, `z` and `intensity` (posterior
variance value). `z` is only a display offset (and the height in
`visualization_z_mode: height`).

### Example commands

#### Publish one test measurement

```bash
ros2 topic pub --once /uuv00/ir_measurement \
  scalar_field_interfaces/msg/ScalarMeasurement \
  "{header: {frame_id: 'map'}, pose: {position: {x: 1.0, y: 2.0, z: 0.0}, orientation: {w: 1.0}}, value: 0.01}"
```

With the default `refit_policy: every_measurement`, the node should fit a GP immediately after receiving the first measurement.

#### Query the belief for mean and variance

```bash
ros2 service call /uuv00/query_scalar_field_belief \
  scalar_field_interfaces/srv/QueryScalarFieldBelief \
  "{queries: [{header: {frame_id: 'map'}, pose: {position: {x: 1.0, y: 2.0, z: 0.0}, orientation: {w: 1.0}}}]}"
```

If no fitted model exists yet, the service returns `success: false`.

#### Query the belief for mean and full covariance matrix

```bash
ros2 service call /uuv00/query_scalar_field_belief_covariance \
  scalar_field_interfaces/srv/QueryScalarFieldBeliefCovariance \
  "{queries: [{header: {frame_id: 'map'}, pose: {position: {x: 1.0, y: 2.0, z: 0.0}, orientation: {w: 1.0}}}, {header: {frame_id: 'map'}, pose: {position: {x: 1.1, y: 2.0, z: 0.0}, orientation: {w: 1.0}}}]}"
```

If no fitted model exists yet, the service returns `success: false` as well.

#### Reset the belief

```bash
ros2 service call /uuv00/reset_scalar_field_belief std_srvs/srv/Trigger {}
```
