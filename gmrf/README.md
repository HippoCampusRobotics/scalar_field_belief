# GMRF scalar-field belief (C++)

A port of the Gaussian Markov random field (GMRF) estimator by Finn Lukas Busch
(UnderwaterMultiagentExploration, `gmrf/`; see [PROVENANCE.md](PROVENANCE.md)) as

* a ROS-independent C++17 library (`scalar_field_belief::gmrf`, depends only on
  the C++ standard library and Eigen 3),
* an optional TOML loader (`scalar_field_belief::gmrf_toml`, needs toml++),
* a ROS 2 Jazzy node `gmrf_belief_node` with the same topics/services as the GP
  node `scalar_field_belief_node.py`.

The existing Python GP backend is unchanged. The GMRF node is built only on
request; the core and its Python module are built by default for the belief
playground.
Optional Python bindings are described in [Python bindings](#python-bindings-optional).

## Layout

```
gmrf/
  CMakeLists.txt                     standalone project, also used via add_subdirectory
  include/scalar_field_belief/gmrf/  config.hpp, gmrf_belief.hpp, toml_loader.hpp
  src/                               config.cpp, gmrf_belief.cpp, geometry.hpp, toml_loader.cpp
  examples/scalar_example.cpp
  tests/                             one GTest executable per tests/test_*.cpp (+ data/)
  ros/                               gmrf_belief_node.cpp, belief_params_gmrf.yaml, gmrf_belief.launch.py
  cmake/                             package config template
```

## Standalone build (no ROS)

Requirements: CMake >= 3.18 (3.20 for `ctest --test-dir`), a C++17 compiler, Eigen >= 3.3 (`libeigen3-dev`),
optionally toml++ (`libtomlplusplus-dev`) and GoogleTest (`libgtest-dev`).
Run from the repository root; keep the build directory outside the source tree.

```bash
cmake -S gmrf -B /tmp/gmrf_build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/gmrf_build -j
ctest --test-dir /tmp/gmrf_build --output-on-failure
/tmp/gmrf_build/examples/gmrf_scalar_example
cmake --install /tmp/gmrf_build --prefix $HOME/.local/scalar_field_belief_gmrf
```

No ROS environment is needed (verified with `env -i HOME=$HOME PATH=/usr/bin:/bin`).
Options: `GMRF_WITH_TOML` (ON; skipped with a message if toml++ is missing),
`GMRF_BUILD_TESTS`, `GMRF_BUILD_EXAMPLES`, `GMRF_INSTALL` (ON only for top-level
builds). Both libraries are static and position independent.

Use the installed package from another CMake project:

```cmake
find_package(scalar_field_belief_gmrf 0.1 REQUIRED)   # CMAKE_PREFIX_PATH=<prefix>
target_link_libraries(my_app PRIVATE scalar_field_belief::gmrf)        # + ::gmrf_toml
```

Tests: add a file `tests/test_<name>.cpp` and re-run CMake; it becomes its own
executable `gmrf_test_<name>`. Files named `test_toml_*.cpp` also link the TOML
loader. `GMRF_TEST_DATA_DIR` points to `tests/data/`.

## ROS 2 build and run

toml++ has no rosdep key; install it with `sudo apt install libtomlplusplus-dev`.

```bash
cd ~/ros2
colcon build --packages-select scalar_field_belief --cmake-args -DSCALAR_FIELD_BELIEF_BUILD_GMRF=ON
source install/local_setup.bash
ros2 launch scalar_field_belief gmrf_belief.launch.py
```

Without `-DSCALAR_FIELD_BELIEF_BUILD_GMRF=ON` (default OFF) the node is not
built and rclcpp/toml++ are not needed. The GMRF core and its Python module are
also built for the belief playground (`-DSCALAR_FIELD_BELIEF_BUILD_PLAYGROUND=ON`,
default OFF). With both options OFF, the package builds without any C++
dependencies. If no
`CMAKE_BUILD_TYPE` is set, the core library alone is compiled with `-O2`;
`--cmake-args -DCMAKE_BUILD_TYPE=Release` is still recommended.

Launch arguments: `belief_params` (default: installed
`config/belief_params_gmrf.yaml`), `namespace` (default `''`), `use_sim_time`
(default `false`). The launch file starts only the GMRF node (node name
`gmrf_belief`).

Side-by-side comparison with the GP backend: both use the same relative names
(`ir_measurement`, `query_scalar_field_belief`, ...), so never run them in the
same namespace. Either use a namespace and remap the measurement input:

```bash
ros2 launch scalar_field_belief gmrf_belief.launch.py namespace:=gmrf
ros2 run scalar_field_belief gmrf_belief_node --ros-args -r __ns:=/gmrf \
  -r ir_measurement:=/ir_measurement \
  --params-file install/scalar_field_belief/share/scalar_field_belief/config/belief_params_gmrf.yaml
```

or relay `/<vehicle>/ir_measurement` into `/gmrf/ir_measurement` (e.g.
`ros2 run topic_tools relay ...`). Services are then
`/gmrf/query_scalar_field_belief`, `/gmrf/query_scalar_field_belief_covariance`,
`/gmrf/reset_scalar_field_belief`; clouds `/gmrf/belief/mean_cloud` and
`/gmrf/belief/variance_cloud`.

### Node behaviour

| Interface | Behaviour |
| --- | --- |
| `ir_measurement` (`ScalarMeasurement`) | x, y from `pose.position`, `value`. Empty `frame_id` = configured frame; other frames are ignored with a warning. Non-finite, out-of-domain, or unrepresentable measurements are rejected with a warning. Each accepted measurement is assimilated immediately (`addMeasurement` + `update`), then clouds are published. |
| `query_scalar_field_belief` | `mean`, `variance` (latent field, see below). Empty request: `success=false`, `'No query poses provided.'`; wrong frame: `"Expected frame 'map', got 'odom'."`; any invalid or out-of-domain point: the whole request fails (`success=false` with the reason). |
| `query_scalar_field_belief_covariance` | `mean`, full `covariance_matrix` (N x N, row-major, explicit loop), `measurement_noise_variance = sigma_s^2`. |
| `reset_scalar_field_belief` (`Trigger`) | Clears all data; the belief returns to the GMRF prior. |
| `belief/mean_cloud`, `belief/variance_cloud` | `PointCloud2` with float32 fields `x, y, z, intensity`; intensity is the actual posterior mean / marginal variance; flat plane at `z_offset`; grid with `visualization_grid_step` over the domain (same axis rule as the GP node); points that cannot be represented (hidden by walls / inactive) are omitted. QoS depth 1, reliable, transient local. Published at startup, after each measurement, and after reset. |

Backend differences to the GP node: a GMRF always has a prior posterior, so
queries before the first measurement succeed (prior mean and prior marginal
variance) instead of failing with "Belief has no fitted model yet.". Query
points outside the domain (or hidden by walls / inactive) make the whole
request fail (`success=false`, reason in `status_message`, logged at WARN);
the GP node extrapolates instead. Genuine solver failures are logged at ERROR.

Invalid parameters stop the node with a FATAL message and exit code 1. Ctrl+C
shuts down cleanly. All parameters are static (read-only).

### Parameters (`ros/belief_params_gmrf.yaml`)

All `sigma_*` values are standard deviations; variances are their squares.

| Parameter | Default | Unit / meaning |
| --- | --- | --- |
| `frame_id` | `map` | frame of measurements, queries, walls |
| `x_min, x_max, y_min, y_max` | 0, 2, 0, 4 | domain [m] |
| `nx, ny` | 41, 81 | grid nodes (0.05 m spacing) |
| `prior_mean` | 0.0 | [units] |
| `prior_sigma` | 0.01 | per-node prior std [units] |
| `sigma_p_per_m` | 0.8 | neighbour std = `sigma_p_per_m * dx^2` [units/m^2] |
| `sigma_s` | 0.0005 | measurement noise std [units] |
| `walls` | unset | flat `[x0, y0, x1, y1, ...]` [m]; use floats |
| `wall_geometry_file` | `''` | TOML wall file, absolute or relative to the package share dir (e.g. `config/walls/simple_wall_case.toml`); combined with `walls` |
| `wall_endpoint_extension` | 0.01 | [m] |
| `free_space_seeds` | unset | flat `[x, y, ...]`, see walls |
| `publish_visualization` | true | |
| `visualization_grid_step` | 0.1 | [m] |
| `z_offset` | -0.8 | [m] |

ROS 2 cannot type an empty YAML list: write `walls: [1.0, 0.6, 1.0, 3.4]` or
leave the key out, but do not write `walls: []`. Integer lists (`[1, 0, 1, 4]`)
are rejected as the wrong type. The ROS node always runs the stationary
scalar-only model; flow energies and aging are only available in the C++ API.

## Minimal C++ API

```cpp
#include "scalar_field_belief/gmrf/gmrf_belief.hpp"
namespace gm = scalar_field_belief::gmrf;

gm::GmrfConfig cfg;              // stationary scalar model by default
cfg.x_min = 0.0; cfg.x_max = 2.0; cfg.y_min = 0.0; cfg.y_max = 4.0;
cfg.nx = 41; cfg.ny = 81;
cfg.prior_sigma = 0.01; cfg.sigma_p_per_m = 0.8; cfg.sigma_s = 0.0005;

gm::GmrfBelief belief(cfg, {gm::WallSegment{1.0, 0.6, 1.0, 3.4}});
belief.addMeasurement(0.8, 2.0, 0.010);   // validated and queued
gm::UpdateReport rep = belief.update();   // assimilate (queries throw while data is pending)

Eigen::MatrixX2d xy(2, 2);
xy << 0.85, 2.05,
      1.15, 2.05;
Eigen::VectorXd mean, var;
Eigen::MatrixXd cov;
belief.queryMeanVariance(xy, mean, var);    // marginals only
belief.queryMeanCovariance(xy, mean, cov);  // full H Q^-1 H^T
double noise_var = belief.measurementNoiseVariance();  // sigma_s^2
belief.reset();                             // back to the prior
```

TOML (`scalar_field_belief/gmrf/toml_loader.hpp`): `loadGmrfConfigToml(path)`
reads the source's `gmrf_conf.toml` layout (`[general] N_X, N_Y, size_x, size_y`
plus optional `x_min, y_min, x_max, y_max, wall_endpoint_extension`;
`[field_params]` with `sigma_p` = `sigma_p_per_m`; `[flow_field_params]`;
`[energies]`; `[solver]`). Missing keys keep the `GmrfConfig` defaults; wrong
types, unknown keys/tables, and syntax errors throw. `loadWallsToml(path)` reads
`[objects.walls] wall_vertices = [[x0, y0, x1, y1], ...]` (the source map format
and the GP wall files), optional `[bounds]`, `[general] name`, `num_walls`.

Errors: `std::invalid_argument` for invalid configuration, walls, positions,
values, times; `std::logic_error` for queries with pending data or flow
measurements in the scalar state; `std::runtime_error` for factorisation /
numerical failure. All mutating calls validate before changing state, and
`update()` has the strong exception guarantee.

## Python bindings (optional)

`python/` contains a small pybind11 module, `scalar_field_belief_gmrf_py`. It
exposes the existing public scalar API listed above and does not duplicate
any numerics. It is used by the belief playground in
`tools/belief_playground/` (see its README). It needs `pybind11-dev` and
`python3-dev`.

**ROS workspace build:** the `scalar_field_belief` package builds
the module and installs it into its Python install directory
(`install/scalar_field_belief/lib/python3.12/site-packages/`) when
`SCALAR_FIELD_BELIEF_BUILD_PLAYGROUND` is ON (default OFF). It is built for the
interpreter that ament uses (`/usr/bin/python3`, Python 3.12) and can be
imported after sourcing the workspace. The gmrf core is added to the build
only once, also when `SCALAR_FIELD_BELIEF_BUILD_GMRF=ON`.

**Standalone (no ROS):** Python support is off by default
(`GMRF_BUILD_PYTHON=OFF`), so the plain C++ build and tests are unchanged.
Nothing is installed from `python/` in this mode (an including project can set
`GMRF_PYTHON_INSTALL_DIR`). To build and test the module:

```bash
cmake -S gmrf -B /tmp/gmrf_py_build -DCMAKE_BUILD_TYPE=Release \
      -DGMRF_BUILD_PYTHON=ON -DPython_EXECUTABLE=/usr/bin/python3.12
cmake --build /tmp/gmrf_py_build -j
ctest --test-dir /tmp/gmrf_py_build --output-on-failure   # + gmrf_python_bindings
PYTHONPATH=/tmp/gmrf_py_build/python python3 -c "import scalar_field_belief_gmrf_py"
```

```python
import numpy as np, scalar_field_belief_gmrf_py as gm
cfg = gm.GmrfConfig(); cfg.x_max, cfg.y_max, cfg.nx, cfg.ny = 2.0, 4.0, 41, 81
b = gm.GmrfBelief(cfg, [gm.WallSegment(1.0, 0.6, 1.0, 3.4)])
b.add_measurement(0.8, 2.0, 0.010); b.update()
mean, var = b.query_mean_variance(np.array([[0.85, 2.05]]))   # (N,), (N,)
```

`GmrfConfig` (all fields, `validate()`), `WallSegment`, `Point2`,
`UpdateReport`, and `GmrfBelief` are bound. `GmrfBelief` provides `reset`,
`add_measurement`, `update`, `discard_pending_measurements`,
`has_pending_measurements`, `num_measurements`, `query_mean`,
`query_mean_variance`, `query_mean_covariance`, `is_representable`,
`representable_mask`, `measurement_noise_variance`, `nx`, `ny`, `dx`, `dy`,
`config`, and the read-only display helper `model_grid()`. `model_grid()` returns
`nx`, `ny`, `node_xy` (node order `j*nx + i`), `active` and `edges`, which
lists the neighbour pairs that are not cut by a wall and join two active
nodes. It is built from `nodePosition`, `isNodeActive` and `is*EdgeCut`.
Queries accept any `(N, 2)` array-like (converted to float64;
`N = 0` is allowed) and return new, owned float64 arrays. Every call releases
the GIL and then takes a per-belief mutex, so one belief is never mutated
concurrently. `invalid_argument` maps to `ValueError`; `logic_error` and
`runtime_error` map to `RuntimeError`. The ctest `gmrf_python_bindings`
(registered when pytest is available) compares the module with the C++
executable `gmrf_py_reference` (`python/reference_case.cpp`), which calls the
C++ API directly.

## Grid and model

Grid: `nx * ny` nodes at `(x_min + i*dx, y_min + j*dy)`, `dx = (x_max-x_min)/(nx-1)`,
`dy = (y_max-y_min)/(ny-1)`; the domain edges are nodes. Node index
`n = j*nx + i` (x fastest). Joint state `x = [c(0..N-1), w_x(0..N-1), w_y(0..N-1)]`.

The posterior maximises `-1/2 sum_k info_k r_k(x)^2` with the source's energy
terms (information = 1/sigma^2):

| Term (flag) | Residual | Std |
| --- | --- | --- |
| prior | `c_n - prior_mean`; `w_x,n - ff_prior_mean_x`; `w_y,n - ff_prior_mean_y` | `prior_sigma`, `ff_prior_sigma` |
| field neighbours (`field_neighbors`) | `c_a - c_b` per uncut grid edge | `sigma_p_per_m * dx^2` (dx also for vertical edges, source formula, kept) |
| flow neighbours (`flow_neighbors`) | `w_x,a - w_x,b` and `w_y,a - w_y,b` per uncut edge | `ff_sigma_p_per_m * dx^2` |
| flow conservation (`flow_conservation`) | per cell: `wx00+wy00-wx10+wy10+wx01-wy01-wx11-wy11` (negated unscaled divergence; ignores walls, as the source) | `ff_sigma_c` |
| flow/field (`flow_field`, nonlinear) | per node: `[x active](c_{i+1,j}-c_{i,j}) w_x + [y active](c_{i,j+1}-c_{i,j}) w_y`, a direction is active if the neighbour exists and the edge is uncut | `sigma_flow` |
| obstacle/flow (`obstacle_flow`) | per node `w_x*cut(+x)`, `w_x*cut(-x)`, `w_y*cut(+y)`, `w_y*cut(-y)`; at the domain boundary the boundary-wall flag replaces the edge | `ff_sigma_o` |
| field measurement | `h^T c - value` (masked bilinear `h`) | `sqrt(sigma_s^2 + sigma_zeta^2 (t - t_k))`, aging only if `enable_measurement_aging` |
| flow measurement (C++ only) | `h^T w_x - value` / `h^T w_y - value` | `ff_sigma_s`, `ff_sigma_zeta` |

Modes:

* Stationary scalar (default, ROS node): `field_neighbors` on, all flow terms
  off, no aging. The normal matrix is then block diagonal, so only the field
  block (size N) is stored and solved. `force_joint_state = true` keeps the
  3N joint state with the flow terms off; the test
  `ScalarJoint.ScalarStateEqualsJointStateWithFlowEnergiesOff` shows equal
  mean, variance and covariance (to 1e-10) and an identical field precision block.
* Joint linear (flow terms on, `flow_field` off): one sparse direct solve.
* Joint nonlinear (`flow_field` on): Gauss-Newton with step halving from the
  current mean (from 0 at construction, as the source), stopping when
  `||step|| < gn_step_tolerance`, the relative cost decrease of an accepted step
  is `< gn_relative_cost_tolerance`, no decrease is possible down to
  `gn_min_step_fraction`, or after `max_gauss_newton_iterations` (then
  `UpdateReport::converged = false`; the improved state is still kept). If the
  line search fails before any step was accepted, the update reports
  `converged = false`, `iterations = 0` and keeps the starting state. The
  reported uncertainty is the inverse Gauss-Newton information `J^T W J`
  re-evaluated at the final accepted point: a local Gaussian (Laplace)
  approximation, not an exact posterior.

Measurement aging is off by default; if enabled, the caller passes times to
`addMeasurement(..., t)` and `update(t)` (`update` time must not precede a
measurement time). Without aging time is ignored and repeated `update()` calls
without new data change nothing. The prior never ages.

## Uncertainty semantics

For query positions with interpolation matrix `H` (masked bilinear weights on
the field block) and posterior precision `Q`:

```
mean       = H m
covariance = H Q^{-1} H^T      (latent field, measurement units^2)
variance   = diag(covariance)
```

Nodal variances are not interpolated. Variances use `||L^{-1} P h||^2` from the
sparse Cholesky factor (AMD ordering); the covariance uses `Q^{-1} H^T` solved
in chunks of 64 columns and is symmetrised. No dense N x N matrix is formed.
These are latent-field quantities; the observation noise `sigma_s^2` is
returned separately (`measurementNoiseVariance()`, service field
`measurement_noise_variance`) and is never added to them.

`prior_sigma` is the std of the independent per-node prior factor. Because the
neighbour terms add information, the prior marginal variance is smaller than
`prior_sigma^2`.

Indicative timing (41 x 81 grid, Release; hardware-dependent): update ~11 ms, marginal
variances at the 861 points of the default visualization grid ~41 ms, 3321
points ~150 ms, full covariance of 100 points ~17 ms.

## Walls

* Walls are segments `(x0, y0, x1, y1)` in the belief frame; order of endpoints
  does not matter; zero-length or non-finite walls are rejected; parts outside
  the domain have no effect.
* Each wall is extended by `wall_endpoint_extension` (default 0.01 m) plus
  `2 * delta` (see tie-break) beyond both endpoints along its own direction, so
  that walls ending exactly on the domain boundary still reach it even with
  `wall_endpoint_extension = 0`. The source extended the scan range by
  0.01 m in y (for x-edges) or x (for y-edges); this is identical for
  axis-aligned walls and slightly different for diagonal walls.
* A grid edge is cut iff its segment intersects a wall. Cut edges remove the
  field/flow neighbour terms of that edge. Interpolation uses the four cell
  corners; a corner is dropped if its node is inactive or the segment from the
  point to the node intersects a wall (the same predicate); the remaining
  weights are renormalised. If nothing remains, the position cannot be
  represented and is rejected (`isRepresentable()` reports this).
* Tie-break: every wall is evaluated as if translated by
  `(delta, 1e-3 * delta)`, `delta = 1e-9 * max(domain extent)`. A node or point
  exactly on a wall therefore belongs to the wall's -x side (for horizontal walls
  the -y side). For axis-aligned walls this is the source's floor rule (a
  crossing at node column k cuts the edge k -> k+1). Because edges and
  interpolation use the same translated wall, a separating wall yields disjoint
  components and no observation leaks across it.
* Walls lying on the domain boundary (within `1e-9 * extent`) cut nothing and
  only set the boundary flags used by `obstacle_flow`.
* Free space: by default every node is active, so disconnected compartments
  keep their prior and their own data. Optional `free_space_seeds` make nodes
  that are not connected to any seed (through uncut edges) inactive: they are
  excluded from interpolation and decoupled (source semantics). This is an
  explicit modelling choice, separate from robot reachability.
* Limitations: walls are infinitely thin; coupling is removed only between grid
  neighbours, so features thinner than the grid spacing are resolved only to
  the grid; the extended ends of non-collinear wall pieces (e.g. L-corners) can
  cut edges up to `wall_endpoint_extension` beyond the corner; points within
  ~`delta` of a wall are assigned by the tie-break; floating-point rounding of
  node and wall coordinates decides exact ties.

## Intentional numerical corrections vs. the source

| # | Source behaviour | Port | Test / reproducer |
| --- | --- | --- | --- |
| 1 | Flow-neighbour cuts written to rows `L+num_edges`, `L+2*num_edges`, but the flow rows are interleaved (`2e`, `2e+1`) and offsets assume `field_neighbors` on: wrong edges lose coupling | both flow rows of the cut edge get zero information, for any energy combination | `ScalarJoint.FlowNeighbourCutRemovesBothFlowRowsOfThatEdgeOnly` |
| 2 | Flow/field term: `updateF` requires both edges uncut (and reads wrong edge ids at the right/top boundary, `edges[num_edges]` out of bounds at the top-right node) while `updateJ` treats the directions independently | residual and Jacobian from the same per-direction definition (the `updateJ` one) | `ScalarJoint.NonlinearFlowModeConvergesWithFiniteUncertainty` |
| 3 | `addLineObstacle` writes out of bounds for walls crossing rows/columns outside the grid (`xyToN(-1, i)`, rows beyond `N_Y`) | segment/edge intersection; parts outside the domain have no effect | `ConfigValidation.RejectsInvalidWalls`, walls extending beyond the domain in `WallsBasic.*` |
| 4 | zero-length walls not checked | rejected | `ConfigValidation.RejectsInvalidWalls` |
| 5 | no lower-bound position checks; upper checks via `M_Assert`, compiled out with `NDEBUG` | finite and in-domain positions required, no clamping, `std::invalid_argument` | `Lifecycle.InvalidMeasurementsAreRejectedWithoutStateChange`, `Lifecycle.InvalidQueriesThrowWithoutPartialResults` |
| 6 | `getEstimate(x, y)` clips to [0, 1] | no clipping, measurement units | `Lifecycle.MeanUsesMeasurementUnitsWithoutClipping` |
| 7 | `getCovariance` interpolates nodal variances | `H Q^{-1} H^T`, full covariance available | `Lifecycle.PriorQueryBeforeAnyData` (diag(cov) = var); dense-oracle tests by the test engineer |
| 8 | dense `3N x 3N` `Sigma` and a custom selected-inverse recursion on the permuted Cholesky factor | sparse Cholesky + chunked solves (recursion not ported) | as 7 |
| 9 | reachability seeded at `nodes[0]` removes disconnected compartments | all nodes active by default; explicit optional `free_space_seeds` | `WallsBasic.DisconnectedCompartmentStaysValidWithoutSeeds`, `WallsBasic.FreeSpaceSeedsDeactivateUnreachableNodes` |
| 10 | convergence test `a \|\| (b) && !first_run` (precedence) and no iteration cap; a stalled line search counts as converged | explicit step / relative-cost tests, iteration cap; a stall before any accepted step is reported as not converged | `ScalarJoint.NonlinearFlowModeConvergesWithFiniteUncertainty`, `ScalarJoint.StallBeforeAnyAcceptedStepIsReportedAsNotConverged` |
| 11 | solver failure throws after the state was modified | transactional update (strong guarantee) | `ScalarJoint.MeasurementAgingIsExplicitAndUsesCallerTime` (rejected update leaves data pending and the belief unchanged) |
| 12 | fully masked interpolation silently gives an all-zero row (normalisation set to 1) | position rejected | `WallsBasic.FreeSpaceSeedsDeactivateUnreachableNodes` |
| 13 | interpolation masking by per-edge min/max cut heuristics | visibility with the same wall predicate as the edge cuts | `WallsBasic.FullSeparatorOnNodeColumnUsesFloorRule`, `WallsBasic.FullSeparatorBlocksInformation` |
| 14 | floor rule per edge orientation: a node on a rising diagonal wall stays connected to both sides | one consistent translated wall, node belongs to the -x side | `WallsBasic.NodeOnRisingDiagonalBelongsToMinusXSide` |
| 15 | boundary flags also set when a sloped wall crosses a row at/after `x_max`; boundary detection by exact `== 0` / `== size` | flags only for walls lying on the domain boundary (with tolerance) | `WallsBasic.WallsOnTheDomainBoundaryCutNothing` |
| 16 | domain origin fixed at (0, 0) | `x_min`, `y_min` configurable (also negative) | `Lifecycle.NonzeroNegativeOriginAndNonSquareCells` |
| 17 | aging always applied (`sigma_zeta` from the config), update times before a measurement give negative noise increments | aging explicit and off by default; such times rejected | `ScalarJoint.MeasurementAgingIsExplicitAndUsesCallerTime` |
| 18 | config parsing: `value_or(0.0)` defaults (e.g. `prior_sigma = 0`), parse errors only printed | documented defaults, typed errors | `TomlLoader.*` |

Preserved on purpose: energy definitions and weights (including
`sigma_p = sigma_p_per_m * dx^2` with dx for both directions,
`ScalarJoint.FieldNeighbourSigmaUsesDxForBothDirections`), the unscaled flow
conservation that ignores walls, bilinear corner weights with renormalisation,
the per-node prior, the zero initial linearisation point, and step halving.
Not ported: source localisation (`sourceHypothesisNeighbor`), likelihoods,
uncertainty gradient, visualisation (SFML), `Scenario`/simulation truth, the
unused `FieldBoundary` energy, and the multi-agent/planning code.
