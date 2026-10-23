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

The numerical core (`scalar_field_belief.belief`, `config`, `models`, `transforms`, `geometry`, `wall_features`) does not depend on ROS 2 and can also be used as an ordinary Python library (see [Standalone Python use](#standalone-python-use-without-ros-2)).

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

This is independent of the ROS installation above. The numerical core only needs NumPy, PyTorch, and GPyTorch (Python >= 3.11). Install it in editable mode into an ordinary virtual environment, without sourcing ROS 2:

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

To load the geometry from a wall-only TOML file instead (path used as given, relative to the current working directory):

```python
from scalar_field_belief.geometry import load_wall_geometry_from_toml

geometry = load_wall_geometry_from_toml(
    '/home/<user>/ros2/src/scalar_field_belief/config/walls/simple_wall_case.toml'
)
belief = ScalarFieldBelief(config, wall_geometry=geometry)
```


## Running the node

After pulling changes that add Python modules or config files (e.g. the wall-aware mode with `config/walls/`), rebuild the package with `colcon build --packages-select scalar_field_belief`, since an existing (symlink) install does not contain the new modules or files.

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
- `wall_aware`: requires a wall geometry with at least one wall whose bounds match `x_min/x_max/y_min/y_max`. The node refuses to start if the geometry file is missing, invalid, or contains no walls; there is no fallback to baseline.

### Parameters

| Parameter | Default | Unit | Meaning |
| --- | --- | --- | --- |
| `model_variant` | `baseline` | - | `baseline` or `wall_aware` |
| `wall_beta` | `1.0` | 1/m | Fixed sharpness of the sign change across the wall line |
| `wall_rho` | `0.35` | m | Fixed decay length of the feature away from the segment |
| `init_wall_lengthscale` | `1.0` | dimensionless | Initial lengthscale of every wall feature |
| `wall_geometry_file` | `''` | - | Node only: wall geometry TOML (only read in `wall_aware` mode) |

`wall_beta` and `wall_rho` are fixed; they are not learned. Like the spatial lengthscales, the wall lengthscales are optimized only if `optimize_hyperparameters: true`; otherwise they stay at `init_wall_lengthscale`. Use float literals (e.g. `1.0`, not `1`) for these parameters in YAML files.

`fitted_hyperparameters_physical()` additionally reports `wall_feature_lengthscale_0`, `wall_feature_lengthscale_1`, ... in wall-aware mode. These are dimensionless and are not scaled by the domain size.

### Geometry files

Wall-only TOML schema (source sections, if present, are ignored):

```toml
[general]
name = "simple_wall_case"   # optional, defaults to the file name stem

[bounds]
x_min = 0.0
x_max = 2.0
y_min = 0.0
y_max = 4.0

[objects.walls]
wall_vertices = [[1.0, 0.6, 1.0, 3.4], [1.0, 2.1, 1.5, 2.1]]  # [x1, y1, x2, y2]
```

Path rules for `wall_geometry_file` in the node:

- absolute paths are used as given,
- relative paths are resolved against the installed package share directory (`$(ros2 pkg prefix --share scalar_field_belief)`), e.g. `config/walls/simple_wall_case.toml`. Files under `config/` are installed by `colcon build`; rebuild after adding or changing them.

The pure Python loader `load_wall_geometry_from_toml(path)` does not perform this lookup; it uses the path it is given.

### Frames

Walls, measurements, and queries are all interpreted in the configured `frame_id` (usually `map`). The node performs no TF transformation, so the geometry file must already be expressed in that frame.

### Launching the wall-aware configuration

Reuse the `belief_params` argument of the existing launch file:

```bash
ros2 launch scalar_field_belief test_belief.launch.py vehicle_name:=uuv00 \
  belief_params:=$(ros2 pkg prefix --share scalar_field_belief)/config/belief_params_wall_aware.yaml
```

`config/belief_params_wall_aware.yaml` uses the same settings as `belief_params_default.yaml`, except `model_variant: wall_aware` and `wall_geometry_file: config/walls/simple_wall_case.toml` (bounds `0..2 x 0..4`).


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
ros2 launch scalar_field_sim top_field_sim_stack.launch.py vehicle_name:=uuv00
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
