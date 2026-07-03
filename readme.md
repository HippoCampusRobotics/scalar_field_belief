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

## Running the node

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
