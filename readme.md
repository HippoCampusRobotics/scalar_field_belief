# scalar_field_belief

A minimal ROS 2 package for maintaining a simple GPyTorch-based scalar field belief from scalar measurements.

The current version is intentionally basic:
- 2D field belief over physical x/y
- batch query service for posterior mean and variance
- ...

## Installation

This package currently depends on `torch` and `gpytorch`, which should be installed in a workspace-local virtual environment.



## Public API

Topics:
- `ir_measurement` (`scalar_field_interfaces/msg/ScalarMeasurement`)
- `belief/mean_cloud` (`sensor_msgs/msg/PointCloud2`)
- `belief/variance_cloud` (`sensor_msgs/msg/PointCloud2`)

Services:
- `query_scalar_field_belief` (`scalar_field_interfaces/srv/QueryScalarFieldBelief`)
- `reset_scalar_field_belief` (`std_srvs/srv/Trigger`)

## Test launch

```bash
ros2 launch scalar_field_belief test_belief.launch.py
```

Then query e.g.

```bash
ros2 run scalar_field_belief query_belief.py 0.5 1.0
```