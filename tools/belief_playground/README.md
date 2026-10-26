# Belief playground (GP / GMRF)

Interactive tester for the scalar-field beliefs. Click a position on a map:
the tool samples the scenario there, adds one scalar observation to every
selected backend, and shows each backend's posterior estimate and standard
deviation next to the ground truth. It is inspired by Finn Busch's SFML GMRF
viewer, but it is an independent implementation. It adds scalar observations
only (no flow measurements, no source localisation).

* **GP baseline** (`gp`) and **GP wall-aware** (`gp_wall_aware`):
  `scalar_field_belief.belief.ScalarFieldBelief`, called directly in Python.
  `gp_wall_aware` only accepts a config with `model_variant: wall_aware`.
* **GMRF**: the C++ `scalar_field_belief::gmrf::GmrfBelief`, called through the
  pybind11 module `scalar_field_belief_gmrf_py` (`gmrf/python/`).
* **Scenario / truth**: `scalar_field_sim.config.load_field_from_toml` and
  `field.evaluate()`.
* **GUI**: PySide6 + PyQtGraph. `belief_playground` is a plain Qt
  application: it does not start rclpy, create ROS nodes, or need the
  simulator or estimator nodes. The separate [live viewer](#live-viewer)
  (`belief_live_viewer`) uses the same window parts to show a running
  estimator node.

```text
tools/belief_playground/
  belief_playground/scenario.py   scenario + truth (controller only), display grid
  belief_playground/backends.py   GPAdapter, GmrfAdapter (reset/add_measurement/query[/model_grid])
  belief_playground/session.py    observation log, worker thread, save/load (no GUI)
  belief_playground/gui.py        Qt window
  belief_playground/launcher.py   scenario/config lookup, installed entry point
  belief_playground/__main__.py   command line
  belief_playground/live_data.py  live viewer: shared state, display resampling (no Qt, no ROS)
  belief_playground/live_ros.py   live viewer: rclpy subscriber node (no Qt)
  belief_playground/live_gui.py   live viewer: Qt window (no rclpy)
  belief_playground/live_main.py  live viewer: command line
  scripts/belief_playground       installed executable (ros2 run)
  scripts/belief_live_viewer      installed live viewer executable (ros2 run)
  tests/                          pytest suite, gui_smoke.py
```

## One-time setup

* System packages:
  `sudo apt install pybind11-dev python3-dev libeigen3-dev`.
* Python packages, installed in the Python environment that `python3` refers
  to when you run the tool. The installed executable uses
  `#!/usr/bin/env python3`, the same as the GP node. You need the GP
  dependencies (`torch`, `gpytorch`, as for the GP node) plus the GUI:

  ```bash
  python3 -m pip install PySide6-Essentials pyqtgraph
  ```

  Do not install them into the system Python. Create the venv from
  `/usr/bin/python3` with `--system-site-packages`.
* The GMRF module is compiled by the workspace build for the interpreter that
  ament uses (`/usr/bin/python3`, Python 3.12). It therefore works from any
  Python 3.12 venv based on that interpreter. With a different Python
  version, the import fails with a message that names both interpreters.

## Build

Use the normal workspace build:

```bash
cd ~/ros2
colcon build --symlink-install --packages-select scalar_field_belief scalar_field_sim \
  --cmake-args -DSCALAR_FIELD_BELIEF_BUILD_PLAYGROUND=ON
```

`SCALAR_FIELD_BELIEF_BUILD_PLAYGROUND` is ON by default. It builds the GMRF
core and its pybind11 Python module (needs a C++ toolchain, `pybind11-dev`,
`python3-dev` and Eigen), and installs the module, the `belief_playground`
Python package, the `belief_playground` and `belief_live_viewer` executables
and `config/belief_params_gmrf.yaml`. `-DSCALAR_FIELD_BELIEF_BUILD_PLAYGROUND=OFF`
skips all of this. CMake caches the option: an existing build directory keeps
its previous value until you pass `-D...` explicitly (as above) or rebuild
with `--cmake-clean-cache`. The scenarios come from the
installed `scalar_field_sim` package (`share/scalar_field_sim/scenarios`).
`-DSCALAR_FIELD_BELIEF_BUILD_GMRF=ON` additionally builds the GMRF ROS node,
as before. After the first build, open a new shell or source
`install/local_setup.zsh`.

> [!NOTE]
> Set the option to OFF when building on the robot or on any other machine
> that does not run the GUI tools. This avoids the extra build dependencies
> and the GMRF/pybind11 compile:
>
> ```bash
> colcon build --cmake-args -DSCALAR_FIELD_BELIEF_BUILD_PLAYGROUND=OFF
> ```

## Launch

From any directory, in your normal shell (ROS sourced, venv active):

```bash
ros2 run scalar_field_belief belief_playground
ros2 run scalar_field_belief belief_playground --scenario 01_single_wall
ros2 run scalar_field_belief belief_playground --scenario 01_single_wall --hide-measurements
ros2 run scalar_field_belief belief_playground --list-scenarios
ros2 run scalar_field_belief belief_playground --scenario ~/my_case.toml --backends gmrf
```

`python3 -m belief_playground [options]` is the direct Python entry point. It
has the same options. In a sourced workspace it uses the installed package; it
also works from a source checkout if `scalar_field_belief`, `scalar_field_sim`
and the GMRF module are importable. When run from the checkout, the default
configs are the checkout's files.

| Option | Default | Meaning |
| --- | --- | --- |
| `--scenario NAME\|FILE` | `01_single_wall` | scenario TOML path, or the name of an installed scenario (`01_single_wall`, or `tests/fields/01_single_wall` to disambiguate) |
| `--list-scenarios` | | list installed scenario names and exit |
| `--backends` | `gp,gp_wall_aware,gmrf` | ordered, comma-separated subset of `gp`, `gp_wall_aware`, `gmrf`; one column each |
| `--gp-config YAML` | installed `config/belief_params_default.yaml` | GP ROS parameter file for `gp` |
| `--gp-wall-aware-config YAML` | installed `config/belief_params_wall_aware.yaml` | GP ROS parameter file for `gp_wall_aware` (`model_variant: wall_aware`) |
| `--gmrf-config YAML` | installed `config/belief_params_gmrf.yaml` | GMRF ROS parameter file |
| `--grid-step M` | `0.05` | display/evaluation grid spacing [m] (not the GMRF model grid) |
| `--noise-std S` | `0` (off) | observation noise std for sampled values, in the scenario unit |
| `--seed N` | scenario seed, else 0 | noise generator seed |
| `--load CSV` | | load and replay a saved session at start |
| `--out-dir DIR` | `./belief_playground_sessions` | start directory of the save/export dialogs |
| `--hide-measurements` | | start with measurement markers and labels hidden (display only; toolbar checkbox) |
| `--mean-range MIN MAX` | shared auto | fixed colour range for truth and all estimates |
| `--std-max MAX` | shared auto | fixed colour range `[0, MAX]` for all standard deviations |

Unknown scenario names report close matches. If a name exists in several
directories, the error lists the candidates.

Backend configuration: each adapter reads its own ROS parameter YAML
(`/**: ros__parameters:`) with PyYAML. The domain bounds are always taken from
the scenario, and so are the known walls (`belief_geometry`, `scenario_path`,
and `walls` in the YAML are ignored). The wall-aware GP gets the scenario walls as its
`WallGeometry`. The GMRF keeps `nx`/`ny` from its YAML, so `dx`/`dy` follow the
scenario bounds. All overrides are recorded in the configuration snapshot
saved with each session. If a required parameter is missing, that backend is
shown as unavailable and the others keep working.

## Window

```text
Ground truth        | GP baseline - estimate           | GP wall-aware - estimate           | GMRF - estimate (+ model grid)   | ...
(reserved, empty)   | GP baseline - standard deviation | GP wall-aware - standard deviation | GMRF - standard deviation        | ...
```

* There is one column for the truth plus one column per selected backend, in
  `--backends` order. Each standard deviation panel (sqrt of the latent
  predictive variance, computed for display only) sits under that backend's
  estimate. The bottom-left cell is intentionally left empty.
* The maps are aspect-correct and always show the whole domain. They are
  packed with 7 px gaps and centred, with no axes, colour bars or plot
  controls. The window is sized from the available screen area and can be
  maximised. With many backends the maps keep a minimum size and the area
  scrolls horizontally.
* Images use the display grid's cell centres (row = y, y up). The scenario
  extent is mapped exactly, with no transpose, flip or half-cell offset.
* Colour scales are shared but not shown: viridis for the truth and all
  estimates (one common range), magma for all standard deviations (one
  common range starting at 0). Use `--mean-range`/`--std-max` for fixed
  ranges; the current ranges are listed in the status tooltip. Cells a
  backend cannot represent (e.g. GMRF cells hidden behind walls) and maps that
  are not available yet are grey, never zero.
* Walls (white with a black outline) are drawn on every map. The GMRF estimate
  panel also shows the GMRF's own model graph (thin translucent lines): node
  spacing from the model, without edges cut by walls or touching inactive
  nodes. The geometry is read once from the module (`model_grid()`).
* Measurements are numbered red circles on every map. The **Show measurement
  positions** checkbox at the right end of the toolbar shows or hides them
  immediately. `--hide-measurements` starts with it unticked. This only
  changes the display: observations, noise, fitting and queries are
  unchanged. The saved metadata records the current setting
  (`show_measurements`).

### Interaction

* **Left-click** any map inside the domain to add one sampled observation:
  the scenario truth at that point, plus optional noise, clipped to the
  scenario `clip_min/clip_max` (like `field.sample()`). Clicks on titles,
  gaps, the empty cell, outside the domain, with other buttons, and drags are
  ignored.
* Toolbar and shortcuts: **Undo** (Ctrl+Z), **Reset** (Ctrl+R, re-seeds the
  noise generator), **Save session** (Ctrl+S), **Load session** (Ctrl+O),
  **Export image** (Ctrl+E, PNG or JPEG of the window). Quit with Ctrl+Q, by
  closing the window, or with Ctrl+C in the terminal. An image alone is not a
  saved experiment.
* **Save session** opens a save dialog in `--out-dir`. If that directory does
  not exist, the dialog starts in its nearest existing parent; opening the
  dialog creates nothing. The suggested name is `session_<time>.csv`, and
  `.csv` is added if missing. The metadata is written next to the CSV with
  the same name (`<name>.json`). If the `.csv` or the `.json` already exists,
  you are asked before anything is overwritten. Cancel writes nothing.
  **Export image** works the same way (`figure_<time>.png`; `.png` is added
  unless the name ends in `.png`/`.jpg`). **Load session** starts in the
  directory of the last saved session.
* Status bar: observation count, then for each backend:
  * its state and assimilated/total count;
  * timings and RMSE against the truth (on the cells where all shown estimates
    are finite);
  * then the last message.

  Hover over the status bar for details. A backend that rejected an
  observation is marked **STOPPED n=k/N (#m failed)** in the status bar and
  in its panel titles. Its maps keep its last good result. The GP shows
  *waiting* until its first configured fit.
* Timings are wall-clock times of the most recent job in the worker thread
  (details in the status tooltip):
  * `assimilate X ms`: the job that added the newest observation.
    * GP: storing it (plus target standardisation) and, when `refit_policy`
      says so, hyperparameter training; otherwise only `set_train_data`.
      The GP posterior solve is **not** included here. GPyTorch builds it
      lazily in the next query, so it counts as grid query.
    * GMRF: `add_measurement` plus the sparse `update()` solve (posterior
      mean).
  * `reset+replay K obs X ms`: after Undo/Reset/Load, the backend reset plus
    re-assimilation of all K stored observations.
  * `grid query X ms (N cells)`: one query on all N display cells.
    * GP: the posterior solve for the current training set (it grows roughly
      cubically with the number of observations), plus mean and variance.
    * GMRF: representability mask plus mean and marginal variances (sparse
      solves).
    * It scales with `--grid-step` (N), not only with the model size.

Responsiveness: clicks only append to the log. One worker thread runs all
backend updates and grid queries in order. The GUI polls results every 100 ms
on the Qt main thread. Reset, undo and load start a new generation, and
results from older generations are discarded. Closing the window stops the
worker.

## Sessions

`session_*.csv` columns: `index, x, y, value, latent, noise, noise_std,
source, clipped` (full float precision). `value` is exactly what every backend
received. Undo and load replay the stored values without drawing new noise.

`session_*.json` records:

* the scenario path, the sha256 and full text of the scenario file and its
  referenced `geometry_file` and `field_file`;
* the domain, walls and units;
* the noise settings (`seed_in_use`, `pending_seed`, draws);
* the display grid;
* each backend's effective configuration (YAML text + sha256) and status at
  save time;
* the display state (colour ranges, `show_measurements`);
* Git state of `scalar_field_belief` and `scalar_field_sim`;
* package versions, including PySide6 and pyqtgraph.

Loading warns if the scenario content or a backend's effective configuration
differs from the saved one, and replays the observations anyway.
`session.py` has no GUI dependency and can be used headlessly:

```python
from belief_playground.scenario import Scenario, make_display_grid
from belief_playground.backends import make_backends
from belief_playground.session import Session
sc = Scenario('.../01_single_wall.toml')
s = Session(sc, make_backends(['gp', 'gp_wall_aware', 'gmrf'], sc.domain),
            make_display_grid(sc.domain.x_range, sc.domain.y_range, 0.05))
s.add_sample(0.5, 1.0); s.wait_idle(); _, status, _, _ = s.poll(); s.close()
```

## Live viewer

`belief_live_viewer` shows one running field-belief node (GP node or GMRF
node) live: its posterior mean and standard deviation, optionally the
scenario ground truth, the travelled trajectory, the current position, the
measurement positions and the walls. It reuses the playground's map panels,
layout, colour maps and style, but it never adds observations: clicks on the
maps do nothing. Unlike `belief_playground` it starts rclpy and creates one
node (`belief_live_viewer`) that only subscribes.

It is installed together with the playground
(`SCALAR_FIELD_BELIEF_BUILD_PLAYGROUND`, ON by default, see [Build](#build)) and needs
the same Python packages (PySide6, pyqtgraph). Run it next to the estimator,
in the same ROS domain, in the vehicle namespace:

```bash
# Simulation: domain, walls and ground truth come from the running system
ros2 launch ipp simulation.launch.py
ros2 run scalar_field_belief belief_live_viewer --ros-args -r __ns:=/bluerov00
# Explicit scenario instead of discovery (top_field_sim_stack.launch.py, uuv00)
ros2 run scalar_field_belief belief_live_viewer --scenario simple_field_1 \
  --ros-args -r __ns:=/uuv00
# Hardware: pose from vision_pose_cov, walls from a geometry file, no truth
ros2 run scalar_field_belief belief_live_viewer --geometry my_geometry.toml \
  --pose-topic vision_pose_cov --pose-type pose_cov --ros-args -r __ns:=/uuv00
# Bounds only (no walls), GMRF node
ros2 run scalar_field_belief belief_live_viewer --bounds 0 2 0 4 \
  --belief-node gmrf_belief --ros-args -r __ns:=/uuv00
```

`python3 -m belief_playground.live_main [options] [--ros-args ...]` is the
direct Python entry point, with the same options. The viewer uses wall-clock
time only, so it needs no `use_sim_time`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--scenario NAME\|FILE` | | scenario TOML or installed scenario name (as for the playground): domain, walls, unit and ground truth |
| `--geometry FILE` | | geometry or scenario TOML: domain and walls only, no truth |
| `--bounds XMIN XMAX YMIN YMAX` | | domain [m] only, no walls, no truth |
| `--no-truth` | | do not show the scenario truth |
| `--field-server-node` | `/field_server` | discovery: node whose `scenario_path` gives the domain and truth |
| `--belief-node` | `scalar_field_belief` | discovery: belief node for bounds, walls and `frame_id` (relative names resolve in the viewer namespace; GMRF node: `gmrf_belief`) |
| `--param-timeout S` | `15` | discovery: wait at most S s for these nodes |
| `--mean-topic` | `belief/mean_cloud` | posterior mean cloud |
| `--variance-topic` | `belief/variance_cloud` | posterior variance cloud |
| `--measurement-topic` | `ir_measurement` | measurements; `''` disables |
| `--pose-topic` | `odometry` | vehicle pose; `''` disables |
| `--pose-type` | `odometry` | `odometry` (`nav_msgs/Odometry`), `pose` (`geometry_msgs/PoseStamped`) or `pose_cov` (`geometry_msgs/PoseWithCovarianceStamped`) |
| `--frame-id` | belief node's `frame_id`, else `map` | expected frame of all inputs |
| `--method-label` | detected | override the automatically detected method name (no lookup) |
| `--grid-step M` | `0.02` | display grid spacing [m] |
| `--mean-range MIN MAX` | shared auto | fixed colour range for truth and estimate |
| `--std-max MAX` | auto | fixed colour range `[0, MAX]` for the standard deviation |
| `--trajectory-min-step M` | `0.01` | store a trajectory point only after moving at least M [m] |
| `--stale-after S` | `10` | mark the pose stream as stale after S s without pose messages |
| `--hide-measurements` | | start with the measurement positions hidden |
| `--hide-trajectory` | | start with the trajectory hidden |

At most one of `--scenario`, `--geometry` and `--bounds` may be given. With
one of them nothing is discovered (`--frame-id` defaults to `map`).

Domain discovery (no `--scenario`, `--geometry` or `--bounds`): before the
window opens, the viewer reads ROS parameters of the running nodes, waiting
up to `--param-timeout` seconds and printing what it waits for and what it
found:

1. `--field-server-node` (`/field_server`, started by the scalar_field_sim
   launch files without namespace): a non-empty `scenario_path` is resolved
   like the scalar_field_sim nodes do (absolute, or relative to the installed
   scalar_field_sim share) and gives the domain, walls, unit and ground
   truth, as `--scenario` would. If that file cannot be loaded where the
   viewer runs (other install, missing or malformed file), the viewer warns
   and continues with 2.
2. Otherwise `--belief-node`: `x_min`, `x_max`, `y_min`, `y_max` give the
   domain, without truth. Walls follow that node's own rule: the GP node
   uses `belief_geometry` (`scenario` = the file in its `scenario_path`)
   only with `model_variant: wall_aware`; the GMRF node uses its flat
   `walls` list plus the `belief_geometry` file, if set (read like the GMRF
   reads it: `[bounds]` optional, one level of `geometry_file`). If the
   walls cannot be loaded, the bounds are shown without walls and a warning
   is printed.

The frame is `--frame-id` if given, else the belief node's `frame_id`, else
the field-server scenario's `frame_id`, else `map`; a warning says when the
belief node was not reached and the frame is such a fallback. If the belief
node's bounds differ from a field-server domain, a warning says so (belief
values outside the shown domain are cut off).

Both nodes are asked at the same time. Discovery ends as soon as the answers
are complete; once one node has given something usable, the other one gets
at most 2 s more, including requests that are still pending. It never takes
longer than `--param-timeout`, and Ctrl+C ends it at once (exit code 130).
If neither node gives a domain, the viewer exits with an error that lists
what it tried; then pass `--scenario`, `--geometry` or `--bounds` (on
hardware without field server the belief node is usually enough). The
status tooltip shows where the domain and the frame came from, and any
warnings.

Subscriptions (relative names resolve in the node namespace, set with
`--ros-args -r __ns:=/<vehicle>`):

| Topic | Type | QoS |
| --- | --- | --- |
| `belief/mean_cloud` | `sensor_msgs/PointCloud2`, fields `x`, `y`, `intensity` | reliable, transient local, depth 1 (receives the latched cloud) |
| `belief/variance_cloud` | `sensor_msgs/PointCloud2`, fields `x`, `y`, `intensity` | as above |
| `ir_measurement` | `scalar_field_interfaces/ScalarMeasurement` | reliable, depth 100 |
| `odometry` | see `--pose-type` | best effort, depth 10 (matches reliable and best-effort publishers) |

Values are read from the `intensity` field only (posterior mean / variance;
any numeric field type, either byte order). Clouds that are not a regular
lattice are rejected with an error. A mean cloud without it, for example from a GP node that publishes only the
packed `rgb` colours, is reported as an error in the status bar and is not
decoded from the colours.

Frames and coordinates:

* All inputs must be in `--frame-id`; an empty frame is accepted as
  `--frame-id`. Messages in other frames are ignored and counted in the
  status bar. There is no TF.
* The estimator grid (the cloud points) is shown on a cell-centred display
  grid (`--grid-step`, row = y, y up, as in the playground). Each display
  cell takes the value of the nearest estimator node per axis, without
  interpolation, so every estimator node is drawn as a block of display
  cells around it (up to half a node spacing per axis). A value can
  therefore appear up to half a visualization step on the other side of a
  wall. Cells farther than half a node spacing from the estimator grid and
  cells whose nearest node is missing from the cloud (e.g. GMRF nodes that
  are not representable) are grey.
* The standard deviation is sqrt(variance), for display only. Negative
  variances are shown as 0 and counted in the status tooltip.

Window:

```text
Ground truth        | <method> - estimate
(reserved, empty)   | <method> - standard deviation
```

* The method name in the panel titles, the window title and at the start
  of the status line is detected from the node that actually publishes the
  mean cloud (found in the ROS graph, so it also works with remapped topics
  and with `--scenario`/`--geometry`/`--bounds`): a node with
  `model_variant` is shown as *GP baseline* or *GP wall-aware* (with its
  kernel), one with `nx`/`ny` as *GMRF* (with its grid), any other node by
  its name. Before a publisher exists, `--belief-node` is asked. The lookup
  runs on the ROS thread, never delays the window, is retried every 2 s
  until the node answers, and is repeated when a different node starts
  publishing. Until then the title says *Belief* and the status line says
  what is being queried. The status line also shows the node's full name
  and notes several publishers. `--method-label` sets a fixed name instead
  and skips the lookup.
* The truth comes from a scenario: `--scenario`, or the field server's
  `scenario_path` in discovery mode (not with `--no-truth`). Otherwise its
  panel is grey and marked *not available*.
* Colour ranges as in the playground: one shared range for truth and
  estimate, the standard deviation from 0; `--mean-range`/`--std-max` fix
  them.
* Every map shows the walls, the measurement positions (red circles, without
  numbers), the travelled trajectory (thin line; at most 200 000 points, the
  oldest are dropped) and the current position (filled yellow dot).
* Toolbar: **Clear trajectory**, **Clear measurements** (removes the shown
  positions only, e.g. after an estimator reset), **Export image** (Ctrl+E,
  PNG or JPEG of the window), **Show measurement positions**, **Show
  trajectory**. Quit with Ctrl+Q, by closing the window, or with Ctrl+C in
  the terminal.
* Status bar: for each stream (mean, variance, pose, measurements) the
  message count and the time since the last message (wall clock, so sim time
  does not matter), `waiting for <topic>` before the first message, ignored
  frames and decode errors, and a note if the mean and variance clouds have
  different points. Only the pose stream is marked *stale* (after
  `--stale-after` seconds): the estimators publish their clouds only after
  new measurements, so old clouds are normal. If the ROS thread stops, the
  status bar starts with **ROS STOPPED** and the reason. The window polls
  every 100 ms and redraws the maps only when new data arrived.

## Tests

Run the GUI tests on a virtual display only (`xvfb`). Unset
`WAYLAND_DISPLAY`, otherwise Qt opens on the desktop. Without these settings
the GUI tests are skipped.

```bash
cd /tmp
env -u WAYLAND_DISPLAY QT_QPA_PLATFORM=xcb BELIEF_PLAYGROUND_GUI_SMOKE=1 \
  xvfb-run -a -s "-screen 0 1920x1080x24" \
  python3 -m pytest -q -p no:cacheprovider ~/ros2/src/scalar_field_belief/tools/belief_playground/tests
# Real baseline GP + wall-aware GP + GMRF window: click, wait, undo, save, reset, load, close
env -u WAYLAND_DISPLAY QT_QPA_PLATFORM=xcb xvfb-run -a -s "-screen 0 1920x1080x24" \
  python3 ~/ros2/src/scalar_field_belief/tools/belief_playground/tests/gui_smoke.py \
  --out-dir /tmp/belief_playground_smoke --screenshots /tmp/belief_playground_smoke
```

The installed-launcher test runs when `ros2` and an installed
`belief_playground` are available. `test_live_ros.py` needs a sourced ROS
workspace (rclpy, `nav_msgs`, `scalar_field_interfaces`) and is skipped
otherwise; it runs its publishers and the viewer node in-process under a
random namespace. GMRF binding tests: see `gmrf/README.md`.

## Troubleshooting

* `cannot import the Qt frontend`: install PySide6-Essentials and pyqtgraph
  for the interpreter named in the message (usually: activate your venv).
* `backend gmrf unavailable: ModuleNotFoundError ... scalar_field_belief_gmrf_py`:
  rebuild the workspace with the playground enabled (the default) and source
  it. A Python version mismatch is reported with both interpreters.
* `unknown scenario`: see `--list-scenarios`. Rebuild `scalar_field_sim` after
  adding scenarios, or pass a TOML path.
* GP panels stay grey with *waiting*: the GP has not reached its first
  configured refit yet (`refit_policy`/`refit_every_k`).
* GMRF **STOPPED** at observation #k: the GMRF rejects positions it cannot
  represent (hidden from all four grid corners by walls, or on inactive
  nodes). The GP continues. Undo that observation (or Reset) to replay and
  recover.
* With very few observations the GP's standard deviation can be much larger
  than the GMRF's (target standardisation). It then dominates the shared std
  range; use `--std-max` to compare details.
