# Source-parity harness (opt-in)

Compares the port (`scalar_field_belief::gmrf`) with the student's original
GMRF (`reference/UnderwaterMultiagentExploration/gmrf`) on small deterministic
scalar-only cases. Separate CMake project; not part of the normal gmrf build,
ctest, or colcon. Off unless `-DGMRF_BUILD_SOURCE_PARITY=ON`.

```bash
R=<repo>                      # .../src/scalar_field_belief
B=<scratch>/parity            # any directory OUTSIDE the reference tree
env -i HOME=$HOME PATH=/usr/bin:/bin bash -c "
  cmake -S $R/gmrf/tests/source_parity -B $B -DGMRF_BUILD_SOURCE_PARITY=ON \
        -DGMRF_REFERENCE_DIR=$R/reference/UnderwaterMultiagentExploration/gmrf &&
  cmake --build $B -j && ctest --test-dir $B -V"
```

Needs Eigen 3.3+, SFML 2.6 (graphics) and toml++ (3.4 system package tested),
C++20 compiler.

## What is compiled from the reference tree (read-only, unmodified)

`src/gmrf/gmrf.cpp`, `src/viz.cpp`, `src/color_mixer.cpp`, `src/configs.cpp`,
`src/scenario.cpp` with `include/` (viz/scenario/configs are only needed to
link `GMRF`; no window is opened).

## Shims / deviations from the reference build (complete list)

1. `res_loc.h` is generated from the reference's `include/res_loc.h.in` into
   `<build>/generated/` and that directory is put FIRST on the include path
   (the reference CMake writes it into its own source tree; the checked-in
   copy there points to another machine). `RES_LOC` points into the build
   directory; a minimal `configs/gmrf_conf.toml` is written there.
2. `configs.cpp` is compiled with `-include fstream`: it uses `std::ofstream`
   without including `<fstream>` (it compiled against the FetchContent toml++
   v3.3.0, which included it transitively; system toml++ 3.4 does not).
3. System toml++ 3.4 is used instead of FetchContent toml++ v3.3.0 (no
   network). Boost (required by the reference top-level CMake but unused by
   these sources) is not used.
4. Reference sources are compiled with `-w -UNDEBUG` (warnings silenced,
   `M_Assert`/Eigen asserts kept active).
5. The executables `manual.cpp` / `load_map_test.cpp` are not built; the
   driver (`student_driver.cpp`) uses only the public `GMRF` API:
   `GmrfParams(path)` followed by direct overrides of every field,
   a tiny `Scenario` subclass that only provides walls, `addMeasurement(...,
   field)`, `updateEstimates(0.0)`, `getEstimates()`, `getCovariances()`.

## Case set-up and known source behaviour handled by the harness

* Domain origin (0, 0) (the source has no origin), `sigma_zeta = 0`, all flow
  energies off, `flow_field = false` (linear model).
* Wall cases enable `flow_neighbors` in the SOURCE only: its cut handler
  (`updateCorrelationInfomat`) also writes rows `L+num_edges` and
  `L+2*num_edges`, which do not exist without the FlowN block (out-of-range
  sparse access). With `flow_field` off the flow block is decoupled from the
  field, so field marginals are unaffected. The port runs scalar-only.
* The source stores walls in `sf::Vertex` (float); the port is given the same
  float-rounded coordinates (printed by the harness).
* The source seeds reachability at node 0 and isolates every other
  compartment (its nodes keep the isolated prior, measurements there are
  silently dropped). Hence wall cases are compared (a) against the port with
  `free_space_seeds = {(0,0)}` on all nodes (port-inactive nodes are not
  queryable; the harness checks the source holds `prior_mean` /
  `prior_sigma^2` there and the port's nodal mean equals `prior_mean`), and
  (b) against the default (unseeded) port on node 0's compartment only.
  Observations are placed only in node 0's compartment.
* Compared quantities: nodal mean (source `getEstimates()` vs port
  `fieldNodeMean()`) and nodal marginal variance (source
  `covariances[field]`, Takahashi recursion on the Cholesky factor, vs port
  `queryMeanVariance()` at the node positions). Pass: max relative difference
  < 1e-8 (mean relative to max |mean|, variance element-wise).
