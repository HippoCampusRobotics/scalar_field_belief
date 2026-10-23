# Provenance

## Original work

* Author: Finn Lukas Busch (finn.lukas.busch@gmail.com), "Created by Finn Lukas
  Busch" headers in all source files used here.
* Repository: UnderwaterMultiagentExploration (master's thesis code,
  "Stochastic Field Exploration of cluttered underwater Environments:
  Multi-Robot Informative Path Planning under Communication Constraints"),
  subdirectory `gmrf/`. Local read-only copy:
  `reference/UnderwaterMultiagentExploration/gmrf/` (git-ignored, not modified,
  not built).
* References given by the original code:
  * `gmrf/include/gmrf/gmrf.h`: "Implementation of
    https://link.springer.com/content/pdf/10.1007/s10514-015-9437-0.pdf"
    (Autonomous Robots), covariance via upper-triangular Cholesky
    (https://mathoverflow.net/questions/163414), reference implementation
    https://github.com/MAPIRlab/gmrf_gas_mapping.
  * Repository README: "Joint estimation of gas and wind maps for fast-response
    applications", https://doi.org/10.1016/j.apm.2020.06.026, extended to
    continuous measurement locations.

## License

No LICENSE/COPYING file and no license statement were found anywhere in the
local copy `reference/UnderwaterMultiagentExploration/` (searched
case-insensitively for `*licen*`/`copying*`; the README and
`readme_install.md` contain none). The scalar_field_belief package declares
GPLv2 in `package.xml`. Redistribution terms for code derived from the
original therefore need to be clarified with the original author before
publishing; attribution is kept in the ported files.

## Mapping of source components

| Source (`gmrf/`) | Port (`gmrf/` in scalar_field_belief) |
| --- | --- |
| `include/gmrf_configs.h` `GmrfParams` (model fields) | `include/scalar_field_belief/gmrf/config.hpp` `GmrfConfig` (+ domain origin, aging flag, `force_joint_state`, wall extension, solver settings) |
| `src/configs.cpp` `GmrfParams(std::string)` | `src/toml_loader.cpp` `loadGmrfConfigToml` / `parseGmrfConfigToml` |
| `resources/configs/gmrf_conf.toml` layout | same layout accepted; test copy of the layout in `tests/data/gmrf_conf_source_layout.toml` |
| `src/scenario.cpp` `ConfigFileScenario` (`[objects.walls] wall_vertices` only) | `src/toml_loader.cpp` `loadWallsToml` / `parseWallsToml` (simulation truth, sources, flow fields not ported) |
| `GMRF::GMRF`, `initializeNodes`, `initializeEdges`, `nodepairToL`, `xyToN` | `GmrfBelief::GmrfBelief`, `buildGeometry`, `nodeIndex`, `hEdge`/`vEdge` (`src/gmrf_belief.cpp`) |
| `GMRF::addLineObstacle`, `updateCorrelationInfomat` | `GmrfBelief::buildGeometry` (edge cuts, boundary flags) + `src/geometry.hpp` `segmentsIntersect` |
| `GMRF::determineReachableNodes`, `checkChildren` | optional free-space mask in `buildGeometry` (`free_space_seeds`) |
| `GMRF::getShapeFacs` | `GmrfBelief::tryInterpolate` (+ `visible`) |
| `FieldN_EnergyFunc`, `FlowN_EnergyFunc`, `FlowC_EnergyFunc`, `FlowField_EnergyFunc`, `ObstacleFlow_EnergyFunc`, `initializePrior`, `addMeasurement`, `updateMeasurementInfomat`, `updateF` | `GmrfBelief::assemble` (sparse J, targets, information) and `GmrfBelief::addMeasurement` / `addFlowMeasurement` |
| `GMRF::updateEstimates` (Gauss-Newton loop) | `GmrfBelief::solve` (direct solve for linear modes, Gauss-Newton with step halving otherwise), `GmrfBelief::update` |
| `GMRF::updateEstimates` covariance recursion, `getCovariance` | `GmrfBelief::queryMeanVariance`, `queryMeanCovariance` (`H Q^{-1} H^T` via sparse Cholesky) |
| `GMRF::getEstimate`, `getEstimates`, `getFlowXMap`/`getFlowYMap` | `GmrfBelief::queryMean`, `fieldNodeMean`, `flowNodeMean` |
| `FieldBoundary_EnergyFunc` (disabled in the source) | not ported |
| `sourceHypothesisNeighbor`, `getLikelihood`, `getLogLikelihood`, `uncertaintyGradient`, `isReachable`, visualisation (`gmrf_viz.h`, `viz.cpp`, colour maps, SFML), `executables/` | not ported (out of scope) |

The ROS node (`ros/gmrf_belief_node.cpp`), launch file, parameter file, CMake
packaging, examples and tests are new. Model differences and corrections are
listed in README.md ("Intentional numerical corrections vs. the source").

Note for comparisons with the original: the source passes wall coordinates
through `sf::Vertex` (single precision floats) and assumes a domain origin of
(0, 0).
