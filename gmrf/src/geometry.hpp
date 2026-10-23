// Copyright (C) 2023 Finn Lukas Busch
// Copyright (C) 2026 Nathalie Bauschmann
// SPDX-License-Identifier: GPL-3.0-only
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3 of the License.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

// Internal geometry helpers (not installed).
#ifndef SCALAR_FIELD_BELIEF_GMRF_SRC_GEOMETRY_HPP
#define SCALAR_FIELD_BELIEF_GMRF_SRC_GEOMETRY_HPP

#include <algorithm>

#include "scalar_field_belief/gmrf/config.hpp"

namespace scalar_field_belief::gmrf::detail {

inline double orient(double ax, double ay, double bx, double by, double cx,
                     double cy) {
  return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

/// (cx, cy) inside the closed bounding box of segment a-b.
inline bool inBox(double ax, double ay, double bx, double by, double cx,
                  double cy) {
  return std::min(ax, bx) <= cx && cx <= std::max(ax, bx) &&
         std::min(ay, by) <= cy && cy <= std::max(ay, by);
}

/// Closed segment/segment intersection test (touching counts as intersecting).
/// This single predicate is used for both edge cuts and interpolation
/// visibility, so a separating wall blocks both in the same way.
inline bool segmentsIntersect(double px, double py, double qx, double qy,
                              const WallSegment& w) {
  if (std::max(px, qx) < std::min(w.x0, w.x1) ||
      std::min(px, qx) > std::max(w.x0, w.x1) ||
      std::max(py, qy) < std::min(w.y0, w.y1) ||
      std::min(py, qy) > std::max(w.y0, w.y1)) {
    return false;
  }
  const double o1 = orient(w.x0, w.y0, w.x1, w.y1, px, py);
  const double o2 = orient(w.x0, w.y0, w.x1, w.y1, qx, qy);
  const double o3 = orient(px, py, qx, qy, w.x0, w.y0);
  const double o4 = orient(px, py, qx, qy, w.x1, w.y1);
  if (((o1 > 0.0 && o2 < 0.0) || (o1 < 0.0 && o2 > 0.0)) &&
      ((o3 > 0.0 && o4 < 0.0) || (o3 < 0.0 && o4 > 0.0))) {
    return true;
  }
  if (o1 == 0.0 && inBox(w.x0, w.y0, w.x1, w.y1, px, py)) {
    return true;
  }
  if (o2 == 0.0 && inBox(w.x0, w.y0, w.x1, w.y1, qx, qy)) {
    return true;
  }
  if (o3 == 0.0 && inBox(px, py, qx, qy, w.x0, w.y0)) {
    return true;
  }
  if (o4 == 0.0 && inBox(px, py, qx, qy, w.x1, w.y1)) {
    return true;
  }
  return false;
}

}  // namespace scalar_field_belief::gmrf::detail

#endif  // SCALAR_FIELD_BELIEF_GMRF_SRC_GEOMETRY_HPP
