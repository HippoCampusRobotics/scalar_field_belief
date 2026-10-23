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

"""PySide6 + PyQtGraph frontend.

Only this module touches Qt. All widgets and graphics items are created and
updated on the Qt main thread (signal handlers and a QTimer that polls the
Session); backend work runs in the Session worker thread.

Layout: two rows, one column for the ground truth plus one column per
backend, in the session's backend order::

    Ground truth        | <A> - estimate           | <B> - estimate           | ...
    (reserved, empty)   | <A> - standard deviation | <B> - standard deviation | ...

Images use row-major arrays (row = y, column = x) from `DisplayGrid`, mapped
with `ImageItem.setRect(x_min, y_min, width, height)` into a ViewBox whose y
axis points up, so pixel (j, i) covers the display cell centred at
(xs[i], ys[j]).
"""

from __future__ import annotations

import html
import math
import signal
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from belief_playground.session import Session, rmse_on_common_grid

MEAN_CMAP = 'viridis'
STD_CMAP = 'magma'
# NaN / not representable cells; distinct from both colour maps' ends.
INVALID_RGBA = (96, 96, 96, 255)

WINDOW_BG = '#161616'
PANEL_BG = '#000000'
TEXT_COLOR = '#d6d6d6'
GAP = 7  # [px] between neighbouring panels
MARGIN = 8  # [px] outer margin
TITLE_H = 16  # [px] title strip above each map
MIN_PANEL_W = 220  # [px] below this, scroll horizontally instead of shrinking

WALL_OUTLINE_PEN = (0, 0, 0, 255), 4.0
WALL_PEN = (255, 255, 255, 255), 2.0
MODEL_GRID_PEN = (255, 255, 255, 50), 1.0
MARKER_COLOR = '#ff4040'

Z_IMAGE, Z_MODEL_GRID, Z_WALLS, Z_MARKERS = 0, 10, 20, 30


def variance_to_std(var, rel_tol: float = 1e-9) -> tuple[np.ndarray, int]:
    """Display-only std = sqrt(variance).

    Tiny negative variances (round-off, |v| <= rel_tol * max|var|) are shown
    as 0; the number of cells more negative than that is returned so it can
    be reported. NaN (not representable) stays NaN.
    """
    var = np.asarray(var, dtype=float)
    finite = var[np.isfinite(var)]
    scale = float(np.max(np.abs(finite))) if finite.size else 0.0
    n_bad = int(np.sum(var < -rel_tol * scale))
    with np.errstate(invalid='ignore'):
        std = np.sqrt(np.clip(var, 0.0, None))
    return std, n_bad


def _finite_range(arrays) -> tuple[float, float] | None:
    vals = [np.asarray(a)[np.isfinite(a)] for a in arrays if a is not None]
    vals = [v for v in vals if v.size]
    if not vals:
        return None
    lo = min(float(v.min()) for v in vals)
    hi = max(float(v.max()) for v in vals)
    if hi <= lo:
        hi = lo + max(abs(lo) * 1e-6, 1e-12)
    return lo, hi


def lookup_table(name: str) -> np.ndarray:
    """(256, 4) uint8 RGBA table of a PyQtGraph colour map."""
    return np.asarray(
        pg.colormap.get(name).getLookupTable(0.0, 1.0, 256, alpha=True),
        dtype=np.uint8,
    )


def colorize(
    image: np.ndarray, levels: tuple[float, float], lut: np.ndarray
) -> np.ndarray:
    """Map values to RGBA with fixed levels; non-finite cells get INVALID_RGBA.

    Values outside the levels are clamped to the end colours.
    """
    v = np.asarray(image, dtype=float)
    lo, hi = levels
    finite = np.isfinite(v)
    t = np.zeros(v.shape)
    t[finite] = (v[finite] - lo) / (hi - lo)
    idx = np.clip(np.floor(t * len(lut)), 0, len(lut) - 1).astype(np.intp)
    rgba = lut[idx]
    rgba[~finite] = INVALID_RGBA
    return rgba


TIMING_HELP = (
    'Timings: wall-clock time in the worker thread for the most recent job.\n'
    '  assimilate X ms: the job that added the newest observation.\n'
    '    GP: store it (+ target standardisation) and, when refit_policy says\n'
    '      so, hyperparameter training; otherwise only set_train_data. The GP\n'
    '      posterior solve is NOT included (GPyTorch builds it lazily in the\n'
    '      next query, so it counts as grid query).\n'
    '    GMRF: add_measurement + sparse update() solve (posterior mean).\n'
    '  reset+replay K obs X ms: after Undo/Reset/Load, backend reset plus\n'
    '    re-assimilation of all K stored observations (same per-backend\n'
    '    content as assimilate).\n'
    '  grid query X ms (N cells): one query on all N display cells.\n'
    '    GP: posterior solve for the current training set (grows roughly\n'
    '      cubically with the number of observations) + mean/variance.\n'
    '    GMRF: representability mask + mean and marginal variances\n'
    '      (sparse solves).\n'
    '    Scales with --grid-step (N), not only with the model size.'
)


def timing_labels(st) -> list[str]:
    """Status labels for the last assimilation job and grid query."""
    out = []
    if st.update_ms is not None:
        n = st.update_count
        if st.update_replay:
            out.append(
                f'reset+replay {n} obs {st.update_ms:.0f} ms'
                if n
                else f'reset {st.update_ms:.0f} ms'
            )
        elif n == 1:
            out.append(f'assimilate {st.update_ms:.0f} ms')
        else:
            out.append(f'assimilate {n} obs {st.update_ms:.0f} ms')
    if st.query_ms is not None:
        cells = (
            f' ({st.query_cells} cells)' if st.query_cells is not None else ''
        )
        out.append(f'grid query {st.query_ms:.0f} ms{cells}')
    return out


@dataclass(frozen=True)
class PanelLayout:
    """Pixel geometry of the map grid (see `compute_panel_layout`)."""

    map_w: int  # map width [px]
    map_h: int  # map height [px]
    x0: int  # left edge of the first column
    y0: int  # top edge of the first title
    content_w: int  # width incl. outer margins
    content_h: int  # height incl. outer margins
    min_content_w: int  # width needed at the minimum panel size

    def cell(self, row: int, col: int) -> QtCore.QRect:
        """Panel rect (title + map) of grid cell (row, col)."""
        x = self.x0 + col * (self.map_w + GAP)
        y = self.y0 + row * (TITLE_H + self.map_h + GAP)
        return QtCore.QRect(x, y, self.map_w, TITLE_H + self.map_h)


def compute_panel_layout(
    avail_w: float,
    avail_h: float,
    ncols: int,
    aspect: float,
    min_w: int = MIN_PANEL_W,
) -> PanelLayout:
    """Largest aspect-correct maps for 2 rows x ncols in (avail_w, avail_h).

    aspect = domain width / height. The maps are as large as the height and
    the width allow, but not narrower than `min_w` (unless the height forces
    it); the columns are packed with GAP between them and the block is
    centred, so tall domains do not get wide gaps between columns.
    """
    h_lim = (avail_h - 2 * MARGIN - GAP - 2 * TITLE_H) / 2.0
    w_by_h = max(h_lim * aspect, 1.0)
    w_by_cols = (avail_w - 2 * MARGIN - (ncols - 1) * GAP) / ncols
    w_min = min(min_w, w_by_h)
    w = max(min(w_by_h, w_by_cols), w_min)
    map_w = max(int(math.floor(w)), 1)
    map_h = max(int(round(map_w / aspect)), 1)
    if map_h > h_lim >= 1:
        map_h = int(math.floor(h_lim))
        map_w = max(int(round(map_h * aspect)), 1)
    content_w = 2 * MARGIN + ncols * map_w + (ncols - 1) * GAP
    content_h = 2 * MARGIN + 2 * (TITLE_H + map_h) + GAP
    min_content_w = (
        2 * MARGIN + ncols * int(math.floor(w_min)) + (ncols - 1) * GAP
    )
    x0 = MARGIN + max(0, int((avail_w - content_w) // 2))
    y0 = MARGIN + max(0, int((avail_h - content_h) // 2))
    return PanelLayout(
        map_w, map_h, x0, y0, content_w, content_h, min_content_w
    )


def _pen(spec):
    color, width = spec
    pen = pg.mkPen(color=color, width=width)
    pen.setCosmetic(True)
    return pen


def _pairs(segments) -> tuple[np.ndarray, np.ndarray]:
    seg = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    return seg[:, :, 0].ravel(), seg[:, :, 1].ravel()


class MapPanel(QtWidgets.QWidget):
    """Title strip + one aspect-locked ViewBox showing the full domain."""

    clicked = QtCore.Signal(float, float)  # world position [m]

    def __init__(self, key, title: str, domain, lut: np.ndarray, parent=None):
        super().__init__(parent)
        self.key = key
        self.base_title = title
        self.lut = lut
        self.domain = domain
        self.title_label = QtWidgets.QLabel(title, self)
        self.title_label.setObjectName('panelTitle')
        self.title_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.view = pg.GraphicsView(self, background=PANEL_BG)
        self.view.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.view.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.view.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.vb = pg.ViewBox(
            enableMouse=False,
            enableMenu=False,
            lockAspect=True,
            invertY=False,
            defaultPadding=0.0,
        )
        self.view.setCentralItem(self.vb)
        (x0, x1), (y0, y1) = domain.x_range, domain.y_range
        self.image = pg.ImageItem(axisOrder='row-major')
        self.image.setZValue(Z_IMAGE)
        self.vb.addItem(self.image)
        self._rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)
        self.vb.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0.0)
        self.vb.disableAutoRange()

        self.model_grid_item = None
        if domain.walls:
            xs, ys = _pairs(domain.walls)
            for spec, z in (
                (WALL_OUTLINE_PEN, Z_WALLS),
                (WALL_PEN, Z_WALLS + 1),
            ):
                item = pg.PlotCurveItem(xs, ys, connect='pairs', pen=_pen(spec))
                item.setZValue(z)
                self.vb.addItem(item)
        self.markers = pg.ScatterPlotItem(
            symbol='o',
            size=9,
            pen=pg.mkPen(MARKER_COLOR, width=1.6),
            brush=pg.mkBrush(None),
            pxMode=True,
        )
        self.markers.setZValue(Z_MARKERS)
        self.vb.addItem(self.markers)
        self._labels: list[pg.TextItem] = []
        self._markers_visible = True
        self.view.scene().sigMouseClicked.connect(self._on_scene_click)

    # ---- content
    def set_rgba(self, rgba: np.ndarray) -> None:
        self.image.setImage(rgba, autoLevels=False)
        self.image.setRect(self._rect)

    def set_model_grid(self, segments: np.ndarray) -> None:
        xs, ys = _pairs(segments)
        self.model_grid_item = pg.PlotCurveItem(
            xs, ys, connect='pairs', pen=_pen(MODEL_GRID_PEN)
        )
        self.model_grid_item.setZValue(Z_MODEL_GRID)
        self.vb.addItem(self.model_grid_item)

    def set_title_suffix(self, suffix_html: str) -> None:
        self._suffix_html = suffix_html
        self._update_title()

    def _update_title(self) -> None:
        """Base title + state suffix; on narrow panels the base title is
        elided so the (short) state suffix always stays visible."""
        suffix = getattr(self, '_suffix_html', '')
        fm = self.title_label.fontMetrics()
        suffix_w = (
            fm.horizontalAdvance(
                QtGui.QTextDocumentFragment.fromHtml(suffix).toPlainText()
            )
            if suffix
            else 0
        )
        avail = max(self.title_label.width() - suffix_w - 8, 0)
        base = self.base_title
        if self.title_label.width() > 0 and fm.horizontalAdvance(base) > avail:
            base = fm.elidedText(
                base, QtCore.Qt.TextElideMode.ElideRight, avail
            )
        self.title_label.setText(html.escape(base, quote=False) + suffix)
        self.title_label.setToolTip(self.base_title)

    def set_observations(self, obs) -> None:
        self.markers.setData([o.x for o in obs], [o.y for o in obs])
        while len(self._labels) > len(obs):
            self.vb.removeItem(self._labels.pop())
        for k, o in enumerate(obs):
            if k < len(self._labels):
                item = self._labels[k]
            else:
                item = pg.TextItem(color=MARKER_COLOR, anchor=(-0.15, 1.0))
                item.setFont(QtGui.QFont(self.font().family(), 7))
                item.setZValue(Z_MARKERS)
                item.setVisible(self._markers_visible)
                self.vb.addItem(item)
                self._labels.append(item)
            item.setText(str(o.index))
            item.setPos(o.x, o.y)

    def set_markers_visible(self, visible: bool) -> None:
        self._markers_visible = bool(visible)
        self.markers.setVisible(self._markers_visible)
        for item in self._labels:
            item.setVisible(self._markers_visible)

    # ---- geometry / input
    def resizeEvent(self, ev):  # noqa: N802 - Qt API
        super().resizeEvent(ev)
        w, h = self.width(), self.height()
        self.title_label.setGeometry(0, 0, w, TITLE_H)
        self.view.setGeometry(0, TITLE_H, w, max(h - TITLE_H, 1))
        self._update_title()

    def map_to_view(self, x: float, y: float) -> QtCore.QPoint:
        """Widget pixel of `self.view` showing world point (x, y)."""
        scene = self.vb.mapViewToScene(QtCore.QPointF(x, y))
        return self.view.mapFromScene(scene)

    def _on_scene_click(self, ev) -> None:
        # GraphicsScene only reports a click if the press did not turn into
        # a drag, so drags never add observations.
        if ev.button() != QtCore.Qt.MouseButton.LeftButton or ev.double():
            return
        pos = self.vb.mapSceneToView(ev.scenePos())
        self.clicked.emit(float(pos.x()), float(pos.y()))


class PanelGrid(QtWidgets.QWidget):
    """Places the map panels with fixed gaps (no Qt layout stretching)."""

    def __init__(self, n_methods: int, aspect: float, parent=None):
        super().__init__(parent)
        self.ncols = 1 + n_methods
        self.aspect = aspect
        self.cells: dict[tuple[int, int], QtWidgets.QWidget] = {}
        self.layout_info: PanelLayout | None = None
        # Height used for the layout. The window passes the scroll area's
        # height minus a horizontal scrollbar, so showing/hiding that
        # scrollbar cannot feed back into the panel size.
        self.available_height = lambda: self.height()
        self.reserved = QtWidgets.QWidget(self)  # bottom-left, left empty
        self.reserved.setObjectName('reservedCell')
        self.reserved.setAutoFillBackground(True)
        pal = self.reserved.palette()
        pal.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(PANEL_BG))
        self.reserved.setPalette(pal)
        self.cells[(1, 0)] = self.reserved

    def place(self, row: int, col: int, widget: QtWidgets.QWidget) -> None:
        widget.setParent(self)
        self.cells[(row, col)] = widget

    def relayout(self) -> None:
        lay = compute_panel_layout(
            self.width(), self.available_height(), self.ncols, self.aspect
        )
        self.layout_info = lay
        if self.minimumWidth() != lay.min_content_w:
            self.setMinimumWidth(lay.min_content_w)
        for (row, col), w in self.cells.items():
            r = lay.cell(row, col)
            if (row, col) == (
                1,
                0,
            ):  # reserved: map area only, below a title strip
                r = QtCore.QRect(
                    r.x(), r.y() + TITLE_H, r.width(), r.height() - TITLE_H
                )
            w.setGeometry(r)

    def resizeEvent(self, ev):  # noqa: N802 - Qt API
        super().resizeEvent(ev)
        self.relayout()


class PlaygroundWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        session: Session,
        out_dir: str | Path = 'belief_playground_sessions',
        poll_ms: int = 100,
        show_measurements: bool = True,
        mean_range: tuple[float, float] | None = None,
        std_max: float | None = None,
    ):
        super().__init__()
        self.session = session
        self.scenario = session.scenario
        self.domain = session.scenario.domain
        self.grid = session.grid
        self.unit = self.domain.unit
        self.out_dir = Path(out_dir)
        self.names = session.backend_names
        self.truth = self.scenario.truth(self.grid.xy)  # same grid as backends
        self._truth_img = self.grid.to_image(self.truth)
        self.fixed_mean = (
            tuple(map(float, mean_range)) if mean_range is not None else None
        )
        self.fixed_std = (0.0, float(std_max)) if std_max is not None else None
        self.mean_clim = (0.0, 1.0)
        self.std_clim = (0.0, 1.0)
        self.message = 'left-click a map to add a measurement'
        self.last_session_path: Path | None = None
        self.last_save_dir: Path | None = None
        self._last_key = None
        self._shown_result = {n: None for n in self.names}
        self._mean_img = {n: None for n in self.names}
        self._std = {n: None for n in self.names}
        self._std_img = {n: None for n in self.names}
        self._neg_var = {n: 0 for n in self.names}
        self._closed = False
        self._show_measurements = True

        self.setWindowTitle(f'belief playground - {self.scenario.name}')
        # Dark style only on the window's own parts, so that file dialogs
        # and message boxes (children of the window) keep a readable style.
        self.setStyleSheet(f'QMainWindow {{ background: {WINDOW_BG}; }}')
        (x0, x1), (y0, y1) = self.domain.x_range, self.domain.y_range
        self.panel_grid = PanelGrid(len(self.names), (x1 - x0) / (y1 - y0))
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidget(self.panel_grid)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setCentralWidget(self.scroll)
        sb_h = self.scroll.style().pixelMetric(
            QtWidgets.QStyle.PixelMetric.PM_ScrollBarExtent
        )
        self.panel_grid.available_height = lambda: max(
            self.scroll.height() - sb_h, 1
        )

        mean_lut, std_lut = lookup_table(MEAN_CMAP), lookup_table(STD_CMAP)
        self.panels: dict = {}

        def add_panel(row, col, key, title, lut):
            p = MapPanel(key, title, self.domain, lut)
            p.clicked.connect(self.on_map_click)
            self.panel_grid.place(row, col, p)
            self.panels[key] = p
            return p

        add_panel(0, 0, 'truth', 'Ground truth', mean_lut)
        for k, name in enumerate(self.names):
            backend = session.backends[name]
            title = getattr(backend, 'display_name', None) or backend.label
            add_panel(0, 1 + k, (name, 'mean'), f'{title} — estimate', mean_lut)
            add_panel(
                1,
                1 + k,
                (name, 'std'),
                f'{title} — standard deviation',
                std_lut,
            )
            # Optional adapter capability: the model's own node graph.
            grid_fn = getattr(backend, 'model_grid', None)
            overlay = grid_fn() if callable(grid_fn) else None
            if overlay is not None:
                self.panels[(name, 'mean')].set_model_grid(overlay.segments)

        self._build_toolbar()
        # Status: backends + count, then the last message. Wraps instead of
        # forcing the window wider; details are in the tooltip.
        self.status_label = QtWidgets.QLabel()
        self.status_label.setObjectName('status')
        self.status_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.status_label.setFixedHeight(
            3 * self.status_label.fontMetrics().lineSpacing() + 4
        )
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().setSizeGripEnabled(False)
        for w in (self.scroll, self.toolbar, self.statusBar()):
            w.setStyleSheet(_STYLE)

        self.show_measurements = show_measurements
        self._size_to_screen()
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(poll_ms)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh(force=True)

    # ------------------------------------------------------------- display
    @property
    def show_measurements(self) -> bool:
        """Presentation only: measurement markers and labels on all maps."""
        return self._show_measurements

    @show_measurements.setter
    def show_measurements(self, visible: bool) -> None:
        self._show_measurements = bool(visible)
        for p in self.panels.values():
            p.set_markers_visible(self._show_measurements)
        box = getattr(self, 'measurements_checkbox', None)
        if box is not None and box.isChecked() != self._show_measurements:
            with QtCore.QSignalBlocker(box):  # no feedback loop
                box.setChecked(self._show_measurements)

    def _size_to_screen(self) -> None:
        screen = self.screen() or QtGui.QGuiApplication.primaryScreen()
        avail = screen.availableGeometry()
        max_w, max_h = int(avail.width() * 0.96), int(avail.height() * 0.92)
        sb_h = self.scroll.style().pixelMetric(
            QtWidgets.QStyle.PixelMetric.PM_ScrollBarExtent
        )
        chrome = (
            self.toolbar.sizeHint().height()
            + self.statusBar().sizeHint().height()
            + sb_h
            + 8
        )
        lay = compute_panel_layout(
            max_w, max_h - chrome, self.panel_grid.ncols, self.panel_grid.aspect
        )
        # Shrink to the packed maps (wide domains do not need the full
        # height); keep room for the status line when maps are narrow.
        width = min(max_w, max(lay.content_w + 4, 900))
        height = min(max_h, lay.content_h + chrome)
        self.resize(width, height)
        frame = self.frameGeometry()
        frame.moveCenter(avail.center())
        self.move(frame.topLeft())

    def _build_toolbar(self) -> None:
        tb = QtWidgets.QToolBar('Actions', self)
        tb.setObjectName('actions')
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, tb)
        self.toolbar = tb
        self.actions_by_name = {}
        for label, shortcut, slot in (
            ('Undo', 'Ctrl+Z', self.undo),
            ('Reset', 'Ctrl+R', self.reset),
            ('Save session', 'Ctrl+S', self.save_session),
            ('Load session', 'Ctrl+O', self.load_session_dialog),
            ('Export image', 'Ctrl+E', self.export_image),
        ):
            act = QtGui.QAction(label, self)
            act.setShortcut(QtGui.QKeySequence(shortcut))
            act.setToolTip(f'{label} ({shortcut})')
            act.triggered.connect(lambda _checked=False, f=slot: f())
            tb.addAction(act)
            self.actions_by_name[label] = act
        # Right-aligned display option (presentation only, no session work).
        spacer = QtWidgets.QWidget(tb)
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        tb.addWidget(spacer)
        box = QtWidgets.QCheckBox('Show measurement positions', tb)
        box.setChecked(True)
        box.setToolTip(
            'Show or hide the measurement markers and their numbers on all maps '
            '(display only)'
        )
        box.toggled.connect(
            lambda checked: setattr(self, 'show_measurements', checked)
        )
        tb.addWidget(box)
        self.measurements_checkbox = box
        quit_act = QtGui.QAction(
            'Quit', self
        )  # shortcut only, not in the toolbar
        quit_act.setShortcut(QtGui.QKeySequence('Ctrl+Q'))
        quit_act.triggered.connect(self.close)
        self.addAction(quit_act)
        self.actions_by_name['Quit'] = quit_act

    # ------------------------------------------------------------- actions
    def undo(self) -> None:
        removed = self.session.undo()
        self._set_message(
            'nothing to undo'
            if removed is None
            else f'undid #{removed.index}; replaying '
            f'{len(self.session.observations)} stored values'
        )

    def reset(self) -> None:
        self.session.reset()
        self._set_message(f'reset (seed {self.session.seed})')

    def _timestamp(self) -> str:
        return datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    # Dialog hooks: small methods so tests can replace them (no modal
    # dialogs in automated runs).
    def _ask_save_path(
        self, title: str, suggested: Path, filters: str
    ) -> tuple[str, str] | None:
        """Save dialog; returns (chosen path, selected filter) or None if
        cancelled. Qt's own overwrite prompt is off: `_confirm_overwrite`
        also covers the .json."""
        path, selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            title,
            str(suggested),
            filters,
            options=QtWidgets.QFileDialog.Option.DontConfirmOverwrite,
        )
        return (path, selected) if path else None

    def _ask_open_path(
        self, title: str, start: Path, filters: str
    ) -> str | None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, title, str(start), filters
        )
        return path or None

    def _confirm_overwrite(self, existing: list[Path]) -> bool:
        names = '\n'.join(str(p) for p in existing)
        answer = QtWidgets.QMessageBox.question(
            self,
            'Overwrite?',
            f'These files already exist and will be replaced:\n{names}',
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def _dialog_dir(self, prefer_last_save: bool = False) -> Path:
        """Start directory for dialogs (never created here): --out-dir, else
        its nearest existing parent, else cwd. The load dialog prefers the
        directory of the last saved session."""
        candidates = [self.last_save_dir] if prefer_last_save else []
        for d in candidates + [self.out_dir]:
            if d is not None and Path(d).is_dir():
                return Path(d)
        for parent in Path(self.out_dir).absolute().parents:
            if parent.is_dir():
                return parent
        return Path.cwd()

    def save_session(self, path: str | Path | None = None) -> Path | None:
        """Save the log (CSV) and metadata (JSON with the same name).

        Without `path` (toolbar, Ctrl+S): a save dialog, `.csv` appended if
        missing, and a confirmation if the .csv or the .json already exists.
        With `path` (scripts/tests): written as given, without dialog or
        overwrite confirmation, like `Session.save`.
        """
        if path is None:
            chosen = self._ask_save_path(
                'Save session',
                self._dialog_dir() / f'session_{self._timestamp()}.csv',
                'Session CSV (*.csv)',
            )
            if chosen is None:
                self._set_message('save cancelled')
                return None
            path = Path(chosen[0])
            if path.suffix.lower() != '.csv':
                path = path.with_name(path.name + '.csv')
            existing = [
                p for p in (path, path.with_suffix('.json')) if p.exists()
            ]
            if existing and not self._confirm_overwrite(existing):
                self._set_message('save cancelled (existing files kept)')
                return None
        try:
            csv_path, json_path = self.session.save(
                path, extra={'display': self._display_record()}
            )
        except Exception as exc:  # noqa: BLE001
            self._set_message(f'save failed: {exc}')
            return None
        self.last_session_path = csv_path
        self.last_save_dir = csv_path.parent
        self._set_message(f'saved {csv_path} (+ {json_path.name})')
        return csv_path

    def load_session_dialog(self) -> None:
        path = self._ask_open_path(
            'Load session',
            self._dialog_dir(prefer_last_save=True),
            'Session CSV (*.csv)',
        )
        if path:
            self.load_session(path)

    def load_session(self, path: str | Path) -> bool:
        try:
            warnings = self.session.load(path)
        except Exception as exc:  # noqa: BLE001
            self._set_message(f'load failed: {exc}')
            return False
        n = len(self.session.observations)
        self._set_message(
            f'loaded {n} observations from {Path(path).name}, replaying'
            + ('; WARNING: ' + '; '.join(warnings) if warnings else '')
        )
        return True

    def export_image(self, path: str | Path | None = None) -> Path | None:
        """PNG/JPEG of the window. Without `path`: save dialog (a name
        without .png/.jpg/.jpeg gets the suffix of the selected file type)
        with overwrite confirmation. With `path`: written as given (parent
        created), no dialog/confirmation."""
        if path is None:
            chosen = self._ask_save_path(
                'Export image',
                self._dialog_dir() / f'figure_{self._timestamp()}.png',
                'PNG image (*.png);;JPEG image (*.jpg *.jpeg)',
            )
            if chosen is None:
                self._set_message('export cancelled')
                return None
            path, selected = Path(chosen[0]), chosen[1] or ''
            if path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                suffix = '.jpg' if selected.startswith('JPEG') else '.png'
                path = path.with_name(path.name + suffix)
            if path.exists() and not self._confirm_overwrite([path]):
                self._set_message('export cancelled (existing file kept)')
                return None
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
        ok = self.grab().save(str(path))
        self._set_message(
            f'exported {path} (an image alone is not a saved session)'
            if ok
            else f'export failed: {path}'
        )
        return path if ok else None

    def _display_record(self) -> dict:
        _, status, _, _ = self.session.poll()
        return {
            'backend_status_at_save': {
                n: {
                    'state': st.state,
                    'n_assimilated': st.n_assimilated,
                    'maps_observation_count': st.result_count,
                    'error': st.error,
                    'failed_at': st.failed_at,
                    'query_error': st.query_error,
                    'last_assimilation_ms': st.update_ms,
                    'last_assimilation_observations': st.update_count,
                    'last_assimilation_was_reset_replay': st.update_replay,
                    'last_grid_query_ms': st.query_ms,
                    'last_grid_query_cells': st.query_cells,
                }
                for n, st in status.items()
            },
            'auto_color_limits': self.fixed_mean is None
            and self.fixed_std is None,
            'mean_clim': list(self.mean_clim),
            'std_clim': list(self.std_clim),
            'show_measurements': self.show_measurements,
        }

    def _set_message(self, text: str) -> None:
        self.message = text
        self.refresh(force=True)

    # --------------------------------------------------------------- clicks
    def on_map_click(self, x: float, y: float) -> None:
        """Add one sampled observation at a left click on any map."""
        if not self.domain.contains(x, y):
            self._set_message('click outside the domain ignored')
            return
        try:
            obs = self.session.add_sample(x, y)
        except ValueError as exc:
            self._set_message(f'not added: {exc}')
            return
        extra = f' (truth {obs.latent:.6g}, noise {obs.noise:+.3g}' + (
            ', clipped)' if obs.clipped else ')'
        )
        self._set_message(
            f'#{obs.index} at ({obs.x:.3f}, {obs.y:.3f}) m: '
            f'{obs.value:.6g} {self.unit}{extra}'
        )

    # -------------------------------------------------------------- refresh
    def refresh(self, force: bool = False) -> None:
        if self._closed:
            return
        version, status, pending, gen = self.session.poll()
        obs = self.session.observations
        # `pending` too: the worker publishes its last result before it
        # decrements the job count, so the busy text must refresh on that.
        key = (version, gen, pending, tuple((o.index, o.x, o.y) for o in obs))
        if key == self._last_key and not force:
            return
        self._last_key = key
        n_obs = len(obs)

        for name in self.names:
            st = status[name]
            if self._shown_result[name] != (st.generation, st.result_id):
                self._shown_result[name] = (st.generation, st.result_id)
                if st.mean is None:
                    self._mean_img[name] = None
                    self._std[name] = None
                    self._std_img[name] = None
                    self._neg_var[name] = 0
                else:
                    self._mean_img[name] = self.grid.to_image(st.mean)
                    std, self._neg_var[name] = variance_to_std(st.var)
                    self._std[name] = std
                    self._std_img[name] = self.grid.to_image(std)
            suffix = self._title_suffix(st, n_obs)
            self.panels[(name, 'mean')].set_title_suffix(suffix)
            self.panels[(name, 'std')].set_title_suffix(suffix)

        # Shared levels: one for truth + all estimates, one for all stds.
        if self.fixed_mean is not None:
            self.mean_clim = self.fixed_mean
        else:
            self.mean_clim = _finite_range(
                [self.truth] + [status[n].mean for n in self.names]
            ) or (0.0, 1.0)
        if self.fixed_std is not None:
            self.std_clim = self.fixed_std
        else:
            r = _finite_range([s for s in self._std.values() if s is not None])
            self.std_clim = (0.0, max(r[1], 1e-12)) if r else (0.0, 1.0)

        blank = np.full(self.grid.shape, np.nan)
        truth_panel = self.panels['truth']
        truth_panel.set_rgba(
            colorize(self._truth_img, self.mean_clim, truth_panel.lut)
        )
        for name in self.names:
            for kind, img, clim in (
                ('mean', self._mean_img[name], self.mean_clim),
                ('std', self._std_img[name], self.std_clim),
            ):
                p = self.panels[(name, kind)]
                p.set_rgba(colorize(blank if img is None else img, clim, p.lut))

        for p in self.panels.values():
            p.set_observations(obs)
        self._refresh_status(status, pending, obs)

    @staticmethod
    def _title_suffix(st, n_obs: int) -> str:
        # Short: panels can be narrow; the status bar has the details.
        if st.error is not None:
            return (
                f' <span style="color:#ff7043">· STOPPED n={st.n_assimilated}'
                '</span>'
            )
        if st.mean is None:
            note = 'waiting' if st.state == 'waiting' else st.state
            return f' <span style="color:#9e9e9e">· {note}</span>'
        if st.result_count != n_obs:
            return f' <span style="color:#ffca28">· n={st.result_count}/{n_obs}</span>'
        return ''

    def _refresh_status(self, status, pending, obs) -> None:
        u = self.unit
        rmse, n_valid = rmse_on_common_grid(
            self.truth, {n: status[n].mean for n in self.names}
        )
        parts, details = [], []
        for name in self.names:
            st = status[name]
            backend = self.session.backends[name]
            title = getattr(backend, 'display_name', None) or backend.label
            n_text = f'{st.n_assimilated}/{len(obs)}'
            if st.error is not None:
                state = (
                    f'<span style="color:#ff7043">STOPPED n={n_text}'
                    + (f' (#{st.failed_at} failed)' if st.failed_at else '')
                    + '</span>'
                )
            elif st.state == 'waiting':
                state = (
                    f'<span style="color:#9e9e9e">{st.note or "waiting"}</span>'
                )
                state += ''.join(f' · {t}' for t in timing_labels(st))
            else:
                state = f'{st.state} n={n_text}'
                state += ''.join(f' · {t}' for t in timing_labels(st))
                if name in rmse and math.isfinite(rmse[name]):
                    state += f' · RMSE {rmse[name]:.3g} {u}'
            if st.query_error:
                state += ' <span style="color:#ff7043">query error</span>'
            parts.append(f'<b>{title}</b>: {state}')

            details.append(f'{backend.label}: {st.state}')
            details.append(
                f'  assimilated {n_text}'
                + (
                    f', maps n={st.result_count}'
                    if st.result_count is not None
                    else ''
                )
            )
            details.extend(f'  {t}' for t in timing_labels(st))
            if st.error:
                details.append(f'  ERROR: {st.error}')
            if st.query_error:
                details.append(f'  {st.query_error}')
            hidden = getattr(backend, 'hidden_cells', 0)
            if st.mean is not None and hidden:
                details.append(f'  {hidden} cells not representable (grey)')
            if self._neg_var[name]:
                details.append(
                    f'  WARNING: {self._neg_var[name]} cells with negative variance'
                )
        details.append(f'RMSE grid: {n_valid}/{len(self.grid.xy)} cells')
        details.append(
            f'estimate scale [{self.mean_clim[0]:.4g}, {self.mean_clim[1]:.4g}] {u}'
            + (' (fixed)' if self.fixed_mean else ' (shared, auto)')
        )
        details.append(
            f'std scale [{self.std_clim[0]:.4g}, {self.std_clim[1]:.4g}] {u}'
            + (' (fixed)' if self.fixed_std else ' (shared, auto)')
        )
        busy = f' · worker: {pending} job(s)' if pending else ''
        msg = html.escape(self.message)
        self.status_label.setText(
            f'{len(obs)} observations{busy} &nbsp;|&nbsp; '
            + ' &nbsp;|&nbsp; '.join(parts)
            + f'<br><span style="color:#9e9e9e">{msg}</span>'
        )
        self.status_label.setToolTip('\n'.join(details) + '\n\n' + TIMING_HELP)

    # ---------------------------------------------------------------- close
    def closeEvent(self, ev):  # noqa: N802 - Qt API
        self.shutdown()
        super().closeEvent(ev)

    def shutdown(self) -> None:
        """Stop polling and the session worker (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self.timer.stop()
        self.session.close()


_STYLE = f"""
QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {WINDOW_BG}; }}
QLabel {{ color: {TEXT_COLOR}; }}
QLabel#panelTitle {{ font-size: 11px; padding: 0px 2px; }}
QToolBar {{ background: {WINDOW_BG}; border: none; spacing: 3px; padding: 2px 6px; }}
QToolButton {{ color: {TEXT_COLOR}; background: #262626; border: 1px solid #383838;
              border-radius: 3px; padding: 2px 8px; }}
QToolButton:hover {{ background: #333333; }}
QCheckBox {{ color: {TEXT_COLOR}; padding: 0px 6px; }}
QToolButton:pressed {{ background: #444444; }}
QStatusBar {{ background: {WINDOW_BG}; color: {TEXT_COLOR}; }}
QStatusBar QLabel {{ padding: 0px 6px; }}
QToolTip {{ color: #e0e0e0; background: #202020; border: 1px solid #444444;
           font-family: monospace; }}
QScrollBar:horizontal {{ background: {WINDOW_BG}; height: 10px; }}
QScrollBar::handle:horizontal {{ background: #444444; border-radius: 4px; }}
"""


def ensure_app() -> QtWidgets.QApplication:
    """The QApplication (created on first use)."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(['belief_playground'])
        app.setApplicationName('belief_playground')
    return app


def run_app(session: Session, args) -> int:
    """Open the window for `session`, run the Qt event loop, return its code."""
    app = ensure_app()
    win = PlaygroundWindow(
        session,
        out_dir=args.out_dir,
        show_measurements=not args.hide_measurements,
        mean_range=args.mean_range,
        std_max=args.std_max,
    )
    if args.load is not None:
        win.load_session(args.load)
    win.show()
    # Ctrl+C: quit the event loop cleanly. Qt's loop does not run Python
    # code by itself, so a short timer lets the signal handler run.
    previous = signal.signal(signal.SIGINT, lambda *_: app.quit())
    wake = QtCore.QTimer()
    wake.timeout.connect(lambda: None)
    wake.start(200)
    try:
        return int(app.exec())
    finally:
        wake.stop()
        signal.signal(signal.SIGINT, previous)
        win.shutdown()
