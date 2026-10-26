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

"""PySide6 + PyQtGraph window of the live viewer (no rclpy).

Reuses the playground's map panels, layout, colour maps and style. A QTimer
on the Qt main thread polls the `LiveState` filled by the ROS thread and
redraws only when its version changed; the ROS thread never touches Qt.

The estimator's method name (from `LiveState.method`, or a fixed
`method_label`) is shown in the panel titles, the window title and at the
start of the status line.

Layout (one estimator)::

    Ground truth        | <method> - estimate
    (reserved, empty)   | <method> - standard deviation

Every map shows the walls, the measurement positions (red circles), the
travelled trajectory (thin line) and the current position (filled dot).
Clicks on the maps do nothing. Only the pose stream can become stale: the
estimators publish their clouds only after new measurements.
"""

from __future__ import annotations

import dataclasses
import html
import signal
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from belief_playground.gui import (
    _STYLE,
    MEAN_CMAP,
    STD_CMAP,
    WINDOW_BG,
    Z_MARKERS,
    Z_WALLS,
    MapPanel,
    PanelGrid,
    _finite_range,
    _pen,
    colorize,
    compute_panel_layout,
    lookup_table,
    variance_to_std,
)
from belief_playground.live_data import (
    DEFAULT_METHOD,
    STREAMS,
    LiveState,
    lattice_to_image,
)

TRAJECTORY_PEN = '#4fc3f7', 1.2
POSITION_COLOR = '#ffeb3b'
Z_TRAJECTORY = Z_WALLS + 5  # above the walls, below the measurements
Z_POSITION = Z_MARKERS + 5

GREY, AMBER, RED = '#9e9e9e', '#ffca28', '#ff7043'


def _span(color: str, text: str) -> str:
    return f'<span style="color:{color}">{html.escape(text)}</span>'


def format_age(seconds: float) -> str:
    if seconds < 10.0:
        return f'{seconds:.1f} s'
    if seconds < 120.0:
        return f'{seconds:.0f} s'
    return f'{seconds / 60.0:.1f} min'


class LiveViewerWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        state: LiveState,
        domain,
        grid,
        truth=None,
        *,
        title: str = '',
        method_label: str | None = None,
        mean_range: tuple[float, float] | None = None,
        std_max: float | None = None,
        show_measurements: bool = True,
        show_trajectory: bool = True,
        stale_after: float = 10.0,
        poll_ms: int = 100,
        domain_source: str = '',
        frame_source: str = '',
        domain_warnings=(),
        shutdown_hook=None,
    ):
        """`truth`: values on `grid.xy` (None: not available).
        `domain_source`/`frame_source` describe where the domain and the
        frame come from, `domain_warnings` problems found on the way
        (status tooltip). `shutdown_hook` is called once when the window
        closes."""
        super().__init__()
        self.state = state
        self.domain = domain
        self.grid = grid
        self.unit = domain.unit
        self.truth = None if truth is None else np.asarray(truth, dtype=float)
        self._truth_img = (
            None if self.truth is None else grid.to_image(self.truth)
        )
        self.method_label = method_label  # overrides the detected method
        self.title = title
        self.method_shown = None
        self.domain_source = domain_source
        self.frame_source = frame_source
        self.domain_warnings = tuple(domain_warnings)
        self.stale_after = float(stale_after)
        self.fixed_mean = (
            tuple(map(float, mean_range)) if mean_range is not None else None
        )
        self.fixed_std = (0.0, float(std_max)) if std_max is not None else None
        self.mean_clim = (0.0, 1.0)
        self.std_clim = (0.0, 1.0)
        self.message = ''
        self._shutdown_hook = shutdown_hook
        self._closed = False
        self._last_version = None
        self._seen = {}
        self._snap = state.snapshot()
        self._mean_img = None
        self._std_img = None
        self._neg_var = 0
        self._grid_note = None
        self._shown_text: dict = {}
        self._show_measurements = True
        self._show_trajectory = True

        self.setStyleSheet(f'QMainWindow {{ background: {WINDOW_BG}; }}')
        (x0, x1), (y0, y1) = domain.x_range, domain.y_range
        self.panel_grid = PanelGrid(1, (x1 - x0) / (y1 - y0))
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
        self.panels: dict[str, MapPanel] = {}
        self.trajectory_items: list[pg.PlotCurveItem] = []
        self.tail_items: list[pg.PlotCurveItem] = []
        self.position_items: list[pg.ScatterPlotItem] = []
        for row, col, key, text, lut in (
            (0, 0, 'truth', 'Ground truth', mean_lut),
            (0, 1, 'mean', '', mean_lut),
            (1, 1, 'std', '', std_lut),
        ):
            p = MapPanel(key, text, domain, lut)  # clicks are not connected
            self.panel_grid.place(row, col, p)
            self.panels[key] = p
            self._add_overlays(p)
        if self.truth is None:
            self.panels['truth'].set_title_suffix(
                ' ' + _span(GREY, '· not available')
            )

        self._build_toolbar()
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
        self.show_trajectory = show_trajectory
        self._size_to_screen()
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(poll_ms)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh(force=True)

    def _add_overlays(self, p: MapPanel) -> None:
        # Stored trajectory, and the segment from its last point to the
        # current position (updated on every pose without redrawing all).
        traj = pg.PlotCurveItem(pen=_pen(TRAJECTORY_PEN), skipFiniteCheck=True)
        tail = pg.PlotCurveItem(pen=_pen(TRAJECTORY_PEN))
        for item in (traj, tail):
            item.setZValue(Z_TRAJECTORY)
            p.vb.addItem(item)
        pos = pg.ScatterPlotItem(
            symbol='o',
            size=9,
            pen=pg.mkPen('#000000', width=1.2),
            brush=pg.mkBrush(POSITION_COLOR),
            pxMode=True,
        )
        pos.setZValue(Z_POSITION)
        p.vb.addItem(pos)
        self.trajectory_items.append(traj)
        self.tail_items.append(tail)
        self.position_items.append(pos)

    # ------------------------------------------------------------- display
    @property
    def show_measurements(self) -> bool:
        return self._show_measurements

    @show_measurements.setter
    def show_measurements(self, visible: bool) -> None:
        self._show_measurements = bool(visible)
        for p in self.panels.values():
            p.set_markers_visible(self._show_measurements)
        self._sync_box(self.measurements_checkbox, self._show_measurements)

    @property
    def show_trajectory(self) -> bool:
        return self._show_trajectory

    @show_trajectory.setter
    def show_trajectory(self, visible: bool) -> None:
        self._show_trajectory = bool(visible)
        for item in self.trajectory_items + self.tail_items:
            item.setVisible(self._show_trajectory)
        self._sync_box(self.trajectory_checkbox, self._show_trajectory)

    @staticmethod
    def _sync_box(box: QtWidgets.QCheckBox, checked: bool) -> None:
        if box.isChecked() != checked:
            with QtCore.QSignalBlocker(box):  # no feedback loop
                box.setChecked(checked)

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
            ('Clear trajectory', None, self.clear_trajectory),
            ('Clear measurements', None, self.clear_measurements),
            ('Export image', 'Ctrl+E', self.export_image),
        ):
            act = QtGui.QAction(label, self)
            if shortcut:
                act.setShortcut(QtGui.QKeySequence(shortcut))
                act.setToolTip(f'{label} ({shortcut})')
            act.triggered.connect(lambda _checked=False, f=slot: f())
            tb.addAction(act)
            self.actions_by_name[label] = act
        spacer = QtWidgets.QWidget(tb)
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        tb.addWidget(spacer)
        self.measurements_checkbox = QtWidgets.QCheckBox(
            'Show measurement positions', tb
        )
        self.measurements_checkbox.setChecked(True)
        self.measurements_checkbox.setToolTip(
            'Show or hide the received measurement positions on all maps'
        )
        self.measurements_checkbox.toggled.connect(
            lambda checked: setattr(self, 'show_measurements', checked)
        )
        tb.addWidget(self.measurements_checkbox)
        self.trajectory_checkbox = QtWidgets.QCheckBox('Show trajectory', tb)
        self.trajectory_checkbox.setChecked(True)
        self.trajectory_checkbox.setToolTip(
            'Show or hide the travelled trajectory on all maps'
        )
        self.trajectory_checkbox.toggled.connect(
            lambda checked: setattr(self, 'show_trajectory', checked)
        )
        tb.addWidget(self.trajectory_checkbox)
        quit_act = QtGui.QAction('Quit', self)  # shortcut only
        quit_act.setShortcut(QtGui.QKeySequence('Ctrl+Q'))
        quit_act.triggered.connect(self.close)
        self.addAction(quit_act)
        self.actions_by_name['Quit'] = quit_act

    # ------------------------------------------------------------- actions
    def clear_trajectory(self) -> None:
        self.state.clear_trajectory()
        self._set_message('trajectory cleared')

    def clear_measurements(self) -> None:
        """Remove the shown measurement positions (display only)."""
        self.state.clear_measurements()
        self._set_message('measurement positions cleared')

    # Dialog hooks: small methods so tests can replace them.
    def _ask_save_path(
        self, title: str, suggested: Path, filters: str
    ) -> tuple[str, str] | None:
        path, selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            title,
            str(suggested),
            filters,
            options=QtWidgets.QFileDialog.Option.DontConfirmOverwrite,
        )
        return (path, selected) if path else None

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

    def export_image(self, path: str | Path | None = None) -> Path | None:
        """PNG/JPEG of the window. Without `path`: save dialog in the
        current directory (`.png` added unless the name ends in .png/.jpg/
        .jpeg) with overwrite confirmation. With `path`: written as given
        (parent created)."""
        if path is None:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            chosen = self._ask_save_path(
                'Export image',
                Path.cwd() / f'live_{stamp}.png',
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
            f'exported {path}' if ok else f'export failed: {path}'
        )
        return path if ok else None

    def _set_message(self, text: str) -> None:
        self.message = text
        self.refresh(force=True)

    # -------------------------------------------------------------- refresh
    def _changed(self, part: str, version: int) -> bool:
        if self._seen.get(part) == version:
            return False
        self._seen[part] = version
        return True

    def refresh(self, force: bool = False) -> None:
        if self._closed:
            return
        snap = self._snap = self.state.snapshot(since=self._snap)
        if snap.version != self._last_version or force:
            self._last_version = snap.version
            self._update_method(snap)
            self._update_maps(snap, force)
            self._update_overlays(snap, force)
        self._refresh_status(self.state.clock())

    def _update_method(self, snap) -> None:
        """Panel and window titles with the current method name."""
        name = self.method_label or snap.method.name or DEFAULT_METHOD
        if name == self.method_shown:
            return
        self.method_shown = name
        for key, kind in (('mean', 'estimate'), ('std', 'standard deviation')):
            panel = self.panels[key]
            panel.base_title = f'{name} — {kind}'
            # Re-render with the current state suffix.
            panel.set_title_suffix(self._shown_text.get(('title', key), ''))
        self.setWindowTitle(
            ' - '.join(t for t in ('belief live viewer', self.title, name) if t)
        )

    def _method_status(self, snap) -> tuple[str, list[str], list[str]]:
        """Status part, notes and tooltip lines about the method."""
        if self.method_label:
            return f'<b>{html.escape(self.method_label)}</b>', [], []
        m = snap.method
        part = f'<b>{html.escape(m.name)}</b>'
        if m.details:
            part += f' ({html.escape(m.details)})'
        if m.node:
            part += ' · ' + _span(GREY, m.node)
        notes = [_span(GREY, m.note)] if m.note else []
        details = [
            f'method: {m.name}'
            + (f' ({m.details})' if m.details else '')
            + (f' from {m.node}' if m.node else '')
        ]
        if m.note:
            details.append(f'  {m.note}')
        return part, notes, details

    def _update_maps(self, snap, force: bool) -> None:
        changed = force
        if self._changed('mean', snap.mean_version):
            changed = True
            m = snap.mean
            self._mean_img = (
                None if m is None else lattice_to_image(m.lattice, self.grid)
            )
        if self._changed('variance', snap.variance_version):
            changed = True
            v = snap.variance
            if v is None:
                self._std_img, self._neg_var = None, 0
            else:
                std, self._neg_var = variance_to_std(v.lattice.values)
                self._std_img = lattice_to_image(
                    dataclasses.replace(v.lattice, values=std), self.grid
                )
        if not changed:
            return
        self._grid_note = self._grid_mismatch(snap.mean, snap.variance)
        if self.fixed_mean is not None:
            self.mean_clim = self.fixed_mean
        else:
            self.mean_clim = _finite_range(
                [self._truth_img, self._mean_img]
            ) or (0.0, 1.0)
        if self.fixed_std is not None:
            self.std_clim = self.fixed_std
        else:
            r = _finite_range([self._std_img])
            self.std_clim = (0.0, max(r[1], 1e-12)) if r else (0.0, 1.0)
        blank = np.full(self.grid.shape, np.nan)
        for key, img, clim in (
            ('truth', self._truth_img, self.mean_clim),
            ('mean', self._mean_img, self.mean_clim),
            ('std', self._std_img, self.std_clim),
        ):
            p = self.panels[key]
            p.set_rgba(colorize(blank if img is None else img, clim, p.lut))

    @staticmethod
    def _grid_mismatch(mean, var) -> str | None:
        if mean is None or var is None:
            return None
        if mean.xy.shape == var.xy.shape and np.array_equal(mean.xy, var.xy):
            return None
        return (
            'mean and variance clouds have different points '
            f'({len(mean.xy)} vs {len(var.xy)})'
        )

    def _update_overlays(self, snap, force: bool) -> None:
        traj = snap.trajectory
        new_traj = self._changed('trajectory', snap.trajectory_version)
        if new_traj or force:
            for item in self.trajectory_items:
                item.setData(traj[:, 0], traj[:, 1])
        if self._changed('pose', snap.pose_version) or new_traj or force:
            pose = snap.pose
            pos = ([pose.x], [pose.y]) if pose is not None else ([], [])
            tail = ([], [])
            if pose is not None and len(traj):
                tail = ([traj[-1, 0], pose.x], [traj[-1, 1], pose.y])
            for item in self.position_items:
                item.setData(*pos)
            for item in self.tail_items:
                item.setData(*tail)
        if self._changed('measurements', snap.measurements_version) or force:
            m = snap.measurements
            for p in self.panels.values():
                p.markers.setData(m[:, 0], m[:, 1])

    # --------------------------------------------------------------- status
    def _stream_state(self, name: str, st, now: float) -> tuple[str, str]:
        """('off'|'waiting'|'stale'|'ok', age text). Only poses can be
        stale; clouds and measurements arrive only when there is news."""
        if st.topic is None:
            return 'off', ''
        if st.last_received is None:
            return 'waiting', ''
        age = max(now - st.last_received, 0.0)
        stale = name == 'pose' and age > self.stale_after
        return ('stale' if stale else 'ok'), format_age(age)

    def _title_suffix(self, name: str, st, now: float) -> str:
        state, _ = self._stream_state(name, st, now)
        if state == 'waiting':
            return ' ' + _span(GREY, '· waiting')
        return ''

    def _set_if_changed(self, key, setter, text: str) -> None:
        """Rebuilding labels is not free; only touch them on changes."""
        if self._shown_text.get(key) != text:
            self._shown_text[key] = text
            setter(text)

    def _stream_status(
        self, name: str, st, now: float
    ) -> tuple[str, str | None, list[str]]:
        """(status bar part, error note or None, tooltip lines)."""
        state, age = self._stream_state(name, st, now)
        if state == 'off':
            text = _span(GREY, 'off')
        elif state == 'waiting':
            text = _span(GREY, f'waiting for {st.topic}')
        else:
            text = f'{st.count}, {age} ago'
            if state == 'stale':
                text += ' ' + _span(AMBER, 'stale')
        details = [f'{name}: {st.topic or "off"}', f'  {st.count} received']
        if st.last_received is not None:
            details[-1] += f', last {format_age(now - st.last_received)} ago'
        if st.frame_mismatch:
            text += ' ' + _span(
                RED,
                f'· {st.frame_mismatch} ignored (frame {st.last_bad_frame!r})',
            )
            details.append(
                f'  {st.frame_mismatch} ignored: frame '
                f'{st.last_bad_frame!r} != {self.state.frame_id!r}'
            )
        note = None
        if st.errors:
            text += ' ' + _span(RED, f'· {st.errors} errors')
            note = _span(RED, st.last_error or '')
            details.append(f'  {st.errors} errors, last: {st.last_error}')
        return f'<b>{name}</b>: {text}', note, details

    @staticmethod
    def _ros_status(snap) -> tuple[list[str], list[str], list[str]]:
        """Status parts, notes and tooltip lines about the ROS thread."""
        parts, notes, details = [], [], []
        if not snap.ros_alive:
            reason = f'ROS STOPPED: {snap.ros_stopped_reason}'
            parts.append('<b>' + _span(RED, reason) + '</b>')
            details.append(reason)
        if snap.ros_errors:
            text = f'{snap.ros_errors} ROS errors, last: {snap.ros_last_error}'
            notes.append(_span(RED, text))
            details.append(text)
        return parts, notes, details

    def _refresh_status(self, now: float) -> None:
        snap = self._snap
        streams = snap.streams
        for key, stream in (('mean', 'mean'), ('std', 'variance')):
            self._set_if_changed(
                ('title', key),
                self.panels[key].set_title_suffix,
                self._title_suffix(stream, streams[stream], now),
            )
        parts, notes, details = self._ros_status(snap)
        method_part, method_notes, method_details = self._method_status(snap)
        parts.insert(0, method_part)
        notes[:0] = method_notes
        details[:0] = [
            *method_details,
            f'domain: {self.domain_source or "given"}',
            f'frame: {self.state.frame_id}'
            + (f' ({self.frame_source})' if self.frame_source else ''),
            *(f'WARNING: {w}' for w in self.domain_warnings),
        ]
        for name in STREAMS:
            part, note, lines = self._stream_status(name, streams[name], now)
            parts.append(part)
            if note is not None:
                notes.append(note)
            details.extend(lines)
        if self._grid_note:
            notes.append(_span(AMBER, self._grid_note))
        info = (
            f'trajectory {len(snap.trajectory)} points · '
            f'{len(snap.measurements)} measurements'
        )
        if snap.pose is not None:
            info += f' · position ({snap.pose.x:.2f}, {snap.pose.y:.2f}) m'
        notes.append(_span(GREY, info))
        if self.message:
            notes.append(_span(GREY, self.message))
        if self._neg_var:
            details.append(
                f'WARNING: {self._neg_var} variance points are negative'
            )
        if snap.trajectory_dropped or snap.measurements_dropped:
            details.append(
                f'oldest points dropped: trajectory '
                f'{snap.trajectory_dropped}, measurements '
                f'{snap.measurements_dropped}'
            )
        details.append(
            f'estimate scale [{self.mean_clim[0]:.4g}, '
            f'{self.mean_clim[1]:.4g}] {self.unit}'
            + (' (fixed)' if self.fixed_mean else ' (shared, auto)')
        )
        details.append(
            f'std scale [{self.std_clim[0]:.4g}, {self.std_clim[1]:.4g}] '
            f'{self.unit}' + (' (fixed)' if self.fixed_std else ' (auto)')
        )
        self._set_if_changed(
            'status',
            self.status_label.setText,
            ' &nbsp;|&nbsp; '.join(parts) + '<br>' + ' · '.join(notes),
        )
        self._set_if_changed(
            'tooltip', self.status_label.setToolTip, '\n'.join(details)
        )

    # ---------------------------------------------------------------- close
    def closeEvent(self, ev):  # noqa: N802 - Qt API
        self.shutdown()
        super().closeEvent(ev)

    def shutdown(self) -> None:
        """Stop polling and run the shutdown hook (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self.timer.stop()
        hook, self._shutdown_hook = self._shutdown_hook, None
        if hook is not None:
            hook()


def ensure_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(['belief_live_viewer'])
        app.setApplicationName('belief_live_viewer')
    return app


def run_live_app(**window_kwargs) -> int:
    """Open a `LiveViewerWindow`, run the Qt event loop, return its code."""
    app = ensure_app()
    win = LiveViewerWindow(**window_kwargs)
    win.show()
    # Ctrl+C: quit the event loop cleanly; the timer lets Python run the
    # signal handler while Qt's loop is waiting.
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
