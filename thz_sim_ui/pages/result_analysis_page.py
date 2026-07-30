from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QGridLayout, QLabel, QTabWidget, QWidget

from thz_sim_ui.data.mock_data import RECENT_TASKS
from thz_sim_ui.widgets.common import PlaceholderList
from thz_sim_ui.widgets.forms import combo, make_form_group
from thz_sim_ui.widgets.workbench import WorkbenchPage


class ProjectImageCard(QWidget):
    def __init__(self, title: str, image_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.image_name = image_name
        layout = QGridLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        header = QLabel(title)
        header.setStyleSheet("font-weight: bold; font-size: 11px;")
        layout.addWidget(header, 0, 0)
        self.image_label = QLabel("等待加载图表...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setWordWrap(True)
        self.image_label.setMinimumHeight(280)
        self.image_label.setStyleSheet("background: #FAFBFE; border: 1px solid #E6ECF6;")
        layout.addWidget(self.image_label, 1, 0)
        self._original_pixmap: QPixmap | None = None

    def set_image_path(self, path: Path | None) -> None:
        if path is None or not path.exists():
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText(f"未找到：{self.image_name}")
            self._original_pixmap = None
            return
        try:
            pixmap = QPixmap(str(path))
            if pixmap.isNull():
                self.image_label.setText(f"图表生成中：{self.image_name}")
                return
            self._original_pixmap = pixmap
            self.image_label.setText("")
            self._update_pixmap()
        except Exception:
            self.image_label.setText(f"加载失败：{self.image_name}")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._original_pixmap is None:
            return
        sz = self.image_label.size()
        if sz.width() <= 0 or sz.height() <= 0:
            return
        scaled = self._original_pixmap.scaled(sz, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)


class ResultAnalysisPage(WorkbenchPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("结果分析", "查看单方案仿真结果。", parent)

        # ── 左侧：仅保留任务选择 ──
        self.task = combo([])
        self._known_task_names: list[str] = []
        self._refresh_task_options()
        self.task.currentTextChanged.connect(self.load_task_results)

        self.add_left_widget(make_form_group("结果筛选", [
            ("任务", self.task),
        ]))
        self.add_left_widget(PlaceholderList("说明", [
            "仅显示已完成仿真任务",
            "任务选择后自动加载图表",
            "图表来自工程目录 transmitter/channel/receiver 子文件夹",
        ]))
        self.add_left_stretch()

        # ── 发射端结果 ──
        self.tx_time = ProjectImageCard("时域包络", "transmitter_time.png")
        self.tx_spectrum = ProjectImageCard("频谱图", "transmitter_spectrum.png")
        self.tx_papr = ProjectImageCard("PAPR CCDF", "transmitter_papr.png")
        self.tx_histogram = ProjectImageCard("幅度分布", "transmitter_histogram.png")

        # ── 信道结果 ──
        ch_items = [
            ("RX时域波形", "channel_time.png"),
            ("RX频谱", "channel_spectrum.png"),
            ("RX星座图", "channel_constellation.png"),
            ("RX功率分布", "channel_power.png"),
        ]
        self._ch_cards = [ProjectImageCard(t, f) for t, f in ch_items]

        # ── 接收端结果 ──
        rx_items = [
            ("匹配滤波频谱", "rx_mf_spectrum.png"),
            ("均衡后星座图", "rx_eq_constellation.png"),
        ]
        self._rx_cards = [ProjectImageCard(t, f) for t, f in rx_items]

        # ── 三 Tab ──
        tabs = QTabWidget()
        tabs.addTab(self._grid_tab([self.tx_time, self.tx_spectrum,
                                     self.tx_papr, self.tx_histogram]), "发射端结果")
        tabs.addTab(self._grid_tab(self._ch_cards), "信道结果")
        tabs.addTab(self._grid_tab(self._rx_cards, cols=1), "接收端结果")
        self.add_right_widget(tabs)

        self._all_cards = [self.tx_time, self.tx_spectrum,
                           self.tx_papr, self.tx_histogram] + self._ch_cards + self._rx_cards

        self.load_last_completed_task_results()

    # ── 布局 ──
    def _grid_tab(self, cards: list[ProjectImageCard], cols: int = 2) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        for i, card in enumerate(cards):
            layout.addWidget(card, i // cols, i % cols)
        return panel

    # ── 任务列表刷新 ──
    def _refresh_task_options(self, force: bool = False) -> None:
        completed = [t for t in RECENT_TASKS if t.status == "已完成"]
        names = [t.name for t in completed] or ["TASK-001"]
        if names == self._known_task_names and not force:
            return
        self._known_task_names = names
        cur = self.task.currentText()
        self.task.clear()
        self.task.addItems(names)
        if cur in names:
            self.task.setCurrentText(cur)
        elif self.task.count() > 0:
            self.task.setCurrentIndex(0)

    def load_last_completed_task_results(self) -> None:
        self.load_task_results()

    def load_task_results(self) -> None:
        folder = self._get_project_folder()
        if folder is None:
            for c in self._all_cards:
                c.set_image_path(None)
            return
        tx_dir = folder / "transmitter"
        ch_dir = folder / "channel"
        rx_dir = folder / "receiver"

        self.tx_time.set_image_path(tx_dir / self.tx_time.image_name if tx_dir.exists() else None)
        self.tx_spectrum.set_image_path(tx_dir / self.tx_spectrum.image_name if tx_dir.exists() else None)
        self.tx_papr.set_image_path(tx_dir / self.tx_papr.image_name if tx_dir.exists() else None)
        self.tx_histogram.set_image_path(tx_dir / self.tx_histogram.image_name if tx_dir.exists() else None)

        for c in self._ch_cards:
            c.set_image_path(ch_dir / c.image_name if ch_dir.exists() else None)
        for c in self._rx_cards:
            c.set_image_path(rx_dir / c.image_name if rx_dir.exists() else None)

    def _get_project_folder(self) -> Path | None:
        name = self.task.currentText()
        completed = [t for t in RECENT_TASKS if t.status == "已完成"]
        task = next((t for t in completed if t.name == name), None)
        if task and task.result_path:
            folder = Path(task.result_path)
            if folder.exists():
                return folder
        window = self.window()
        if hasattr(window, "current_project_folder") and window.current_project_folder:
            folder = Path(window.current_project_folder)
            if folder.exists():
                return folder
        return None

    def refresh_from_tasks(self, tasks: list) -> None:
        self._refresh_task_options(force=True)

    def get_all_parameters(self) -> dict[str, object]:
        return {"任务": self.task.currentText()}
